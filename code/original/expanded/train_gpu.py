"""Causal day-ahead price forecasting on one CUDA GPU.

For slot t of day d, the recurrent input contains prices from the same slot
on days d-28,...,d-1 only. Validation chooses the epoch count; the test model
is refit on development+validation without looking at test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ["month", "weekday", "is_weekend", "is_holiday", "is_makeup_workday",
            "sin_slot", "cos_slot", "sin_doy", "cos_doy", "price_fixed_yuan_per_kwh",
            "price_lag_1d", "price_lag_7d", "price_lag_28d",
            "price_same_slot_mean_7d", "price_same_slot_mean_28d", "price_same_slot_std_28d"]


class PriceGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_features, 96), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1)
        )

    def forward(self, history: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(history)
        return self.head(torch.cat((h[-1], known), dim=1)).squeeze(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arrays(data: Path):
    obs_path = data / "03_观测_10min.csv.gz"
    ml_path = data / "10_ML训练表_10min.csv.gz"
    obs = pd.read_csv(obs_path, usecols=["date", "slot_index", "price_yuan_per_kwh"])
    ml = pd.read_csv(ml_path, usecols=["date", "slot_index", "split", "price_yuan_per_kwh"] + FEATURES)
    obs = obs.sort_values(["date", "slot_index"])
    ml = ml.sort_values(["date", "slot_index"]).reset_index(drop=True)
    if len(obs) != 365 * 144 or len(ml) != 334 * 144:
        raise ValueError("Unexpected time coverage")
    if obs.duplicated(["date", "slot_index"]).any() or ml.duplicated(["date", "slot_index"]).any():
        raise ValueError("Duplicate time key")
    obs_dates = np.sort(obs.date.unique())
    if len(obs_dates) != 365 or not np.all(np.diff(pd.to_datetime(obs_dates).values).astype("timedelta64[D]") == np.timedelta64(1, "D")):
        raise ValueError("Observation calendar is not daily and complete")
    day_index = {d: i for i, d in enumerate(obs_dates)}
    day = ml.date.map(day_index).to_numpy(int)
    slot = ml.slot_index.to_numpy(int) - 1
    if day.min() < 28 or slot.min() != 0 or slot.max() != 143:
        raise ValueError("Missing 28-day causal history or invalid slot")
    past = obs.price_yuan_per_kwh.to_numpy(float).reshape(365, 144)
    offsets = np.arange(28, 0, -1, dtype=int)
    history = past[day[:, None] - offsets[None, :], slot[:, None]]
    known = ml[FEATURES].to_numpy(float)
    target = ml.price_yuan_per_kwh.to_numpy(float)
    if not np.isfinite(history).all() or not np.isfinite(known).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite model input")
    for name in ("development", "validation", "test"):
        if not (ml.split == name).any():
            raise ValueError(f"Missing split {name}")
    # The maximum date in every historical input precedes the prediction date.
    assert np.all(day[:, None] - offsets[None, :] < day[:, None])
    return ml, history.astype(np.float32), known.astype(np.float32), target.astype(np.float32), {
        obs_path.name: sha256(obs_path), ml_path.name: sha256(ml_path)
    }


def normalize(history, known, target, train_idx):
    y_mu = float(target[train_idx].mean())
    y_sd = float(target[train_idx].std())
    x_mu = known[train_idx].mean(axis=0)
    x_sd = known[train_idx].std(axis=0)
    x_sd = np.maximum(x_sd, 1e-6)
    y_sd = max(y_sd, 1e-6)
    hist_n = np.clip((history - y_mu) / y_sd, -10, 10)
    known_n = np.clip((known - x_mu) / x_sd, -10, 10)
    target_n = (target - y_mu) / y_sd
    scale = dict(target_mean=y_mu, target_std=y_sd,
                 feature_mean=x_mu.tolist(), feature_std=x_sd.tolist())
    return hist_n, known_n, target_n, scale


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_one(history, known, target, train_idx, valid_idx, *, seed, epochs,
            batch_size, device, fixed_epochs=None):
    seed_all(seed)
    hn, kn, yn, scale = normalize(history, known, target, train_idx)
    ht = torch.from_numpy(hn[:, :, None]).to(device)
    kt = torch.from_numpy(kn).to(device)
    yt = torch.from_numpy(yn).to(device)
    model = PriceGRU(known.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    valid_t = torch.as_tensor(valid_idx, dtype=torch.long, device=device)
    best_mae, best_epoch, best_state, stale = float("inf"), 0, None, 0
    n_epochs = fixed_epochs if fixed_epochs is not None else epochs
    history_log = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = train_t[torch.randperm(len(train_t), device=device)]
        running = 0.0
        for ids in order.split(batch_size):
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(ht[ids], kt[ids]), yt[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(ht[ids], kt[ids]) for ids in valid_t.split(4096)])
            mae = float((pred - yt[valid_t]).abs().mean()) * scale["target_std"]
        history_log.append(dict(epoch=epoch, train_loss=running / len(train_t), validation_mae=mae))
        print(f"seed={seed} epoch={epoch} train_loss={running/len(train_t):.5f} val_mae={mae:.6f}", flush=True)
        if fixed_epochs is None:
            if mae < best_mae - 1e-5:
                best_mae, best_epoch, stale = mae, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch >= 6 and stale >= 5:
                break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
        best_mae = history_log[-1]["validation_mae"]
    model.eval()
    with torch.no_grad():
        predictions = torch.cat([model(h, k) for h, k in zip(ht.split(4096), kt.split(4096))]).cpu().numpy()
    predictions = np.maximum(predictions * scale["target_std"] + scale["target_mean"], 0.05)
    return model, predictions, scale, history_log, best_epoch, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--allow-cpu", action="store_true", help="diagnostic fallback only")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required for this experiment")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ml, hist, known, y, hashes = read_arrays(args.data)
    masks = {s: np.flatnonzero(ml.split.to_numpy() == s) for s in ("development", "validation", "test")}
    devval = np.concatenate((masks["development"], masks["validation"]))
    val_preds, test_preds, metadata = [], [], []
    all_logs = []
    for seed in args.seeds:
        model, pred, scale, log, n_epoch, val_mae = fit_one(
            hist, known, y, masks["development"], masks["validation"],
            seed=seed, epochs=args.epochs, batch_size=args.batch_size, device=device)
        val_preds.append(pred[masks["validation"]])
        torch.save({"state": model.state_dict(), "scale": scale, "features": FEATURES,
                    "train_split": "development", "seed": seed, "selected_epoch": n_epoch},
                   args.out / f"gru_dev_seed{seed}.pt")
        all_logs.extend(dict(seed=seed, stage="development", **x) for x in log)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, pred, scale, log, _, _ = fit_one(
            hist, known, y, devval, masks["validation"], seed=seed,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
            fixed_epochs=n_epoch)
        test_preds.append(pred[masks["test"]])
        torch.save({"state": model.state_dict(), "scale": scale, "features": FEATURES,
                    "train_split": "development+validation", "seed": seed, "epochs": n_epoch},
                   args.out / f"gru_devval_seed{seed}.pt")
        all_logs.extend(dict(seed=seed, stage="development+validation", **x) for x in log)
        metadata.append(dict(seed=seed, selected_epoch=n_epoch, validation_mae=val_mae))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = ml[["date", "slot_index", "split", "price_yuan_per_kwh"]].copy()
    out["gru_price"] = np.nan
    out.loc[masks["validation"], "gru_price"] = np.mean(val_preds, axis=0)
    out.loc[masks["test"], "gru_price"] = np.mean(test_preds, axis=0)
    out["ensemble_std"] = np.nan
    out.loc[masks["validation"], "ensemble_std"] = np.std(val_preds, axis=0)
    out.loc[masks["test"], "ensemble_std"] = np.std(test_preds, axis=0)
    if not np.isfinite(out.loc[ml.split != "development", "gru_price"]).all():
        raise AssertionError("Missing GRU evaluation predictions")
    out.to_csv(args.out / "gru_predictions.csv", index=False)
    pd.DataFrame(all_logs).to_csv(args.out / "gru_training_history.csv", index=False)
    manifest = dict(gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU diagnostic fallback", torch_version=torch.__version__,
                    cuda_version=torch.version.cuda, seeds=args.seeds,
                    training_days=150, validation_days=92, test_days=92,
                    selected_models=metadata, input_sha256=hashes,
                    seconds=time.perf_counter() - t0)
    (args.out / "gru_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
