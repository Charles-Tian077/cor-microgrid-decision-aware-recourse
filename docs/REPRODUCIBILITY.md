# Reproduction protocol

## Levels of evidence

1. **Integrity and numerical reconciliation:** `run.py verify` checks archived files and independently recomputes claims. It does not solve a new dispatch problem.
2. **Frozen-policy regeneration:** `run.py replay` re-solves all four test policies from input data. Saved cost totals are not injected into the optimizer.
3. **Forecast and selection regeneration:** `run.py expanded` refits both tree forecasters, replays all 13 validation configurations and evaluates the frozen selection. By default it uses the archived GRU predictions so that neural backend differences are separated from controller reproducibility.
4. **End-to-end retraining:** `run.py train` retrains all three GRU seeds, after which `run.py expanded --predictions runs/training` propagates the new predictions to the same policy-selection protocol.
5. **Continuation and stress:** `run.py initial` and `run.py stress` reproduce the distinct 334-day continuation and the five paired deterministic perturbations.

Every entry point writes to a new output directory. Never set an output path under `data/`, `results/reference/` or `code/original/`. Source hashes verify that these copies agree with the author's supplied files.

## Environment

Use CPython 3.11, NumPy 1.26.4, pandas 2.3.3, SciPy 1.14.1 and scikit-learn 1.2.2. `environment/requirements-tested.lock.txt` records the actual environment used for archive verification. The original completed recurrent metadata reports PyTorch 2.6.0+cu118 executing on a CPU. The portable CPU retraining command uses the official PyTorch 2.6.0 CPU wheel; CPU/GPU libraries can change final floating-point neural outputs. No pending cluster job is counted as completed.

The original GPU/cluster environment recorded pandas 2.3.3, whereas an earlier replay requirements file named pandas 2.2.3. The package fixes one tested portable environment and compares its numerical outputs explicitly rather than silently treating the requirements file as proof of a historical environment.

## Hyperparameters and stopping

| Component | Fixed configuration |
|---|---|
| HGB | 200 iterations; learning rate 0.08; 31 leaves; minimum leaf 30; L2=1; seed 20261018; no internal early stopping |
| HGB-28 | 250 iterations; learning rate 0.06; remaining tree settings as above |
| GRU | Univariate 28-day same-slot history; one layer, hidden size 64 |
| Dense head | 64+16 inputs → 96 → 48 → 1; SiLU; dropout 0.1 |
| Neural optimizer | AdamW; learning rate 0.001; weight decay 0.001; batch 512; gradient clip 1 |
| Training loss | Smooth-L1, beta=0.5 on standardized targets |
| Random seeds | 17, 29, 43; one ensemble averaged across seeds |
| Early stopping | At most 30 epochs; after at least 6 epochs, stop at 5 stale epochs; improvement threshold 1e-5 CNY/kWh |
| Test refit | Fixed best epoch count selected on development/validation; all archived seeds selected 2 |
| Scaling | Fit-only mean/std; standardized predictor clipping to [-10,10] |
| Forecast floor | 0.05 CNY/kWh for decision predictions only |
| Price correction | Median of up to 6 preceding observed residuals; 4-hour decay |
| Reserve | Positive historical 0.20 empirical quantile correction; no formal chance-constraint guarantee |
| LP | Two lexicographic stages; HiGHS auto method; 60 s/stage; primal/dual tolerance 1e-7 |

The full reserve and baseline-price construction are represented by the supplied release inputs. Reproducing the article's active policy starts from that verified export; rebuilding the upstream competition models is a separate procedure. The unchanged upstream source is retained for provenance, but it is not the portable entry point for the current sequential comparison.

## Metrics and statistical inference

Forecast metrics use 13,248 common slots per validation/test partition. Operational metrics use actual daily accounting: original-plan charge plus directional adjustment charges plus five-times-price emergency charges. Emergency days require daily emergency energy above 1e-6 kWh. Report total-cost savings and controllable-cost savings separately.

For the main mean-cost interval, use paired daily differences, a circular seven-day moving-block bootstrap, 4,000 replications and seed 20261020. The Wilcoxon signed-rank test is exploratory. The older initial report contains a 2,000-replication bootstrap with another seed; it is preserved in `results/reference/initial` and must not be substituted for the expanded comparison's interval.

## Comparison tolerance

Energy and cost comparisons require absolute tolerances of 1e-4 kWh and 1e-4 CNY, respectively, with a 1e-10 relative tolerance. Timing fields are excluded. Stored forecast error summaries are compared at 1e-10 when the same fitted family and software versions are used. `--kind retrained` retains the same policy cost/energy tolerances and permits 1e-6 CNY/kWh for forecast-error summaries, reflecting explicitly measured neural backend effects. Retraining comparison reports mean and maximum prediction differences rather than falsely promising bitwise identity across numerical backends.

## Important interpretation boundaries

The test months were inspected during earlier development, so the current protocol is retrospective. The 334-day continuation includes development and validation, and its state is not reset at split boundaries. In contrast, the expanded validation and test blocks each start from the inherited 1200-kWh state and then remain sequential. Their validation totals can therefore differ without a contradiction.

The supplied scenario-grid benchmark has a different offline execution protocol from the active current-period physical replay. Auxiliary table counts do not increase the independent number of sites, years or test dates. Stress perturbations reuse the same dates and released information rather than generating new observed data.


## MATLAB figure verification

Both unchanged MATLAB plotting sources were executed successfully through `visualization/reproduce_figures.m`. The wrapper creates its output and figure directories before export, keeping original sources and reference artifacts unchanged. All five regenerated vector PDFs have identical page-content streams and identical rendered pixels at 144 dpi to the manuscript figures; PDF timestamps differ. The 900-dpi PNGs and editable FIGs were also produced. The actual per-figure SHA-256 and PNG dimensions are recorded in `validation/summary.json`. An earlier licensing-service failure occurred before execution; it is superseded by this completed verification. An active MATLAB installation and license are needed to regenerate publication figures; Python numerical, training and inference checks do not require MATLAB.
