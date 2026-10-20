# Neural model artifacts

The reference GRU architecture and training procedure are fully contained in `code/original/expanded/train_gpu.py`. The original three development and three development+validation checkpoint files were not present in the local supplied archive. Their recorded predictions, training history, seeds, selected epochs and input checksums are preserved in `results/reference/expanded/`.

`python run.py train --allow-cpu --out runs/training` regenerates six `.pt` checkpoints and their prediction tables. If reconstructed checkpoints are distributed under `models/regenerated_cpu/`, they are new executions of the unchanged source. Their comparison to the original saved predictions is documented under `validation/`; they are not presented as the missing original checkpoint bytes.
