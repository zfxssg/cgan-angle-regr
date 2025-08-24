
# ResCGAN + Angle Regression (Paper Artifact)

This repository contains training and testing code for:
- **ResCGAN**: conditional generator mapping *angle maps* → *simulation realizations*.
- **Angle Regression Net**: CNN mapping *simulation realizations* → *angle maps*.
- **Testing/Visualization**: comparison among DSS‑LA, FFT‑MA, and Generator outputs.

## Files
- `train_rescgan.py` – train the conditional GAN.  
- `train_angle_regression.py` – train the angle regression network.  
- `test_trained_model.py` – load trained models and reproduce the figures/tables for the paper.  
- `requirements.txt` – Python dependencies.  
- `artifacts/` – recommended output folder for `.pth` weights.
- `results/` – recommended output folder for generated figures and CSVs.

## Expected Dataset Format
All training scripts expect a CSV where each row concatenates two 121×121 single‑channel images:
1. **Angle map** (radians), flattened (121*121 values).
2. **FFT‑MA realization**, flattened (121*121 values).

Thus each row has 29,282 numbers.

## Quickstart

```bash
# 1) Setup environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train ResCGAN
python train_rescgan.py --csv /path/to/samples_angle_fftma.csv --epochs 100 --batch 16 --out artifacts

# 3) Train Angle Regression
python train_angle_regression.py --csv /path/to/samples_angle_fftma.csv --epochs 100 --batch 16 --out artifacts/Angle_regression.pth

# 4) Test & Reproduce Figures
python test_trained_model.py --csv /path/to/samples_angle_fftma_repeat.csv     --gen artifacts/Generator.pth     --angle_reg artifacts/Angle_regression.pth     --indices 121 241 281 301 341     --single_idx 241     --outdir results
```

## Using Your Pretrained Weights
Place your provided weights here:
```
artifacts/Generator.pth
artifacts/Angle_regression.pth
```
Then run the testing script as shown above.

## Reproducibility Notes
- Multi‑GPU training is supported via `nn.DataParallel` when multiple GPUs are visible.
- Weight loading strips the optional `'module.'` prefix for DataParallel checkpoints.


## License
MIT (feel free to change to your journal’s preferred license).
