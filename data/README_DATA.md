# Data folder

Place the CSV training/evaluation files here (not tracked in Git):

- `samples_angle_fftma.csv` — ~5 GB (angles + FFT-MA, row-wise concatenation, Used to train conditional generative adversarial networks and angle regression models)
- `samples_angle_fftma_repeat.csv` — ~629 MB (Contains 60 angle images, each corresponding to 20 independent FFT-MA implementations, used to test the three methods of DSS-LA, FFT-MA and cGAN.)

**Schema per row (float values):**
1) 121×121 Angle map (flattened) — 14,641 numbers
2) 121×121 FFT-MA realization (flattened) — 14,641 numbers

Total per row: 29,282 values.

## How to obtain the files

- These large files are not distributed with the repository.  
- Depending on journal policy, they are provided as:   
  - **Kaggle dataset**  Link: https://www.kaggle.com/datasets/zfxssg/varying-anisotropy/data

Please check the **main README** or supplementary materials for the actual download link provided with the paper submission.

---

## Usage Instructions

1. Download the required CSVs from the provided link.  
2. Place them into this `data/` folder, with the exact filenames.
