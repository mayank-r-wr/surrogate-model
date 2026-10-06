# Surrogate Machine Learning Models for Estimating Daily Infiltration in India

Code accompanying the manuscript **"A Surrogate Models for Estimating Daily Infiltration in India"**
(Raturi, Patidar & Khare; *Journal of Hydrology*, manuscript HYDROL83109 — under review).

## Overview

Daily rainfall totals hide the sub-daily intensity that controls infiltration capacity and saturation excess,
while hourly physically based modelling is expensive at national scale. This repository implements a
**surrogate framework**:

1. Hourly ERA5 precipitation + soil hydraulic parameters (NBSS soil texture, Rawls et al. 1983 / Carsel & Parrish 1988)
   drive the **Green–Ampt** model to simulate hourly infiltration over India.
2. Hourly infiltration is aggregated to daily values.
3. **Linear Regression, Random Forest, XGBoost and LightGBM** are trained to predict that daily infiltration
   from **daily rainfall only** (plus lagged rainfall and soil properties).
4. Models are evaluated on a temporal hold-out (R², RMSE, NSE, KGE) and used for annual spatial maps.

**Headline result:** LightGBM R² = 0.963, RMSE = 0.943 mm (test period 2021 onward).

## Repository structure

```
├── src/
│   ├── config.py                       # Paths (env-overridable) and random seed
│   ├── 00_download_era5.py             # Download ERA5 hourly precipitation (CDS API)
│   ├── 01_green_ampt_infiltration.py   # Green–Ampt (Newton solver) on ERA5 grid -> NetCDF
│   ├── 02_xgboost_baseline.py          # Early XGBoost baseline (lat/lon + rainfall features)
│   ├── 03_multi_model_comparison.py    # LR / RF / XGB / LGBM, metrics, violin + Taylor plots
│   ├── 04_scatter_plots.py             # 2x2 scatter plots vs Green–Ampt with R2/RMSE/NSE/KGE
│   └── 05_spatial_maps.py              # Annual rainfall & infiltration maps clipped to India
├── archive/                            # Earlier script versions kept for reference
├── data/                               # Put input data here (not tracked by git)
├── figures/                            # Output figures
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/surrogate-infiltration-india.git
cd surrogate-infiltration-india
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Input files are large and are **not** stored in this repository. See [`data/README.md`](data/README.md) for the full file list and how to obtain each one.

| Data | Source | Used by |
|---|---|---|
| ERA5 hourly total precipitation (`tp`), 0.25° | [Copernicus CDS](https://cds.climate.copernicus.eu/) | 01, 03, 05 |
| Soil texture raster (NBSS&LUP), ~0.1° | National Bureau of Soil Survey & Land Use Planning | 03 |
| India boundary shapefile | user-supplied | 05 |
| Green–Ampt daily infiltration NetCDF | produced by step 1 / archived at `<Zenodo DOI>` | 02, 03, 05 |

## Usage

All paths are set in `src/config.py` (inputs in `data/`, derived files in `outputs/`, figures in `figures/`).
To use other folders without editing code:

```bash
export INFIL_DATA_DIR=/path/to/inputs      # Windows (PowerShell): $env:INFIL_DATA_DIR="D:\inputs"
export INFIL_OUTPUT_DIR=/path/to/outputs
```

Expected input file names in the data folder: `ERA5_IND.nc`, `Infiltration_IND.nc`, `soil_raster_IND.nc`,
`era5_2024.nc`, `infiltration_2024.nc`, `india1.shp` (plus `.shx/.dbf/.prj`).

Run in order:

```bash
python src/00_download_era5.py --start 2012 --end 2025   # optional: fetch ERA5 (needs CDS account)
python src/01_green_ampt_infiltration.py --nc data/ERA5_precipitation.nc --year 2014 --K 1.09 --psi 110 --dtheta 0.247
python src/03_multi_model_comparison.py    # trains LR/RF/XGB/LGBM, writes outputs/model_predictions_test.csv
python src/04_scatter_plots.py             # scatter figure from saved predictions
python src/05_spatial_maps.py              # annual maps
```

**Reproducibility:** a single seed (`SEED = 42` in `config.py`) fixes plot subsampling and all model `random_state`s.

## Green–Ampt parameters (Table 1 of the paper)

| Soil class | Ks (mm/h) | ψf (mm) | θe |
|---|---|---|---|
| Sandy | 25 | 60 | 0.30 |
| Loamy | 12 | 110 | 0.30 |
| Clayey | 3 | 210 | 0.25 |
| Loamy skeletal | 10 | 100 | 0.25 |
| Clay skeletal | 2 | 220 | 0.20 |
| Marshy | 5 | 50 | 0.20 |

## Citation

See `CITATION.cff`. Please cite the article once published.

## Contact

Nitesh Patidar — npatidar.nih@gov.in
Mayank Raturi — IIT Roorkee

## License

MIT — see `LICENSE`.
