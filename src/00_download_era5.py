"""Download ERA5 hourly total precipitation for India, one file per year.

Requires the Copernicus CDS API: `pip install cdsapi` and a ~/.cdsapirc file
(https://cds.climate.copernicus.eu/how-to-api).

Usage:
    python src/00_download_era5.py --start 2012 --end 2025
Files are saved as data/ERA5_tp_<year>.nc. Merge them afterwards if you want a
single ERA5_IND.nc, e.g.:
    python -c "import xarray as xr; xr.open_mfdataset('data/ERA5_tp_*.nc').to_netcdf('data/ERA5_IND.nc')"
"""
import argparse
import cdsapi
from config import DATA_DIR

AREA = [38, 68, 6, 98]  # N, W, S, E  (covers India)

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)
a = ap.parse_args()

c = cdsapi.Client()
for year in range(a.start, a.end + 1):
    target = DATA_DIR / f"ERA5_tp_{year}.nc"
    if target.exists():
        print("exists, skipping:", target)
        continue
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "total_precipitation",
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
        },
        str(target),
    )
