# ======================================================
# YEARLY SPATIAL MAPS WITH SHAPE-CLIPPED GRID POINTS
# ======================================================

from config import DATA_DIR, FIG_DIR
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

plt.rcParams["font.family"] = "serif"

# ------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------

rain_ds = xr.open_dataset(str(DATA_DIR / "era5_2024.nc"))
inf_ds  = xr.open_dataset(str(DATA_DIR / "infiltration_2024.nc"))

rain = rain_ds["tp"]
inf  = inf_ds["infiltration_rate"]

# ------------------------------------------------------
# 2. LOAD SHAPEFILE
# ------------------------------------------------------

boundary = gpd.read_file(str(DATA_DIR / "india1.shp"))
boundary = boundary.to_crs("EPSG:4326")

# ------------------------------------------------------
# 3. COMPUTE ANNUAL TOTALS
# ------------------------------------------------------

rain_year = rain.groupby("valid_time.year").sum()
inf_year  = inf.groupby("time.year").sum()

years = rain_year.year.values
print("Years available:", years)

# ------------------------------------------------------
# 4. CREATE 0.25° GRID
# ------------------------------------------------------

lon_min, lon_max = rain.longitude.min().values, rain.longitude.max().values
lat_min, lat_max = rain.latitude.min().values, rain.latitude.max().values

lon_grid = np.arange(lon_min, lon_max + 0.25, 0.25)
lat_grid = np.arange(lat_min, lat_max + 0.25, 0.25)

lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

# ------------------------------------------------------
# 5. CLIP GRID POINTS TO SHAPEFILE AREA
# ------------------------------------------------------

points = []
for i in range(lon2d.shape[0]):
    for j in range(lon2d.shape[1]):
        points.append(Point(lon2d[i, j], lat2d[i, j]))

points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# Spatial join → keep only points inside polygon
points_clipped = gpd.sjoin(points_gdf, boundary, predicate="within")

# Extract clipped coordinates
clipped_lon = points_clipped.geometry.x.values
clipped_lat = points_clipped.geometry.y.values

# ------------------------------------------------------
# 6. LOOP THROUGH YEARS AND PLOT
# ------------------------------------------------------

for yr in years:

    rain_map = rain_year.sel(year=yr)
    inf_map  = inf_year.sel(year=yr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------------- Rainfall ----------------
    im1 = axes[0].pcolormesh(
        rain_map.longitude,
        rain_map.latitude,
        rain_map,
        shading="auto",
        cmap="Blues"
    )

    # Plot clipped grid points
    axes[0].scatter(
        clipped_lon,
        clipped_lat,
        marker=".",
        s=1,
        color="black",
        alpha=1.0,
        zorder=3
    )

    boundary.boundary.plot(ax=axes[0], color="black", linewidth=0.8)

    axes[0].set_title(f"(a) Annual Rainfall - {yr}", fontsize=13)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Rainfall (mm/year)")

    # ---------------- Infiltration ----------------
    im2 = axes[1].pcolormesh(
        inf_map.longitude,
        inf_map.latitude,
        inf_map,
        shading="auto",
        cmap="YlGnBu"
    )

    axes[1].scatter(
        clipped_lon,
        clipped_lat,
        marker=".",
        s=1,
        color="black",
        alpha=1.0,
        zorder=3
    )

    boundary.boundary.plot(ax=axes[1], color="black", linewidth=0.8)

    axes[1].set_title(f"(b) Annual Infiltration - {yr}", fontsize=13)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Infiltration (mm/year)")

    plt.tight_layout()

    plt.savefig(
        str(FIG_DIR / f"Spatial_Map_{yr}.png"),
        dpi=1200,
        bbox_inches="tight"
    )

    plt.close()

print("All yearly maps saved successfully.")
