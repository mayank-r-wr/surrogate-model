# ==========================================================
# FULL CORRECTED GREEN-AMPT SPATIAL INFILTRATION CODE
# Fixed:
# ✔ ERA5 tp cumulative handling
# ✔ NaN handling
# ✔ Dynamic lat/lon/time names
# ✔ Newton convergence
# ✔ Better initialization
# ✔ Dry period reset
# ✔ No blank outputs
# ✔ NetCDF export
# ==========================================================

from config import DATA_DIR, OUTPUT_DIR
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import newton
from math import log
from datetime import datetime


# ==========================================================
# GREEN AMPT MODEL
# ==========================================================
class GreenAmpt:

    def __init__(self, K, dt, dtheta, psi, rainfall):
        self.K = K
        self.dt = dt
        self.dtheta = dtheta
        self.psi = psi
        self.i = rainfall

    def Fp(self, i):
        if i <= self.K:
            return np.inf
        return self.K * self.psi * self.dtheta / (i - self.K)

    def equation(self, F_t, dt_t, F):
        try:
            return (
                F - F_t
                - self.K * dt_t
                - self.psi * self.dtheta *
                log((self.psi * self.dtheta + F) /
                    (self.psi * self.dtheta + F_t))
            )
        except:
            return np.inf

    def solve_F(self, F_t, dt_t):
        func = lambda F: self.equation(F_t, dt_t, F)

        try:
            return newton(func, max(F_t + 0.1, 0.1), maxiter=100)
        except:
            return F_t + self.K * dt_t

    def infil_rate(self, F):
        if F <= 0:
            return self.K
        return self.K * ((self.psi * self.dtheta / F) + 1)

    def run(self):

        n = len(self.i)

        F_all = np.zeros(n + 1)
        f_all = np.zeros(n + 1)
        t_all = np.arange(n + 1)

        F_all[0] = 0
        f_all[0] = 0

        dry_hours = 0

        for t in range(1, n + 1):

            rain = self.i[t - 1]

            F_prev = F_all[t - 1]

            # ----------------------------------
            # RESET after dry period (6 hrs)
            # ----------------------------------
            if rain == 0:
                dry_hours += 1
            else:
                dry_hours = 0

            if dry_hours >= 6:
                F_prev = 0

            # ----------------------------------
            # No rain
            # ----------------------------------
            if rain == 0:
                F_new = F_prev
                f_new = 0

            # ----------------------------------
            # Rainfall <= Ks
            # ----------------------------------
            elif rain <= self.K:
                F_new = F_prev + rain * self.dt
                f_new = rain

            # ----------------------------------
            # Rainfall > Ks
            # ----------------------------------
            else:
                F_new = self.solve_F(F_prev, self.dt)
                f_new = min(self.infil_rate(F_new), rain)

            F_all[t] = F_new
            f_all[t] = f_new

        return F_all, f_all, t_all


# ==========================================================
# PROCESS YEAR
# ==========================================================
def process_year(
    nc_file,
    output_file,
    year=2024,
    K=1.09,
    psi=110,
    dtheta=0.247,
    dt=1
):

    print("Opening dataset...")
    ds = xr.open_dataset(nc_file)

    print(ds)

    # ------------------------------------------
    # rainfall variable
    # ------------------------------------------
    var = "tp"

    da = ds[var]

    # ------------------------------------------
    # Dynamic dimensions
    # ------------------------------------------
    time_dim = [d for d in da.dims if "time" in d.lower()][0]
    lat_dim = [d for d in da.dims if "lat" in d.lower()][0]
    lon_dim = [d for d in da.dims if "lon" in d.lower()][0]

    print("Time:", time_dim)
    print("Lat :", lat_dim)
    print("Lon :", lon_dim)

    # ------------------------------------------
    # Filter year
    # ------------------------------------------
    time_vals = pd.to_datetime(da[time_dim].values)

    mask = time_vals.year == year

    da = da.isel({time_dim: mask})

    time_vals = pd.to_datetime(da[time_dim].values)

    lats = da[lat_dim].values
    lons = da[lon_dim].values

    nt = len(time_vals)
    ny = len(lats)
    nx = len(lons)

    print(f"\nProcessing {year}")
    print("Time steps:", nt)
    print("Grid:", ny, "x", nx)

    # ------------------------------------------
    # Output arrays
    # ------------------------------------------
    cumulative = np.zeros((nt + 1, ny, nx), dtype=np.float32)
    infil_rate = np.zeros((nt + 1, ny, nx), dtype=np.float32)
    rainfall_total = np.zeros((ny, nx), dtype=np.float32)

    total = ny * nx
    count = 0

    # ==================================================
    # LOOP GRID
    # ==================================================
    for i in range(ny):
        for j in range(nx):

            count += 1

            if count % 200 == 0 or count == total:
                print(f"{count}/{total}")

            rain = da.isel({lat_dim: i, lon_dim: j}).values

            rain = np.array(rain, dtype=float)

            # ------------------------------------------
            # FIX ERA5 cumulative tp
            # ------------------------------------------
            rain = np.diff(rain, prepend=0)

            # Negative reset values
            rain[rain < 0] = 0

            # meters -> mm
            rain = rain * 1000

            rain[np.isnan(rain)] = 0

            rainfall_total[i, j] = np.sum(rain)

            if np.sum(rain) == 0:
                continue

            try:
                model = GreenAmpt(K, dt, dtheta, psi, rain)
                F, f, t = model.run()

                cumulative[:, i, j] = F
                infil_rate[:, i, j] = f

            except Exception as e:
                print("Error:", i, j, e)

    # ==================================================
    # TIME COORD
    # ==================================================
    delta = time_vals[1] - time_vals[0]
    new_time = np.insert(time_vals.values, 0, time_vals[0] - delta)

    # ==================================================
    # SAVE NETCDF
    # ==================================================
    out = xr.Dataset(
        {
            "cumulative_infiltration":
                (["time", lat_dim, lon_dim], cumulative),

            "infiltration_rate":
                (["time", lat_dim, lon_dim], infil_rate),

            "total_rainfall":
                ([lat_dim, lon_dim], rainfall_total),
        },

        coords={
            "time": new_time,
            lat_dim: lats,
            lon_dim: lons
        },

        attrs={
            "title": f"Green-Ampt Infiltration {year}",
            "created": str(datetime.now()),
            "K_mm_hr": K,
            "psi_mm": psi,
            "dtheta": dtheta
        }
    )

    out["cumulative_infiltration"].attrs["units"] = "mm"
    out["infiltration_rate"].attrs["units"] = "mm/hr"
    out["total_rainfall"].attrs["units"] = "mm"

    out.to_netcdf(output_file)

    print("\nSaved:", output_file)

    print("\nSUMMARY")
    print("Mean rainfall :", rainfall_total.mean())
    print("Mean infil :", cumulative[-1].mean())

    return out


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser(description="Green-Ampt infiltration on an ERA5 grid")
    ap.add_argument("--nc", default=str(DATA_DIR / "ERA5_precipitation.nc"),
                    help="ERA5 NetCDF containing 'tp'")
    ap.add_argument("--year", type=int, required=True, help="year to process")
    ap.add_argument("--out", default=None,
                    help="output NetCDF (default: outputs/infiltration_<year>.nc)")
    # Sandy-loam defaults; see Table 1 of the paper for other soil classes
    ap.add_argument("--K", type=float, default=1.09, help="Ks (mm/h)")
    ap.add_argument("--psi", type=float, default=110, help="suction head (mm)")
    ap.add_argument("--dtheta", type=float, default=0.247, help="moisture deficit")
    ap.add_argument("--dt", type=float, default=1, help="time step (h)")
    a = ap.parse_args()

    process_year(
        nc_file=a.nc,
        output_file=a.out or str(OUTPUT_DIR / f"infiltration_{a.year}.nc"),
        year=a.year,
        K=a.K, psi=a.psi, dtheta=a.dtheta, dt=a.dt,
    )
