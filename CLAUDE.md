# torque2026

Codes for a TORQUE 2026 paper studying an atmospheric bore (a sudden gravity-current-driven wind/pressure/temperature perturbation, distinct from a frontal passage) that swept across the AWAKEN wind energy field campaign area in Oklahoma on **2023-08-05** (~10:00-12:00 UTC), and its impact on wind farm operations at **King Plains** (rotor diameter D=127 m, rated power 2800 kW), part of a wind plant cluster that also includes Armadillo Flats and Breckinridge.

This is a flat collection of standalone analysis/plotting scripts (no shared internal package/imports between them beyond copy-pasted helper functions) run interactively (`#%%` cell markers, spyder/VS-code style). There is no test suite and no CI.

## Data (not tracked in git)

`data/` is almost entirely git-ignored (`*.nc`, `*.cdf`, `*.csv`, `*.xlsx`, `*.ar2v`, `*.mdf`, `*.log`, `*.gif`, `*.png`, `*.html`, `*.hpl` — see `.gitignore`). Scripts assume this data already exists locally; it is not fetched by any script here.

- `data/awaken/` — AWAKEN campaign instruments, one subfolder per instrument/site channel:
  - `{site}.assist.z01.tropoe.c0` — TROPoe thermodynamic retrievals (temperature `theta`, water vapor `waterVapor`, cloud base height `cbh`, QC via `gamma`/`rmsa`), for sites `sc1`, `sb`, `sg` (ASSIST instruments), height coord in km, converted to m via `height_assist=1`.
  - `{site}.met.z01.b0` — surface met stations (pressure, etc.), QC via matching `qc_*` variables.
  - `sh.lidar.z02.c0` — scanning Doppler lidar wind profiles (`u`, `v`, `w`).
  - `sgpdlfptS6.b1` / `.b2` — vertically staring Doppler lidar (`.b1` raw, `.b2` standardized `wind_speed`/`qc_wind_speed`, produced by `standardize_lidars.py`).
  - `kp.turbine.z01.b0` — King Plains turbine SCADA at native (~1s/10-min) resolution, one file per turbine (`WTUR.W_10m_Avg` power, `WMET.HorWdSpd_10m_Avg` wind speed, `WMET.EnvTmp_10m_Avg` temperature).
  - `kp.turbine.z02.00` — King Plains turbine SCADA as raw per-timestamp CSVs with columns like `PKGP1HIST01.OKWF001_KP_Turbine{AA}.{var}` (turbine id = two-char row/col code, e.g. `A9`); read and reshaped into an xarray `Dataset(time, turbine)` in `explore_scada.py`, `scada_map.py`, `scada_time_series.py`.
  - `kp.turbine.z03.b0` / `kp.turbine.z03.del` — per-turbine structural load channels (`tb_bend_resultant` tower-base bending, `b1_bend_root_resultant` blade-root bending) and precomputed long-term DEL time series, used by `DEL.py`/`DEL_daily.py`.
  - `sa1.lidar.z03.c0`, `sa1.met.z01.b0`, `sb.met.z01.b0`, `sc1.met.z01.b0`, `sg.met.z01.b0` — additional site instruments.
- `data/mesonet/` — Oklahoma Mesonet 5-min surface obs (`*.mdf`, whitespace-delimited, 2-row header) + `geoinfo.csv` station lat/lon (converted to UTM with `utm.from_latlon`).
- `data/nexrad/` — raw NEXRAD Level II radar volumes from station **KVNX**, read with `pyart.io.read_nexrad_archive` and gridded with `pyart.map.grid_from_radars` (single-elevation Cartesian slice at `z_rad=100` m AGL is the usual choice).
- `data/20250225_AWAKEN_layout.nc` — AWAKEN site layout: `group='turbines'` (per-turbine `name`, `x_utm`/`y_utm`, `Wind plant`) and `group='ground_sites'` (instrument site locations).
- `data/20230805.100000.20230805.115959.scada.nc` — pre-built King Plains SCADA dataset for the bore window (output of `explore_scada.py`, consumed by `scada_map.py`, `scada_lasso.py`, `scada_U_P.py`).
- `data/pblh_siteG_20230805...csv` — planetary boundary layer height (Heffter method) at site G.
- `data/siteA1_met_outages_15min_v3.nc` — customer power outage counts, compared against the bore timing in `ber.py`.
- `data/wd_offsets_Sept2025.csv` — per-turbine nacelle yaw/wind-direction bias correction (`Northing Bias - 2022`), applied in `scada_map.py`.

## Standardization pipeline

`standardize_lidars.py` runs the `lidargo` package's `Standardize` processor over raw lidar files listed in `configs/config.yaml` (per-channel source subfolder + wildcard + `configs/config_awaken_stand.xlsx` config), date-range filtered, serial or multiprocessing (`Pool`) mode, writing per-file logs to `log/` and a run-level error log. This turns `.b1`-type raw scans into `.b2`-type standardized files (see `sgpdlfptS6.b1` -> `.b2` above). CLI args (positional) override the hardcoded defaults at the top of the script.

## Analysis scripts

- **`bore_profiles.py`** / **`ber.py`** — core time-height cross-section plots of the bore passage: wind speed/direction/vertical velocity from Doppler lidar + staring lidar, potential temperature and water vapor from TROPoe at 3 ASSIST sites, PBLH and cloud base height overlays, surface pressure, and (in `ber.py`) a 3D radar reflectivity snapshot and customer outage time series alongside the profiles. `ber.py` is the newer/superset variant (untracked, WIP). Both duplicate the same `time_interp`/`interp_nan` helpers (nearest-neighbor gap-limited time interpolation + linear NaN-gap-limited interpolation on height/time).
- **`front_analysis.py`** — Oklahoma Mesonet pressure perturbation (`dp`, relative to a pre-bore `avg_time`-min baseline) overlaid on NEXRAD reflectivity maps and King Plains/Armadillo Flats/Breckinridge turbine locations; optional per-frame video export to `figures/mesonet/`.
- **`bore_map.py`** — companion horizontal-map view of the bore passage (radar + site layout).
- **`scada_analysis.py`** / **`scada_map.py`** — per-turbine nacelle orientation + power maps drawn as rotated/recolored turbine icons (`figures/Turbine.png`) over NEXRAD reflectivity, colored by normalized power (`power_rated=2800` kW); `scada_map.py` is the refined version using the prebuilt `data/*.scada.nc` file and applying the yaw bias offset; both can render frame sequences to `figures/yaw/` for a video.
- **`scada_lasso.py`** (+ `test_lasso.py` synthetic sanity check) — detects the dominant spatial wavelength/direction of wind-speed perturbations across the turbine array via a custom LASSO-based plane-wave fit (`sklearn.linear_model.Lasso` on a cosine/sine dictionary over a wavenumber-direction search grid), after removing a bilinear spatial trend; used to characterize bore-induced wave-like structure in the SCADA wind field.
- **`scada_time_series.py`** / **`explore_scada.py`** — read raw per-timestamp SCADA CSVs into an xarray `Dataset(time, turbine)` and plot selected turbines' wind speed/power/yaw; `explore_scada.py` also builds/exports the `*.scada.nc` file consumed elsewhere (currently commented out) and plots the farm layout with selected turbines highlighted.
- **`scada_U_P.py`** — power-curve hysteresis analysis: air-density-corrected wind speed and power for 3 representative turbines during the bore vs. a long-term binned reference power curve (IEC 61400-12-1-style binning with a minimum-count filter), plotted as a time-colored trajectory to show transient deviation from the steady-state curve.
- **`DEL.py`** — computes Damage Equivalent Loads (`openfast_toolbox.tools.fatigue.equivalent_load`, rainflow-counting-based, Mahler/Wöhler exponent `m` per load channel: `m=3` tower-base, `m=10` blade-root) in 10-min bins across the bore window for one turbine, and expresses each bin's DEL as a percentile against the long-term DEL distribution (`kp.turbine.z03.del`).
- **`DEL_daily.py`** — batch version of the same DEL calculation run per-day over a full month, writing daily plots and `.nc` outputs into `kp.turbine.z03.del`.
- **`layout.py`** / **`layout_simple.py`** — static plots of the AWAKEN turbine layout (all farms, or just King Plains) with instrument site markers; `layout.py` supports highlighting arbitrary selected turbines, `layout_simple.py` is a minimal single-farm version.
- **`test_nexrad.py`** — minimal smoke test for reading/gridding a single NEXRAD volume with `pyart`.
- **`extract_gif.py`** — utility to explode an animated GIF (radar loop) into individual PNG frames.

## Conventions used throughout

- Every plotting script sets: `matplotlib.rcParams['font.family']='serif'`, `mathtext.fontset='cm'`, `savefig.dpi=500`, and starts with `plt.close("all")`.
- `cd=os.path.dirname(__file__)` at the top of each script; all data/figure paths are built from `cd`, so scripts are runnable regardless of CWD (except `DEL.py`/`DEL_daily.py`, which use `os.getcwd()`).
- QC pattern: xarray data variables `v` are masked via a companion `qc_v` variable (`0` = good), e.g. `Data[v].where(Data[f'qc_{v}']==0)`.
- Distances/coordinates are converted to UTM (`utm.from_latlon`) and plotted in **km**, not m.
- Wind direction convention: `wd = (270 - degrees(atan2(v, u))) % 360` (meteorological convention from u/v components).
- Time-height interpolation is gap-limited in both height (`limit_height`, meters) and time (`limit_time`, seconds) to avoid over-interpolating across data voids — see the shared `time_interp`/`interp_nan` pattern in `bore_profiles.py`/`ber.py`.
- Figures are written under `figures/<topic>/`, created with `os.makedirs(..., exist_ok=True)`.
