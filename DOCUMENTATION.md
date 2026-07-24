# SaGEA Documentation

> **SaGEA** — SAtellite Gravity Error Assessment  
> Version: 0.3.0  
> Reference: Liu, Yang & Forootan (2025), *Computers & Geosciences*, 196, 105825. https://doi.org/10.1016/j.cageo.2024.105825

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Core Data Classes](#3-core-data-classes)
   - 3.1 [SHC — Spherical Harmonic Coefficients](#31-shc--spherical-harmonic-coefficients)
   - 3.2 [GRD — Gridded Data](#32-grd--gridded-data)
4. [Post-Processing L2 Products](#4-post-processing-l2-products)
   - 4.1 [Loading Data](#41-loading-data)
   - 4.2 [Low-Degree Coefficient Replacement](#42-low-degree-coefficient-replacement)
   - 4.3 [GIA Removal](#43-gia-removal)
   - 4.4 [Filtering](#44-filtering)
   - 4.5 [Geometric Correction](#45-geometric-correction)
   - 4.6 [Unit Conversion](#46-unit-conversion)
   - 4.7 [Harmonic Synthesis and Regional Extraction](#47-harmonic-synthesis-and-regional-extraction)
   - 4.8 [Complete Post-Processing Example](#48-complete-post-processing-example)
5. [Error Analysis](#5-error-analysis)
   - 5.1 [Error-I: Formal Error / VCM Propagation](#51-error-i-formal-error--vcm-propagation)
   - 5.2 [Error-II: Between-Product Discrepancy (TCH and TCA)](#52-error-ii-between-product-discrepancy-tch-and-tca)
   - 5.3 [Error-III: Post-Processing Chain Uncertainty](#53-error-iii-post-processing-chain-uncertainty)
   - 5.4 [Complete Error Analysis Example](#54-complete-error-analysis-example)
6. [API Reference](#6-api-reference)
7. [References](#7-references)

---

## 1. Introduction

GRACE (Gravity Recovery and Climate Experiment) and its Follow-On mission GRACE-FO provide time-variable gravity
fields in the form of **Level-2 spherical harmonic coefficient (SHC) products**. These products are widely used in
geoscience studies — from quantifying terrestrial water storage changes and ice-sheet mass balance, to monitoring
global mean ocean mass and crustal deformation.

However, turning raw Level-2 products into scientifically useful Level-3 estimates requires a **non-trivial
post-processing chain** involving filtering, low-degree replacement, physical unit conversion, and various corrections.
Furthermore, the resulting estimates carry **three distinct classes of uncertainty** that are often overlooked or
treated inconsistently in the literature:

| Error type | Source | SaGEA term |
|---|---|---|
| Formal / propagated instrument noise | VCM of Level-2 SHCs, instrument noise, background-model errors | **Error-I** |
| Between-solution discrepancy | Different Level-2 producers (CSR, GFZ, JPL, ITSG, …) | **Error-II** |
| Post-processing chain uncertainty | Non-unique choices of filter, correction, leakage method | **Error-III** |

**SaGEA** integrates a comprehensive set of post-processing tools and all three error assessment strategies into a
single, Python-native, open-source toolbox. Its design philosophy is:

- **Data-class-centric.** Every operation is exposed as a method on the `SHC` or `GRD` data class rather than as a
  standalone function, making pipelines readable and composable.
- **Non-destructive by default.** Every processing method accepts `inplace=False` (the default), so the original data
  are preserved and intermediate states can be inspected.
- **Flexible output.** Methods accept and return the same data class, so any combination of steps can be chained.

SaGEA is developed and maintained by the **NCSG (Numerical Computation and Satellite Geodesy)** research group.
Please contact Shuhao Liu (liushuhao@apm.ac.cn) or Fan Yang (fany@plan.aau.dk) for more information.

---

## 2. Installation

```bash
pip install sagea
```

Requires Python 3.9 or later. The following packages are required and will be installed automatically:
`numpy`, `scipy`, `h5py`, `netCDF4`, `matplotlib`, `cartopy`, `rasterio`.

---

## 3. Core Data Classes

SaGEA is organized around two data classes:

```
SHC  ─── spherical-domain data (degree/order coefficients)
GRD  ─── spatial-domain data  (latitude/longitude grid)
```

Every post-processing step either takes a `SHC` and returns a `SHC`, or takes a `GRD` and returns a `GRD`.
Conversion between the two domains is provided by `shc.synthesize.to_grid()` and `grd.to_SHC()`.

### 3.1 SHC — Spherical Harmonic Coefficients

`sagea.SHC` stores one or more epochs of **fully normalized** (4π convention) spherical harmonic coefficients.

**Internal layout**

Coefficients are stored in a 2-D array of shape `(ntime, (lmax+1)²)`. The second dimension is ordered as:

```
[C(0,0), S(1,1), C(1,0), C(1,1), S(2,2), S(2,1), C(2,0), C(2,1), C(2,2), S(3,3), ...]
```

**Key attributes**

| Attribute | Type | Description |
|---|---|---|
| `.value` | `ndarray (ntime, ncoef)` | Raw coefficient array |
| `.lmax` | `int` | Maximum degree |
| `.ntime` | `int` | Number of epochs |
| `.mean` | `ndarray (ncoef,)` | Time-mean coefficients |
| `.std` | `ndarray (ncoef,)` | Coefficient std over time |
| `.degree_rms` | `ndarray (lmax+1,)` | Degree-RMS spectrum |

**Accessor namespaces**

`SHC` exposes its methods through four accessor namespaces. Call `.help()` on any of them to print available
methods and their signatures:

```python
SHC.io.help()          # I/O: reading and writing .gfc files
SHC.generate.help()    # generators: from array, from trend, from VCM
shc.filter.help()      # filtering methods
shc.synthesize.help()  # harmonic synthesis to GRD or discrete points
shc.correction.help()  # geometric and other corrections
```

### 3.2 GRD — Gridded Data

`sagea.GRD` stores one or more epochs of data on a regular geographic grid.

**Constructor**

```python
import numpy as np
import sagea

grid_array = np.zeros((ntime, nlat, nlon))  # or (nlat, nlon) for single epoch
lat = np.arange(-89.5, 90.5, 1.0)
lon = np.arange(-179.5, 180.5, 1.0)

grd = sagea.GRD(grid_array, lat=lat, lon=lon)
```

**Key attributes**

| Attribute | Description |
|---|---|
| `.value` | `ndarray (ntime, nlat, nlon)` — always 3-D internally |
| `.lat`, `.lon` | Coordinate arrays (degrees) |
| `.mean` | Time-mean grid |
| `.std` | Standard deviation over time |

**Extract accessor**

`grd.extract.maskGRD(mask_grd, average=True)` — compute basin-average time series using a binary mask `GRD`.
Returns `ndarray` of shape `(n_masks, ntime)`.

---

## 4. Post-Processing L2 Products

This section walks through a complete post-processing pipeline from raw `.gfc` files to gridded equivalent water
height (EWH) anomalies, and basin-scale time series.

### 4.1 Loading Data

GRACE Level-2 products are distributed in **ICGEM `.gfc` format** (or GRACE-specific variants such as the
`GRCOF2`-keyed files from CSR/GFZ/JPL). SaGEA reads them via `SHC.io.from_gfc`.

```python
import pathlib
import sagea
from sagea.utils import TimeTool

# --- paths ---
paths_gfc = sorted(pathlib.Path("/data/ITSG/Grace2018/n60/2010").glob("ITSG*.gfc"))
path_gif48 = pathlib.Path("/data/auxiliary/GIF48.gfc")

lmax = 60

# --- read monthly solutions ---
shc = sagea.SHC.io.from_gfc(paths_gfc, lmax=lmax, key="gfc")

# --- read and subtract static background field ---
shc_gif48 = sagea.SHC.io.from_gfc(path_gif48, lmax=lmax, key="gfc")
shc -= shc_gif48
```

`SHC.io.from_gfc` accepts either a single path or a list of paths.  
The `key` parameter identifies the leading token on data rows (e.g. `"gfc"`, `"GRCOF2"`, or `""` for keyless files).

Extract epoch dates from filenames using the utility functions:

```python
dates_begin, dates_end = TimeTool.match_dates_from_name(paths_gfc)
dates = TimeTool.get_average_dates(dates_begin, dates_end)
```

### 4.2 Low-Degree Coefficient Replacement

GRACE Level-2 products lack **degree-1 coefficients** (geocenter motion) and have poorly determined **C20** and
**C30** terms. These are routinely replaced with estimates from satellite laser ranging (SLR) before further use.

SaGEA provides `sagea.sgio.read_low_degs` to parse the standard GRACE Technical Note files:

| Technical Note | Coefficients provided |
|---|---|
| TN-13 | Degree-1: C(1,0), C(1,1), S(1,1) |
| TN-14 | C20, C30 |
| TN-11 | C20 only |

```python
from sagea.sgio import read_low_degs

path_TN13 = pathlib.Path("/data/TN-13_GEOC_JPL_RL06.3.txt")
path_TN14 = pathlib.Path("/data/TN-14_C30_C20_SLR_GSFC.txt")

low_deg = {}
low_deg.update(read_low_degs(path_TN13, dates=dates))
low_deg.update(read_low_degs(path_TN14, dates=dates))
# low_deg is now: {"c1,0": ndarray, "c1,1": ndarray, "s1,1": ndarray,
#                  "c2,0": ndarray, "c3,0": ndarray, ...}

# De-mean each replacement series, then apply
import numpy as np
for key in ("c1,0", "c1,1", "s1,1", "c2,0", "c3,0"):
    vals = low_deg[key]
    low_deg[key] -= np.nanmean(vals)
    shc.replace(key, low_deg[key], inplace=True)
```

`shc.replace(key, array, inplace)` skips `NaN` positions, so months with missing SLR data are left unchanged.

### 4.3 GIA Removal

Glacial Isostatic Adjustment (GIA) appears as a long-term linear trend in GRACE gravity fields. It is removed
by subtracting a model-based trend, which is available as a `.gfc` file containing rate coefficients (unit: per year).

```python
path_gia = pathlib.Path("/data/GIA/GIA.ICE-6G_D.txt")

shc_gia_rate = sagea.SHC.io.from_gfc(path_gia, lmax=lmax, key="")
shc_gia = sagea.SHC.generate.from_trend(shc_gia_rate, dates=dates)

shc -= shc_gia
```

`SHC.generate.from_trend(shc_trend, dates, ref_time=None)` computes `C(t) = C_rate × (t − t_ref)` for each
epoch in `dates`.

### 4.4 Filtering

GRACE Level-2 solutions contain correlated noise (north–south stripes) at high degrees. SaGEA provides the
following spectral filtering methods through the `shc.filter` accessor:

| Method | Parameters | Reference |
|---|---|---|
| `gaussian(radius)` | radius in km | Wahr et al. (1998) |
| `fan(radius1, radius2)` | radii in km | Zhang et al. (2009) |
| `han(radius1, radius2, m0)` | radii in km | Han et al. (2005) |
| `ddk(ddk_id)` | id = 1–8 | Kusche (2007) |
| `pnmm(n, m)` | polynomial degree, order | Chambers (2006) |
| `slidewindowSwenson2006(n, m, a, k, window_length_min)` | — | Swenson & Wahr (2006) |
| `slidewindowDuan2009(n, m, a, k, window_length_min, gamma, p)` | — | Duan et al. (2009) |
| `fsc(vcm_err, vcm_sig_list, ...)` | VCM-based | Bayesian-VCE |

All methods accept `inplace=False` (default) and return an `SHC` instance.

```python
# Option A: DDK filter (recommended for most studies)
shc_filtered = shc.filter.ddk(ddk_id=4)

# Option B: Decorrelation + Gaussian (classic combination)
shc_filtered = shc.filter.pnmm(3, 5)
shc_filtered = shc_filtered.filter.gaussian(300)

# Option C: inplace modification
shc.filter.gaussian(300, inplace=True)
```

Run `print(shc.filter.help())` to list all available filters with their full signatures.

### 4.5 Geometric Correction

Converting geopotential SHCs to terrestrial water storage assumes a spherical Earth. The deviation of the actual
Earth from that sphere (ellipsoidal shape + topography) introduces a systematic bias, especially at high latitudes
and over mountain ranges. SaGEA implements the correction of Yang et al. (2022):

```python
shc_corrected = shc_filtered.correction.geometric(
    auto_load_actual_earth=True,
    phisfc_file="/path/to/PHISFC_ERA5_invariant.nc",   # ERA5 surface geopotential
    gif48_file="/path/to/GIF48.gfc",                   # static gravity model
    inplace=False,
    verbose=True,
)
```

When `auto_load_actual_earth=True`, the bundled ERA5-based surface geopotential file is used automatically
(the explicit `phisfc_file` and `gif48_file` paths can be omitted if you use the default data bundled with SaGEA).

### 4.6 Unit Conversion

Level-2 products are in **dimensionless geopotential** (Stokes coefficients). Most geoscience applications require
conversion to a physical quantity. Use `shc.convert`:

```python
# Geopotential → Equivalent Water Height [m]
shc_ewh = shc_filtered.convert(from_type="Geopotential", to_type="EWH")

# Other available target types (see PhysicalDimension enum):
# "MassDensity", "Gravity", "Geoid", "EWH", ...
```

### 4.7 Harmonic Synthesis and Regional Extraction

**Synthesis to a global grid**

```python
grid_space = 1  # degrees (produces a 180×360 grid)
grid = shc_ewh.synthesize.to_grid(grid_space)

# Convert from meters to centimeters
grid.value *= 100
```

**Evaluation at discrete points**

```python
import numpy as np

lats = np.array([58.5, 57.5, 56.5])   # degrees north
lons = np.array([26.5, 27.5, 28.5])   # degrees east

ewh_pts = shc_ewh.synthesize.evaluate(lats, lons)  # shape (ntime, npoints)
```

**Regional (basin-scale) extraction**

Load a basin mask from a shapefile and compute area-weighted averages:

```python
from sagea.sgio import read_shp_as_GRD
from sagea.utils import TimeTool

mask_grd = read_shp_as_GRD(
    "/data/basin_mask/Greenland/greenland.shp",
    grid_space=grid_space,
    per_feature=True,    # each polygon → one epoch/mask
)

# Extract time series: shape (n_features, ntime)
ewh_basin = grid.extract.maskGRD(mask_grd, average=True)

# Convert dates to decimal year for plotting
year_frac = TimeTool.convert_date_format(dates, output_type=TimeTool.DateFormat.YearFraction)
```

### 4.8 Complete Post-Processing Example

The following is a self-contained pipeline from raw ITSG files to Greenland EWH anomaly time series:

```python
import pathlib
import numpy as np
import sagea
from sagea.sgio import read_low_degs, read_shp_as_GRD
from sagea.utils import TimeTool

# ── 1. Paths ──────────────────────────────────────────────────────────────────
paths_gfc = sorted(pathlib.Path("/data/ITSG/Grace2018/n60").rglob("ITSG*.gfc"))
path_gif48 = pathlib.Path("/data/auxiliary/GIF48.gfc")
path_TN13  = pathlib.Path("/data/TN-13_GEOC_JPL_RL06.3.txt")
path_TN14  = pathlib.Path("/data/TN-14_C30_C20_SLR_GSFC.txt")
path_gia   = pathlib.Path("/data/GIA/GIA.ICE-6G_D.txt")
path_mask  = pathlib.Path("/data/basin_mask/Greenland/greenland.shp")

lmax = 60

# ── 2. Load monthly solutions and remove static field ─────────────────────────
shc = sagea.SHC.io.from_gfc(paths_gfc, lmax=lmax, key="gfc")
shc -= sagea.SHC.io.from_gfc(path_gif48, lmax=lmax, key="gfc")

dates_begin, dates_end = TimeTool.match_dates_from_name(paths_gfc)
dates = TimeTool.get_average_dates(dates_begin, dates_end)

# ── 3. Replace low-degree coefficients ───────────────────────────────────────
low_deg = {}
low_deg.update(read_low_degs(path_TN13, dates=dates))
low_deg.update(read_low_degs(path_TN14, dates=dates))

for key in ("c1,0", "c1,1", "s1,1", "c2,0", "c3,0"):
    low_deg[key] -= np.nanmean(low_deg[key])
    shc.replace(key, low_deg[key], inplace=True)

# ── 4. Remove GIA ────────────────────────────────────────────────────────────
shc_gia_rate = sagea.SHC.io.from_gfc(path_gia, lmax=lmax, key="")
shc -= sagea.SHC.generate.from_trend(shc_gia_rate, dates=dates)

# ── 5. Filter ────────────────────────────────────────────────────────────────
shc_filt = shc.filter.ddk(ddk_id=4)

# ── 6. Convert to EWH and synthesize ─────────────────────────────────────────
shc_ewh = shc_filt.convert(from_type="Geopotential", to_type="EWH")
grid = shc_ewh.synthesize.to_grid(grid_space=1)
grid.value *= 100  # m → cm

# ── 7. Regional extraction ───────────────────────────────────────────────────
mask = read_shp_as_GRD(path_mask, grid_space=1, per_feature=True)
ewh_greenland = grid.extract.maskGRD(mask, average=True)  # (n_features, ntime)
```

---

## 5. Error Analysis

SaGEA quantifies three distinct categories of error in GRACE-based mass change estimates.

### 5.1 Error-I: Formal Error / VCM Propagation

**Error-I** represents the propagated formal error of the Level-2 product itself, including instrument noise and
background-model uncertainty. The ITSG-Grace2018 solution provides the full **variance-covariance matrix (VCM)**
of its SHCs in SINEX format, which can be propagated through an arbitrary post-processing chain using Monte Carlo
sampling.

**Step 1 — Read the SINEX VCM**

```python
import sagea

path_snx = pathlib.Path("/data/ITSG_SINEX/ITSG-Grace2018_n96_2008-03.snx")
lmax = 60

vcm, _ = sagea.io.read_sinex_cov(path_snx, lmax=lmax)
# vcm: ndarray of shape ((lmax+1)², (lmax+1)²)
# Note: parsing a SINEX file may take tens of seconds.
```

`read_sinex_cov` reads the `+SOLUTION/NORMAL_EQUATION_MATRIX` block, inverts the normal equations to form the
VCM, and re-orders it to SaGEA's canonical SHC ordering.

**Step 2 — Generate Monte Carlo samples**

```python
from sagea import SHC

path_gfc = pathlib.Path("/data/ITSG/Grace2018/n60/2008/ITSG-Grace2018_2008-03.gfc")
shc_truth = SHC.io.from_gfc(path_gfc, lmax=lmax, key="gfc")

nsample = 100
shc_samples = SHC.generate.normal_from_vcm(vcm, nsample=nsample, mean=shc_truth)
# shc_samples: SHC with ntime = nsample
```

`normal_from_vcm(vcm, nsample, mean)` draws `nsample` realizations from the multivariate Gaussian
`N(mean, vcm)`. When `mean=None`, the distribution is zero-mean (noise-only samples).

**Step 3 — Propagate through the post-processing chain**

Apply the exact same processing steps to all samples at once. Because `SHC` operations vectorize over epochs,
the code is identical to single-epoch processing:

```python
shc_filtered = shc_samples.filter.gaussian(300)
shc_ewh      = shc_filtered.convert(from_type="Geopotential", to_type="EWH")
grid         = shc_ewh.synthesize.to_grid(grid_space=1)
```

**Step 4 — Compute statistics over the ensemble**

```python
import numpy as np
import sagea

# Pointwise standard deviation map
std_values = np.std(grid.value, axis=0)          # shape (nlat, nlon)
grid_std   = sagea.GRD(std_values * 100, lat=grid.lat, lon=grid.lon)  # cm

# Evaluate at discrete points and compute full covariance
ewh_pts = shc_ewh.synthesize.evaluate(lats, lons)  # (nsample, npoints)
cov_pts = np.cov(ewh_pts.T)                        # (npoints, npoints) covariance
```

The resulting `grid_std` is a spatial map of the formal (Error-I) uncertainty in centimetres EWH for that month.
Repeating for all months gives the full temporal picture.

### 5.2 Error-II: Between-Product Discrepancy (TCH and TCA)

**Error-II** measures how much the choice of Level-2 product producer (CSR, GFZ, JPL, ITSG, …) contributes to
uncertainty in the derived Level-3 estimate. SaGEA implements two statistical methods for this:

- **TCH** (Three-Cornered Hat) — estimates the random error of each dataset by solving for individual variances
  from pairwise difference variances.
- **TCA** (Triple Collocation Analysis) — assumes an explicit signal model `x_i = β_i · t + ε_i` and exploits
  the cross-covariance between dataset pairs.

Both accept `GRD`, `SHC`, or plain `ndarray` as input and return results in the same type.

**TCH**

```python
from sagea.error_assessment import tch, TCHMode

# grid_csr, grid_gfz, grid_jpl: GRD instances from identical post-processing
grid_err_csr, grid_err_gfz, grid_err_jpl = tch(
    grid_csr, grid_gfz, grid_jpl,
    mode=TCHMode.OLS,                   # "OLS" or "KKT"
    negative_variance_policy="NAN",     # "NAN" | "CLIP" | "ABS"
    min_valid_obs=3,
)
# Each grid_err_* contains the estimated pointwise error std for that dataset.
```

Available TCH modes:

| Mode | Description |
|---|---|
| `TCHMode.OLS` | Pairwise difference variances solved via ordinary least squares (recommended for ≥3 datasets) |
| `TCHMode.KKT` | KKT constrained optimization using the last dataset as reference (classic 3-product formulation) |

**TCA**

```python
from sagea.error_assessment import tca, TCAMode

grid_err_csr, grid_err_gfz, grid_err_jpl = tca(
    grid_csr, grid_gfz, grid_jpl,
    mode=TCAMode.CLASSIC,               # "CLASSIC" | "ALL_TRIPLETS" | "NLS"
)
```

Available TCA modes:

| Mode | Description |
|---|---|
| `TCAMode.CLASSIC` | Classic 3-product TCA: `σ²_i = C_ii − C_ij·C_ik / C_jk` (requires exactly 3 datasets) |
| `TCAMode.ALL_TRIPLETS` | For N ≥ 3 datasets: evaluates all C(N,3) triplets, aggregates by `nanmedian` |
| `TCAMode.NLS` | Multiple collocation via non-linear least squares fitting of off-diagonal covariances |

**Working with SHC inputs**

Both `tch` and `tca` work on SHC inputs in the same way — they process each coefficient independently
across the time axis:

```python
shc_err_csr, shc_err_gfz, shc_err_jpl = tch(shc_csr, shc_gfz, shc_jpl)
```

**Object-oriented API (advanced)**

For fine-grained control or reuse of configured estimators:

```python
from sagea.error_assessment import tch as TCH_module
from sagea.error_assessment._tch._TCH import TCH

tch_obj = TCH()
tch_obj.configuration.set_mode("OLS")
tch_obj.configuration.set_negative_variance_policy("NAN")
tch_obj.configuration.set_min_valid_obs(3)

tch_obj.set_datasets(x1, x2, x3)   # 1-D numpy arrays
variances = tch_obj.get_variance()  # ndarray (3,)
stds      = tch_obj.get_std()       # ndarray (3,)
```

### 5.3 Error-III: Post-Processing Chain Uncertainty

**Error-III** arises from the fact that there is no single agreed-upon post-processing chain. Different choices of
filter, leakage correction, correction methods, etc., produce different Level-3 estimates even from the same
Level-2 product. This **within-group, between-chain** uncertainty is quantified by running an **ensemble** of
processing variants and computing their spread.

The strategy is:

1. Define a set of processing variants (e.g., different filter radii, different leakage methods).
2. Apply each variant to the same Level-2 product using SaGEA's post-processing API.
3. Collect the resulting `GRD` or time-series outputs and compute their standard deviation.

Because `SHC` operations are vectorized and non-destructive, each ensemble member can be generated independently
without duplicating data:

```python
filters = [
    lambda shc: shc.filter.gaussian(300),
    lambda shc: shc.filter.gaussian(400),
    lambda shc: shc.filter.ddk(ddk_id=3),
    lambda shc: shc.filter.ddk(ddk_id=5),
]

grids = []
for filt in filters:
    shc_filt = filt(shc)
    shc_ewh  = shc_filt.convert(from_type="Geopotential", to_type="EWH")
    grids.append(shc_ewh.synthesize.to_grid(grid_space=1))

# Compute ensemble spread
import numpy as np
ensemble = np.stack([g.value for g in grids], axis=0)  # (n_variants, ntime, nlat, nlon)
std_chain = np.std(ensemble, axis=0)                    # (ntime, nlat, nlon)
```

---

## 6. API Reference

### `sagea.SHC`

| Member | Kind | Signature | Returns | Description |
|---|---|---|---|---|
| `SHC.io.from_gfc` | class method | `(filepath, lmax, key="gfc", ...)` | `SHC` | Read from `.gfc` file(s) |
| `SHC.generate.from_array` | class method | `(cs, normalization, dates, attrs)` | `SHC` | Construct from array |
| `SHC.generate.from_trend` | class method | `(shc_trend, dates, ref_time=None)` | `SHC` | Build linear trend time series |
| `SHC.generate.normal_from_vcm` | class method | `(vcm, nsample=1, mean=None)` | `SHC` | Monte Carlo samples from VCM |
| `shc.replace` | instance | `(*params, inplace=False)` | `SHC` | Replace individual coefficients |
| `shc.convert` | instance | `(from_type, to_type, inplace=False)` | `SHC` | Physical unit conversion |
| `shc.de_mean` | instance | `(inplace=False)` | `SHC` | Subtract temporal mean |
| `shc.filter.gaussian` | instance | `(radius, inplace=False)` | `SHC` | Gaussian smoothing |
| `shc.filter.fan` | instance | `(radius1, radius2, inplace=False)` | `SHC` | Fan filter |
| `shc.filter.han` | instance | `(radius1, radius2, m0, inplace=False)` | `SHC` | Non-isotropic Gaussian |
| `shc.filter.ddk` | instance | `(ddk_id, inplace=False)` | `SHC` | DDK filter (id 1–8) |
| `shc.filter.pnmm` | instance | `(n, m, inplace=False)` | `SHC` | PnMm de-correlation |
| `shc.filter.slidewindowSwenson2006` | instance | `(n, m, a, k, window_length_min, inplace=False)` | `SHC` | Moving-window de-correlation |
| `shc.filter.slidewindowDuan2009` | instance | `(n, m, a, k, window_length_min, gamma, p, inplace=False)` | `SHC` | Moving-window de-correlation |
| `shc.filter.fsc` | instance | `(vcm_err, vcm_sig_list, ...)` | `SHC` | Bayesian-VCE filter |
| `shc.synthesize.to_grid` | instance | `(grid_space, grid_type=None)` | `GRD` | Harmonic synthesis to global grid |
| `shc.synthesize.evaluate` | instance | `(lat, lon)` | `ndarray` | Evaluate at discrete lat/lon |
| `shc.correction.geometric` | instance | `(auto_load_actual_earth, ..., inplace=False)` | `SHC` | Geometric/ellipsoidal correction |
| `shc.io.save_file` | instance | `(filepath, index, header, key, ...)` | — | Write one epoch to `.gfc` file |

### `sagea.GRD`

| Member | Kind | Signature | Returns | Description |
|---|---|---|---|---|
| `GRD(grid, lat, lon, option=1)` | constructor | — | `GRD` | Create from array |
| `grd.to_SHC` | instance | `(lmax)` | `SHC` | Harmonic analysis |
| `grd.extract.maskGRD` | instance | `(mask, average=True)` | `ndarray` | Basin-integral / area-average |
| `grd.plot` | instance | `(vmin, vmax, projection, ...)` | `(fig, axes)` | Global map visualization |
| `grd.copy` | instance | `()` | `GRD` | Deep copy |

### `sagea.sgio` (also `sagea.io`)

| Function | Signature | Returns | Description |
|---|---|---|---|
| `read_gfc` | `(filepath, key, lmax, ...)` | `SHC` or `ndarray` | Read ICGEM `.gfc` file |
| `read_sinex_cov` | `(filepath, lmax)` | `(ndarray, list)` | Read SINEX VCM |
| `read_low_degs` | `(filepath, dates)` | `dict` | Read TN-11/13/14 replacement coefficients |
| `read_shp_as_GRD` | `(filepath, grid_space, per_feature)` | `GRD` | Rasterize shapefile to grid |
| `read_shp_as_SHC` | `(filepath, lmax, per_feature)` | `SHC` | Rasterize shapefile to SHC |

### `sagea.error_assessment`

| Function/Class | Signature | Returns | Description |
|---|---|---|---|
| `tch(*datasets, mode, negative_variance_policy, min_valid_obs)` | `GRD / SHC / ndarray` | `tuple` | Three-Cornered Hat error estimation |
| `tca(*datasets, mode, negative_variance_policy, min_valid_obs)` | `GRD / SHC / ndarray` | `tuple` | Triple Collocation Analysis |
| `TCHMode.OLS` | enum value | — | OLS-based TCH |
| `TCHMode.KKT` | enum value | — | KKT-based TCH |
| `TCAMode.CLASSIC` | enum value | — | Classic 3-product TCA |
| `TCAMode.ALL_TRIPLETS` | enum value | — | All-triplets aggregation |
| `TCAMode.NLS` | enum value | — | Non-linear least squares TCA |

---

## 7. References

A, G., Wahr, J., Zhong, S. (2013). Computations of the viscoelastic response of a 3-D compressible Earth to surface loading. *Geophys. J. Int.*, 192, 557–572. https://doi.org/10.1093/gji/ggs030

Chen, J. L., et al. (2007). GRACE detects coseismic and postseismic deformation from the Sumatra-Andaman earthquake. *Geophys. Res. Lett.*, 34. https://doi.org/10.1029/2007gl030356

Chen, J. L., et al. (2021). Error Assessment of GRACE and GRACE Follow-On Mass Change. *J. Geophys. Res. Solid Earth*, 126. https://doi.org/10.1029/2021jb022124

Cheng, M., & Ries, J. (2017). The unexpected signal in GRACE estimates of C20. *J. Geodesy*, 91(8), 897–914. https://doi.org/10.1007/s00190-016-0995-5

Ditmar, P. (2018). Conversion of time-varying Stokes coefficients into mass anomalies at the Earth's surface. *J. Geodesy*, 92(12), 1401–1412. https://doi.org/10.1007/s00190-018-1128-0

Duan, X. J., et al. (2009). On the postprocessing removal of correlated errors in GRACE temporal gravity field solutions. *J. Geodesy*, 83(11), 1095–1106. https://doi.org/10.1007/s00190-009-0327-0

Ferreira, V., et al. (2016). Uncertainties of GRACE time-variable gravity-field solutions based on three-cornered hat method. *J. Appl. Remote Sensing*, 10, 015015. https://doi.org/10.1117/1.JRS.10.015015

Han, S.-C., et al. (2005). Non-isotropic filtering of GRACE temporal gravity for geophysical signal enhancement. *Geophys. J. Int.*, 163(1), 18–25. https://doi.org/10.1111/j.1365-246x.2005.02756.x

Kusche, J. (2007). Approximate decorrelation and non-isotropic smoothing of time-variable GRACE-type gravity field models. *J. Geodesy*, 81(11), 733–749. https://doi.org/10.1007/s00190-007-0143-3

Kvas, A., et al. (2019). ITSG-Grace2018: Overview and evaluation. *J. Geophys. Res. Solid Earth*, 124, 9332–9344. https://doi.org/10.1029/2019JB017415

Landerer, F. W., & Swenson, S. C. (2012). Accuracy of scaled GRACE terrestrial water storage estimates. *Water Resour. Res.*, 48(4). https://doi.org/10.1029/2011wr011453

Liu, S., Yang, F., & Forootan, E. (2025). SAGEA: A toolbox for comprehensive error assessment of GRACE and GRACE-FO based mass changes. *Computers & Geosciences*, 196, 105825. https://doi.org/10.1016/j.cageo.2024.105825

Loomis, B. D., et al. (2020). Replacing GRACE/GRACE-FO With Satellite Laser Ranging. *Geophys. Res. Lett.*, 47(3). https://doi.org/10.1029/2019gl085488

Sun, Y., Riva, R., Ditmar, P. (2016). Optimizing estimates of annual variations and trends in geocenter motion and J2. *J. Geophys. Res. Solid Earth*, 121, 8352–8370. https://doi.org/10.1002/2016jb013073

Swenson, S., & Wahr, J. (2006). Post-processing removal of correlated errors in GRACE data. *Geophys. Res. Lett.* https://doi.org/10.1029/2005gl025285

Wahr, J., Molenaar, M., Bryan, F. (1998). Time variability of the Earth's gravity field. *J. Geophys. Res.*, 103, 30205–30229. https://doi.org/10.1029/98jb02844

Yang, F., et al. (2022). On study of the earth topography correction for the GRACE surface mass estimation. *J. Geodesy*, 96. https://doi.org/10.1007/s00190-022-01683-0

Yang, F., et al. (2024). A Monte Carlo Propagation of the Full Variance-Covariance of GRACE-Like Level-2 Data. *Water Resour. Res.*, 60(9), e2023WR036764. https://doi.org/10.1029/2023WR036764

Zhang, Z.-Z., et al. (2009). An effective filtering for GRACE time-variable gravity: Fan filter. *Geophys. Res. Lett.*, 36(17). https://doi.org/10.1029/2009gl039459
