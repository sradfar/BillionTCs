# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script reads precipitation data from ERA5 for tropical cyclones and computes 
# the mean total precipitation (TP) in a 2°×2° box around the landfall point at 
# multiple lead times. It then applies a bootstrap method to compare TP evolution 
# between different cyclone categories (RI vs. non-RI, MHW vs. non-MHW).
#
# Outputs:
# - A time series plot showing the bootstrap-derived confidence intervals for 
#   precipitation differences between:
#     • RI vs. non-RI
#     • MHW vs. non-MHW
# - Prints lead hours with statistically significant positive differences.
#
# For a detailed description of the methodologies and further insights, please refer to:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2024). 
# *Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide*.
# Science Advances.
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1.  User parameters
# ---------------------------------------------------------------------
RI_CSV      = r'.../with_RI_land_info.csv'
NON_RI_CSV  = r'.../non_RI_land_info.csv'
IBTRACS_CSV = r'.../ibtracs_filled.csv'
ERA5_DIR    = r'D:/'          #  files like  era5_precipitation2020.nc

LEAD_HOURS  = np.arange(-120, 49, 3)            # −5 d … +2 d every 3 h
BOX_DEG     = 2                                 # 2°×2° averaging box
BOOTSTRAPS  = 1000
ALPHA       = 0.05

# ---------------------------------------------------------------------
# 2.  Helper functions (from TP_pattern_new.py) :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------
def _adj_lon(lon):        # 0–360°
    return lon % 360

def _nearest_grid(lat, lon, ds):
    lon = _adj_lon(lon)
    lat_sel = ds['latitude'].isel(latitude=np.abs(ds['latitude']-lat).argmin()).item()
    lon_sel = ds['longitude'].isel(longitude=np.abs(ds['longitude']-lon).argmin()).item()
    return lat_sel, lon_sel

def tp_box_mean(ds, when, lat, lon):
    """Mean TP (mm) in a BOX_DEG° square for the 3‑h window starting *when*."""
    lat0, lon0 = _nearest_grid(lat, lon, ds)
    lat_slice  = slice(lat0+BOX_DEG, lat0-BOX_DEG)   # ERA5 lats descend
    lon_slice  = slice(lon0-BOX_DEG, lon0+BOX_DEG)
    tp_sum = 0.0
    for h in range(3):
        tp_sum += (
            ds.sel(time=when+timedelta(hours=h), method='nearest')
              .sel(latitude=lat_slice, longitude=lon_slice)
              .tp.mean().item()
        )
    return tp_sum * 1000        # m → mm

# ---------------------------------------------------------------------
# 3.  Build storm‑level metadata (RI / non‑RI, MHW / non‑MHW) :contentReference[oaicite:1]{index=1}
# ---------------------------------------------------------------------
ri  = pd.read_csv(RI_CSV)
nri = pd.read_csv(NON_RI_CSV)

ri  = ri[  ri['land']==1 ].copy()
nri = nri[nri['land']==1].copy()

ri ['RI'] = 'RI'
nri['RI'] = 'non_RI'
ri ['MHW'] = np.where((ri ['MHW_land']==1)&(ri ['mhw_hours']>48), 'MHW', 'non_MHW')
nri['MHW'] = np.where((nri['MHW_land']==1)&(nri['mhw_hours']>48), 'MHW', 'non_MHW')

meta_cols = ['SEASON','NAME','land_time','land_lat','land_lon','RI','MHW']
meta = pd.concat([ri[meta_cols], nri[meta_cols]], ignore_index=True).drop_duplicates(['SEASON','NAME'])
meta['land_time'] = pd.to_datetime(meta['land_time'])

# ---------------------------------------------------------------------
# 4.  Pre‑compute TP for every storm & lead‑hour
# ---------------------------------------------------------------------
def storm_tp_series(row, cache):
    out = np.full(len(LEAD_HOURS), np.nan, float)
    year = row['land_time'].year
    if year not in cache:                           # lazy‑load ERA5 file once
        f = os.path.join(ERA5_DIR, f'era5_precipitation{year}.nc')
        cache[year] = xr.open_dataset(f) if os.path.exists(f) else None
    ds = cache[year]
    if ds is None:
        return out
    for i, lh in enumerate(LEAD_HOURS):
        when = row['land_time'] + timedelta(hours=int(lh))
        try:
            out[i] = tp_box_mean(ds, when, row['land_lat'], row['land_lon'])
        except Exception:
            continue
    return out

tp_mat = []
idx    = []
cache  = {}
for _, r in tqdm(meta.iterrows(), total=len(meta), desc='TP'):
    tp_mat.append(storm_tp_series(r, cache))
    idx.append((r.SEASON, r.NAME, r.RI, r.MHW))
for ds in cache.values():                            # close files
    if ds is not None:
        ds.close()

tp_df = (
    pd.DataFrame(tp_mat, index=pd.MultiIndex.from_tuples(idx,
               names=['SEASON','NAME','RI','MHW']), columns=LEAD_HOURS)
)

# ---------------------------------------------------------------------
# 5.  Generic bootstrap Δ‑function (same as WS_diff.py) :contentReference[oaicite:2]{index=2}
# ---------------------------------------------------------------------
def bootstrap_difference(df, level, label1, label2,
                         n_boot=1000, alpha=0.05, seed=0):
    rng  = np.random.default_rng(seed)
    g1   = df[df.index.get_level_values(level)==label1]
    g2   = df[df.index.get_level_values(level)==label2]
    dobs = g1.mean() - g2.mean()

    boot = np.empty((n_boot, len(LEAD_HOURS)))
    i1   = g1.index.unique()
    i2   = g2.index.unique()
    for b in tqdm(range(n_boot), desc=f'boot {label1}-{label2}', ncols=80):
        s1 = g1.loc[rng.choice(i1, len(i1), replace=True)]
        s2 = g2.loc[rng.choice(i2, len(i2), replace=True)]
        boot[b] = s1.mean() - s2.mean()

    lo = np.percentile(boot, 100*alpha/2, axis=0)
    hi = np.percentile(boot, 100*(1-alpha/2), axis=0)
    p  = (np.abs(boot) >= np.abs(dobs.values)).mean(axis=0)
    return pd.DataFrame({'Δ_obs': dobs, 'CI_lo': lo, 'CI_hi': hi, 'p': p},
                        index=LEAD_HOURS)

res_RI  = bootstrap_difference(tp_df, 'RI',  'RI',      'non_RI',
                               BOOTSTRAPS, ALPHA)
res_MHW = bootstrap_difference(tp_df, 'MHW', 'MHW',     'non_MHW',
                               BOOTSTRAPS, ALPHA)

# ---------------------------------------------------------------------
# 6.  Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,4))
for res, col, lab in [(res_RI,  '#4a90e2', 'Δ  RI − non‑RI'),
                      (res_MHW, '#e94e4e', 'Δ  MHW − non‑MHW')]:
    ax.fill_between(res.index/24, res.CI_lo, res.CI_hi, color=col, alpha=0.25)
    ax.plot(res.index/24, res.Δ_obs, lw=2, color=col, label=lab)

ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Time relative to landfall (days)')
ax.set_ylabel('Δ mean TP (mm / 3 h)')
ax.set_xticks(np.arange(-5,3))
ax.set_xticklabels(['-5d','-4d','-3d','-2d','-1d','LF','+1d','+2d'])
ax.set_title('Bootstrap CI for mean precipitation differences')
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 7.  Print windows with significant Δ>0
# ---------------------------------------------------------------------
for name, res in [('RI', res_RI), ('MHW', res_MHW)]:
    sig = res[(res.CI_lo>0) & (res.p<ALPHA)]
    print(f'\nLead hours with CI>0 for {name}:')
    print(sig.index.tolist())
