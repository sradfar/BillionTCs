# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script analyzes wind speed evolution of tropical cyclones across four 
# categories defined by the occurrence of rapid intensification (RI) and 
# marine heatwave (MHW) influence. It calculates lead-hour aligned mean wind 
# speeds with 95% confidence intervals and generates a comparative plot.
#
# Outputs:
# - A line plot with shaded confidence intervals showing the temporal evolution 
#   of mean wind speed (Ws) for:
#     • RI with MHW
#     • RI without MHW
#     • No RI with MHW
#     • No RI without MHW
# - The figure is saved as 'wind_speed_evolution_fast.pdf' and is intended for 
#   inclusion in the supplementary materials of the Science Advances paper.
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import t
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# 1.  File paths  (<<< CHANGE THESE >>>)
# ---------------------------------------------------------------------
RI_CSV      = r'.../with_RI_land_info.csv'
NON_RI_CSV  = r'.../non_RI_land_info.csv'
IBTRACS_CSV = r'.../ibtracs_filled.csv'

# ---------------------------------------------------------------------
# 2.  Lead‑time grid  (−5 d … +2 d, every 3 h)
# ---------------------------------------------------------------------
LEAD_HOURS = np.arange(-120, 49, 3)           # hours
LEAD_LABEL = LEAD_HOURS / 24                  # days for x‑axis

# ---------------------------------------------------------------------
# 3.  Read storm‑level metadata and assign the four categories
# ---------------------------------------------------------------------
ri  = pd.read_csv(RI_CSV)
nri = pd.read_csv(NON_RI_CSV)

# --- replicate your original filters ---------------------------------
ri  = ri[ ri['land'] == 1 ].copy()

nri = nri[ nri['land'] == 1].copy()

# --- four sub‑groups --------------------------------------------------
ri_mhw        = ri[  (ri['MHW_land']==1) & (ri['mhw_hours'] > 48) ]
ri_nomhw      = ri[  (ri['MHW_land']==0) & (ri['mhw_hours'] == 0) ]
nori_mhw      = nri[ (nri['MHW_land']==1) & (nri['mhw_hours'] > 48) ]
nori_nomhw    = nri[ (nri['MHW_land']==0) & (nri['mhw_hours'] == 0) ]

def meta(df, label):
    out = df[['SEASON','NAME','land_time']].drop_duplicates().copy()
    out['cat'] = label
    return out

meta_all = pd.concat([
    meta(ri_mhw,      'RI with MHW'),
    meta(ri_nomhw,    'RI without MHW'),
    meta(nori_mhw,    'No RI with MHW'),
    meta(nori_nomhw,  'No RI without MHW')
], ignore_index=True)

meta_all['land_time'] = pd.to_datetime(meta_all['land_time'])
meta_all['cat'] = pd.Categorical(
    meta_all['cat'],
    categories=['RI with MHW','RI without MHW',
                'No RI with MHW','No RI without MHW'],
    ordered=True
)

# ---------------------------------------------------------------------
# 4.  Merge metadata onto IBTrACS & compute lead hours
# ---------------------------------------------------------------------
trk = pd.read_csv(IBTRACS_CSV,
                  usecols=['SEASON','NAME','ISO_TIME','WIND_SPEED'])
trk['ISO_TIME'] = pd.to_datetime(trk['ISO_TIME'])

trk = trk.merge(meta_all, on=['SEASON','NAME'], how='inner')

trk['lead_hr'] = (
    (trk['ISO_TIME'] - trk['land_time']).dt.total_seconds()/3600
).round().astype(int)

trk = trk[trk['lead_hr'].between(LEAD_HOURS.min(), LEAD_HOURS.max())]

# snap to the 3‑hour grid:  e.g. −23 → −24
trk['lead_hr'] = (np.floor(trk['lead_hr']/3)*3).astype(int)

# ---------------------------------------------------------------------
# 5.  Aggregate → mean, count, std   (vectorised)
# ---------------------------------------------------------------------
agg = (
    trk.groupby(['cat','lead_hr'])['WIND_SPEED']
       .agg(['mean','count','std'])
       .reindex(
           pd.MultiIndex.from_product(
               [meta_all['cat'].cat.categories, LEAD_HOURS],
               names=['cat','lead_hr']
           )
       )
)

# 95 % CI half‑width:  t_{α/2, n‑1} * s / √n
crit = t.ppf(0.975, agg['count']-1)
agg['ci'] = crit * agg['std'] / np.sqrt(agg['count'])

# ---------------------------------------------------------------------
# 6.  Pivot for plotting
# ---------------------------------------------------------------------
mean = agg['mean'].unstack('cat')
ci   = agg['ci'  ].unstack('cat')

# ---------------------------------------------------------------------
# 7.  Plot
# ---------------------------------------------------------------------
COL = {'RI with MHW':'#ee5253',      # red
       'RI without MHW':'#6c5ce7',   # violet
       'No RI with MHW':'#ff9f43',   # orange
       'No RI without MHW':'#54a0ff'}# blue

plt.figure(figsize=(10,5))

for cat in mean.columns:
    m      = mean[cat].values
    ci_lo  = m - ci[cat].values
    ci_hi  = m + ci[cat].values
    plt.fill_between(LEAD_LABEL, ci_lo, ci_hi, color=COL[cat], alpha=0.2)
    plt.plot(LEAD_LABEL, m, color=COL[cat], lw=2)

# --- axes formatting --------------------------------------------------
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(10))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(2))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.25))

plt.xticks(np.arange(-5,3),
           ['-5 d','-4 d','-3 d','-2 d','-1 d','Landfall','+1 d','+2 d'])

plt.xlim(-5,2)
plt.xlabel('Time span relative to landfall (days)', fontsize=14)
plt.ylabel('Mean Ws (knots)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)

# --- custom legend (matches your reference style) --------------------
handles = [
    Patch(facecolor=COL['RI with MHW'],      edgecolor='none', label='RI with MHW'),
    Patch(facecolor=COL['RI without MHW'],   edgecolor='none', label='RI without MHW'),
    Patch(facecolor=COL['No RI with MHW'],   edgecolor='none', label='No RI with MHW'),
    Patch(facecolor=COL['No RI without MHW'],edgecolor='none', label='No RI without MHW')
]

legend = plt.legend(handles=handles,
                    loc='upper left',
                    bbox_to_anchor=(0.01, 0.99),
                    frameon=False,
                    fontsize=12,
                    handlelength=1.4,
                    handleheight=1.4,
                    borderaxespad=0.0,
                    labelspacing=0.3)

# (optional) centre the colour boxes vertically
for p in legend.get_patches():
    p.set_height(10)   # points
    p.set_y(-2)

# ---------------------------------------------------------------------
plt.tight_layout()
plt.savefig('wind_speed_evolution_fast.pdf', dpi=300)
plt.show()
