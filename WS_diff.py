# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script compares the wind speed evolution of landfalling tropical cyclones 
# categorized by rapid intensification (RI) and marine heatwave (MHW) influences.
# It computes the lead-hour differences in wind speed and performs bootstrap-based 
# significance testing for RI vs. non-RI and MHW vs. non-MHW scenarios. The script 
# outputs a confidence interval plot of mean wind speed differences over time.
#
# Outputs:
# - A plot of bootstrap-derived confidence intervals comparing wind speed 
#   evolution for RI vs. non-RI and MHW vs. non-MHW groups.
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# User parameters
RI_CSV      = r'.../with_RI_land_info.csv'
NON_RI_CSV  = r'.../non_RI_land_info.csv'
IBTRACS_CSV = r'.../ibtracs_filled.csv'

LEAD_HOURS  = np.arange(-120,  49,  3)
BOOTSTRAPS  = 1000
ALPHA       = 0.05

# Load and label metadata
ri_df     = pd.read_csv(RI_CSV)
non_ri_df = pd.read_csv(NON_RI_CSV)

ri_df     = ri_df[ri_df['land'] == 1].copy()
non_ri_df = non_ri_df[(non_ri_df['land'] == 1) | (non_ri_df['land'] == 10)].copy()

ri_df['RI']      = 'RI'
non_ri_df['RI']  = 'non_RI'
ri_df['MHW']     = np.where((ri_df['MHW_land']==1) & (ri_df['mhw_hours']>48), 'MHW', 'non_MHW')
non_ri_df['MHW'] = np.where((non_ri_df['MHW_land']==1) & (non_ri_df['mhw_hours']>48), 'MHW', 'non_MHW')

meta_cols = ['SEASON', 'NAME', 'land_time', 'RI', 'MHW']
meta = pd.concat([ri_df[meta_cols], non_ri_df[meta_cols]], ignore_index=True).drop_duplicates(['SEASON', 'NAME'])
meta['land_time'] = pd.to_datetime(meta['land_time'])

# Merge with track data
trk = pd.read_csv(IBTRACS_CSV, usecols=['SEASON', 'NAME', 'ISO_TIME', 'WIND_SPEED'])
trk['ISO_TIME'] = pd.to_datetime(trk['ISO_TIME'])
trk = trk.merge(meta, on=['SEASON', 'NAME'], how='inner')
trk['lead_hr'] = ((trk['ISO_TIME'] - trk['land_time']).dt.total_seconds() / 3600).round().astype(int)
trk = trk[trk['lead_hr'].between(LEAD_HOURS.min(), LEAD_HOURS.max())]
trk['lead_hr'] = (np.floor(trk['lead_hr'] / 3) * 3).astype(int)

# Pivot data
pivot = trk.pivot_table(index=['SEASON','NAME','RI','MHW'], columns='lead_hr', values='WIND_SPEED', aggfunc='mean')
pivot = pivot.reindex(columns=LEAD_HOURS)

# Compute peak wind time
peak_time = trk.sort_values(['SEASON','NAME','ISO_TIME']).groupby(['SEASON','NAME']) \
    .apply(lambda g: g.loc[g['WIND_SPEED'].idxmax(), 'lead_hr']).to_frame('peak_hr') \
    .merge(meta[['SEASON','NAME','RI','MHW']], on=['SEASON','NAME'])

ri_peaks = peak_time[peak_time['RI'] == 'RI'].copy()
ri_peaks['group'] = np.where(ri_peaks['MHW']=='MHW', 'RI with MHW', 'RI without MHW')
ri_peaks['hours_before_LF'] = -ri_peaks['peak_hr']
print(ri_peaks.groupby('group')['hours_before_LF'].describe())
u, p = mannwhitneyu(ri_peaks[ri_peaks['group']=='RI with MHW']['hours_before_LF'],
                    ri_peaks[ri_peaks['group']=='RI without MHW']['hours_before_LF'])
print(f'Mann‑Whitney p = {p:.4f}')

# Bootstrap CI
def bootstrap_difference(df, level_name, label1, label2, n_boot=1000, alpha=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    g1 = df[df.index.get_level_values(level_name) == label1]
    g2 = df[df.index.get_level_values(level_name) == label2]
    mu1, mu2 = g1.mean(), g2.mean()
    delta_obs = mu1 - mu2
    boot = np.empty((n_boot, len(LEAD_HOURS)), float)
    for b in tqdm(range(n_boot), desc=f'boot {label1}-{label2}', ncols=80):
        s1 = g1.loc[rng.choice(g1.index.unique(), size=len(g1.index.unique()), replace=True)]
        s2 = g2.loc[rng.choice(g2.index.unique(), size=len(g2.index.unique()), replace=True)]
        boot[b] = s1.mean() - s2.mean()
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)], axis=0)
    p = (np.abs(boot) >= np.abs(delta_obs.values)).mean(axis=0)
    return pd.DataFrame({'Δ_obs': delta_obs, 'CI_lo': lo, 'CI_hi': hi, 'p_boot': p}, index=LEAD_HOURS)

res_RI  = bootstrap_difference(pivot, 'RI',  'RI',     'non_RI',   n_boot=BOOTSTRAPS, alpha=ALPHA)
res_MHW = bootstrap_difference(pivot, 'MHW', 'MHW',    'non_MHW', n_boot=BOOTSTRAPS, alpha=ALPHA)

# Plot
fig, ax = plt.subplots(figsize=(8,4))
ax.fill_between(res_RI.index/24, res_RI['CI_lo'], res_RI['CI_hi'], color='#4a90e2', alpha=0.25)
ax.plot(res_RI.index/24, res_RI['Δ_obs'], lw=2, color='#4a90e2', label='Δ  RI − non‑RI')
ax.fill_between(res_MHW.index/24, res_MHW['CI_lo'], res_MHW['CI_hi'], color='#e94e4e', alpha=0.20)
ax.plot(res_MHW.index/24, res_MHW['Δ_obs'], lw=2, color='#e94e4e', label='Δ  MHW − non‑MHW')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Time relative to landfall (days)')
ax.set_ylabel('Δ mean wind speed (knots)')
ax.set_xticks(np.arange(-5, 3))
ax.set_xticklabels(['-5d','-4d','-3d','-2d','-1d','LF','+1d','+2d'])
ax.set_title('Bootstrap CI for mean wind‑speed differences')
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# Report significant lead hours
for name, res in [('RI', res_RI), ('MHW', res_MHW)]:
    ok = res[(res['CI_lo'] > 0) & (res['p_boot'] < ALPHA)]
    print(f'\nLead hours with CI>0 for {name}:')
    print(ok.index.tolist())
