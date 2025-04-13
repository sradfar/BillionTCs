# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script calculates potential intensity (PI) time series for two 2019 cyclones 
# (IDAI and KENNETH) using ERA5 atmospheric profiles and the pyPI module. 
# It visualizes the PI time evolution for both storms.
#
# Outputs:
# - A dual-panel PNG figure of PI evolution: 'two_hurricanes_pi.png'
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tcpyPI import pi  # potential intensity function from pyPI
import matplotlib.dates as mdates

# --- Settings: Update these paths as needed ---
# ERA5 profile file (temperature and specific humidity on pressure levels)
profile_file = '.../2019ERA5_q_temp_pressure.nc'
# ERA5 surface file (mslp and sst)
surface_file = '.../2019ERA5_mslp_sst.nc'
# IBTrACS track file
ibtracs_file = '.../ibtracs_filled.csv'

# --- Load the ERA5 data using xarray ---
ds_profile = xr.open_dataset(profile_file)
ds_surface = xr.open_dataset(surface_file)

# --- Load the ibtracs data ---
ibtracs_data = pd.read_csv(ibtracs_file, parse_dates=['ISO_TIME'])

# --- Filter the ibtracs tracks for IAN and AGATHA (2022) ---
idai_track = ibtracs_data[(ibtracs_data['SEASON'] == 2019) & (ibtracs_data['NAME'] == 'IDAI')]
kenneth_track = ibtracs_data[(ibtracs_data['SEASON'] == 2019) & (ibtracs_data['NAME'] == 'KENNETH')]

# Ensure tracks are sorted by time
idai_track = idai_track.sort_values('ISO_TIME')
kenneth_track = kenneth_track.sort_values('ISO_TIME')

# --- Define a helper function to extract environmental data at a given track point ---
def extract_environment(lat, lon, time_point):
    """
    Given latitude, longitude, and time (as a numpy.datetime64 or pandas Timestamp),
    interpolate the ERA5 profile and surface data to extract:
      - Sea surface temperature (SST) in °C
      - Mean sea level pressure (msl) in hPa
      - Pressure levels (p) in hPa (assumed constant across the domain)
      - Temperature profile (T) in °C (converted from K)
      - Mixing ratio profile (r) in g/kg (approximate conversion from specific humidity)
    """
    time_val = np.datetime64(time_point)
    
    # Use nearest neighbor in time and space
    prof_point = ds_profile.sel(valid_time=time_val, latitude=lat, longitude=lon, method="nearest")
    surf_point = ds_surface.sel(valid_time=time_val, latitude=lat, longitude=lon, method="nearest")
    
    # Pressure levels (in hPa) from the profile dataset
    p = prof_point['pressure_level'].values  # in hPa

    # Temperature profile: convert from Kelvin to Celsius
    T_profile = prof_point['t'].values - 273.15

    # Specific humidity profile from ERA5 (in kg/kg); approximate mixing ratio (g/kg)
    q_profile = prof_point['q'].values
    r_profile = 1000 * q_profile  # valid when q << 1

    # Surface variables: sst in K -> °C; msl in Pa -> hPa
    sst = surf_point['sst'].values - 273.15
    msl = surf_point['msl'].values / 100.0

    return sst, msl, p, T_profile, r_profile

# --- Define a function to compute PI along a storm track ---
def compute_PI_along_track(track_df, time_col='ISO_TIME', lat_col='LAT', lon_col='LON'):
    """
    For each track point in the DataFrame (which must contain time, latitude, and longitude),
    extract the environmental data and compute potential intensity (Vmax) using pyPI.
    Returns a pandas Series of Vmax values indexed by the track time.
    """
    PI_vals = []
    times = []
    for idx, row in track_df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        time_point = row[time_col]
        try:
            sst, msl, p, T_profile, r_profile = extract_environment(lat, lon, time_point)
            # Compute potential intensity using pyPI. Adjust parameters if needed.
            Vmax, pmin, flag, T0, OTL = pi(sst, msl, p, T_profile, r_profile,
                                            CKCD=0.9, ascent_flag=0, diss_flag=1,
                                            V_reduc=0.8, ptop=50, miss_handle=1)
            PI_vals.append(Vmax)
        except Exception as e:
            print(f"Error at index {idx} for time {time_point}: {e}")
            PI_vals.append(np.nan)
        times.append(time_point)
    return pd.Series(PI_vals, index=pd.to_datetime(times))

# --- Compute PI time series for each storm ---
PI_idai = compute_PI_along_track(idai_track)
PI_kenneth = compute_PI_along_track(kenneth_track)

# Filter data starting from March 9, 2019
start_date = pd.Timestamp('2019-03-09')
PI_idai = PI_idai[PI_idai.index >= start_date]

# --- Plot the time series: Top panel for IAN, bottom panel for AGATHA ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot IAN
ax1.plot(PI_idai.index, PI_idai, marker='o', linestyle='-', label='IDAI (2019)')
ax1.set_ylim([50, 100])  # <-- y-limits between 50 and 100
ax1.set_ylabel('Potential intensity (m/s)')
ax1.legend()
ax1.grid(True)

# Configure daily ticks on the x-axis
ax1.xaxis.set_major_locator(mdates.DayLocator())                 # Major tick = start of each day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))     # Format as "May-28", etc.

# Plot AGATHA
ax2.plot(PI_kenneth.index, PI_kenneth, marker='o', linestyle='-', color='orange', label='KENNETH (2019)')
ax2.set_ylim([50, 100])  # <-- y-limits between 50 and 100
ax2.set_ylabel('Potential intensity (m/s)')
ax2.set_xlabel('Time')
ax2.legend()
ax2.grid(True)

# Configure daily ticks on the x-axis for Agatha
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

plt.tight_layout()
plt.savefig('.../two_hurricanes_pi.png', dpi=600)
plt.show()
