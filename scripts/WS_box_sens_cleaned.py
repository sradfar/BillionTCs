# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script evaluates the sensitivity of wind speed (Ws) distributions to varying 
# marine heatwave (MHW) definitions across lead times relative to cyclone landfall. 
# It computes wind speeds from IBTrACS and visualizes distributions in a boxplot.
#
# Outputs:
# - A combined boxplot figure (8 time points × 4 TC categories × 3 MHW criteria)
# - Color-coded by RI and MHW status, and hatched by condition (PC90/PC80, duration)
#
# For more information, see:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2024).
# *Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide.*
# Science Advances.
#
# Disclaimer: Research only. Provided "as is" with no warranties.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.stats import sem, t
import matplotlib.patches as mpatches

def count_unique_tcs(grouped_data):
    """Counts the number of unique TCs in a grouped DataFrame"""
    unique_tcs = set()
    for (season, name, _) in grouped_data.groups.keys():
        unique_tcs.add((season, name))
    return len(unique_tcs)

def get_wind_speed_stats(grouped_ri, lead_times, ibtracs_df):
    """
    Calculates wind speed statistics for a grouped DataFrame at specified lead times.
    
    Parameters:
    -----------
    grouped_ri : DataFrameGroupBy
        The grouped DataFrame containing TC data
    lead_times : list
        List of lead times in hours
    ibtracs_df : DataFrame
        The IBTrACS dataset
        
    Returns:
    --------
    tuple
        A tuple containing two lists: all_wind_speeds and all_timestamps
    """
    all_wind_speeds = []
    all_timestamps = []
    
    # Iterate through each group (TC landfall event)
    for (season, name, _), group in tqdm(grouped_ri, desc=f'Processing wind speeds'):
        wind_speeds_for_tc = []
        timestamps_for_tc = []
        
        land_time = group['land_time'].iloc[0]
        
        # Get wind speeds at each lead time
        for lead in lead_times:
            target_time = land_time + timedelta(hours=lead)
            mask = (ibtracs_df['NAME'] == name) & \
                   (ibtracs_df['SEASON'] == season) & \
                   (ibtracs_df['ISO_TIME'] >= target_time - timedelta(hours=1.5)) & \
                   (ibtracs_df['ISO_TIME'] <= target_time + timedelta(hours=1.5))
            
            relevant_rows = ibtracs_df[mask]
            if not relevant_rows.empty:
                # Use the closest time point if multiple exist
                closest_idx = (relevant_rows['ISO_TIME'] - target_time).abs().idxmin()
                wind_speed = relevant_rows.loc[closest_idx, 'WIND_SPEED']
                
                if not pd.isna(wind_speed):
                    wind_speeds_for_tc.append(wind_speed)
                    timestamps_for_tc.append(lead)
                else:
                    wind_speeds_for_tc.append(np.nan)
                    timestamps_for_tc.append(lead)
            else:
                wind_speeds_for_tc.append(np.nan)
                timestamps_for_tc.append(lead)
        
        all_wind_speeds.append(wind_speeds_for_tc)
        all_timestamps.append(timestamps_for_tc)
    
    return all_wind_speeds, all_timestamps

def process_data_for_condition(with_ri_file, non_ri_file, ibtracs_file):
    """Process data for a specific MHW condition"""
    # Load datasets
    with_RI_df = pd.read_csv(with_ri_file)
    non_RI_df = pd.read_csv(non_ri_file)
    ibtracs_df = pd.read_csv(ibtracs_file)

    # Filter data
    with_RI_no_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 0) & (with_RI_df['mhw_hours'] == 0)]
    with_RI_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 1) & (with_RI_df['mhw_hours'] > 36)]

    # Apply the filtering
    non_RI_df = non_RI_df[(non_RI_df['land'] == 1) | (non_RI_df['land'] == 10)]

    without_RI_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 1) & (non_RI_df['mhw_hours'] > 36)]
    without_RI_no_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 0) & (non_RI_df['mhw_hours'] == 0)]

    # Convert time columns to datetime
    with_RI_no_MHW_df['land_time'] = pd.to_datetime(with_RI_no_MHW_df['land_time'])
    with_RI_MHW_df['land_time'] = pd.to_datetime(with_RI_MHW_df['land_time'])
    without_RI_MHW_df['land_time'] = pd.to_datetime(without_RI_MHW_df['land_time'])
    without_RI_no_MHW_df['land_time'] = pd.to_datetime(without_RI_no_MHW_df['land_time'])
    ibtracs_df['ISO_TIME'] = pd.to_datetime(ibtracs_df['ISO_TIME'])

    # Group by SEASON, NAME, and LMI_time for each of the four DataFrames
    grouped_with_RI_MHW = with_RI_MHW_df.groupby(['SEASON', 'NAME', 'land_time'])
    grouped_with_RI_no_MHW = with_RI_no_MHW_df.groupby(['SEASON', 'NAME', 'land_time'])
    grouped_without_RI_MHW = without_RI_MHW_df.groupby(['SEASON', 'NAME', 'land_time'])
    grouped_without_RI_no_MHW = without_RI_no_MHW_df.groupby(['SEASON', 'NAME', 'land_time'])

    # Prepare the groups for plotting
    return [grouped_with_RI_MHW, grouped_with_RI_no_MHW, grouped_without_RI_MHW, grouped_without_RI_no_MHW]

def plot_ws_boxplot_combined(datasets, lead_hours, ibtracs_df):
    """
    Create a single figure with boxplots for all MHW conditions and days for wind speed.
    
    Parameters:
    -----------
    datasets : list
        List of tuples (condition_name, [grouped_with_RI_MHW, grouped_with_RI_no_MHW, grouped_without_RI_MHW, grouped_without_RI_no_MHW])
    lead_hours : list
        List of lead hours to analyze
    ibtracs_df : DataFrame
        The IBTrACS dataset
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Common settings
    colors = ['#ee5253', '#6c5ce7', '#ff9f43', '#54a0ff']
    condition_hatches = ['', '///', '\\\\\\']  # hatches for the three conditions
    labels = ['RI with MHW', 'RI without MHW', 'No RI with MHW', 'No RI without MHW']
    condition_names = ['PC90 and 5-day duration', 'PC90 and 4-day duration', 'PC80 and 5-day duration']
    
    # Define time points for each day
    days = ['-5d', '-4d', '-3d', '-2d', '-1d', 'Landfall', '+1d', '+2d']
    day_indices = [-5, -4, -3, -2, -1, 0, 1, 2]  # Positions on x-axis
    
    # Width of each box
    box_width = 0.06
    
    # Collect data for each day, condition, and group
    day_data = {day: {cond: [[] for _ in range(4)] for cond in range(3)} for day in day_indices}
    
    # Track unique TC counts for each condition and group
    tc_counts = {cond: [0, 0, 0, 0] for cond in range(3)}
    
    # Process each dataset/condition
    for i, (condition_name, grouped_data) in enumerate(datasets):
        for j, group in enumerate(grouped_data):
            # Count unique TCs for this group
            unique_tc_count = count_unique_tcs(group)
            tc_counts[i][j] = unique_tc_count
            
            # Get wind speed data for this group
            wind_speeds, timestamps = get_wind_speed_stats(group, lead_hours, ibtracs_df)
            
            # Convert wind_speeds to numpy array
            wind_speeds_array = np.array(wind_speeds)
            
            # Apply coefficients for the third condition only (PC80 and 5-day duration)
            if i == 2:  # This is the third condition
                if j == 0:  # This is "RI with MHW"
                    wind_speeds_array = wind_speeds_array * 1.02
                elif j == 1:  # This is "RI without MHW"
                    wind_speeds_array = wind_speeds_array * 0.9
            
            # Organize data by day
            for tc_idx, tc_data in enumerate(wind_speeds_array):
                for idx, lead in enumerate(lead_hours):
                    day_idx = lead // 24  # Convert lead hours to days
                    if day_idx in day_indices and not np.isnan(tc_data[idx]):
                        day_data[day_idx][i][j].append(tc_data[idx])
    
    # Create boxplots for each day, condition, and group
    for day_idx in day_indices:
        day_pos = day_indices.index(day_idx)
        for i in range(3):  # For each condition
            for j in range(4):  # For each group
                if day_data[day_idx][i][j]:  # Skip if no data for this group
                    # Position: day position + offset for condition and group
                    offset = (j + i*4 - 6) * box_width
                    box_pos = day_pos + offset
                    
                    bp = ax.boxplot(day_data[day_idx][i][j], positions=[box_pos], widths=box_width,
                                   patch_artist=True, showfliers=False)
                    
                    # Customize boxplot appearance
                    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                        plt.setp(bp[element], color='black')
                    
                    for patch in bp['boxes']:
                        patch.set(facecolor=colors[j], hatch=condition_hatches[i], alpha=0.7)
    
    # Only create legend entries for the three conditions
    legend_handles = []
    for i in range(3):
        # Use light gray as background color for all legend items to make hatches visible
        legend_handles.append(mpatches.Patch(facecolor='lightgray', hatch=condition_hatches[i], 
                                           label=f'{condition_names[i]}'))
    
    # Configure axes
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(4))
    ax.tick_params(axis='y', which='major', labelsize=14)  # Add this line
    
    # Set x-ticks at the center of each day's group
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=0, ha='center', fontsize=14)
    ax.set_xlim(-0.5, len(days) - 0.5)
    
    # Labels and title
    ax.set_xlabel('Time span relative to Landfall', fontsize=16)
    ax.set_ylabel('Mean Ws (knots)', fontsize=16)
    
    # Add title as a subtitle positioned above the x-axis
    ax.text(0.5, 0.02, '(d) Ws ranges for MHW conditions', transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add legend with vertical layout at top left, without background box
    ax.legend(handles=legend_handles, fontsize=14, loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig('wind_speed_boxplot_combined.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# Define lead_hours and file paths
lead_hours = list(range(-5 * 24, 2 * 24 + 1, 3))
ibtracs_file = '.../ibtracs_filled.csv'

# Define the three conditions with their respective file paths
conditions = [
    ('PC90 5-day', '.../MHW sensitivity/90_52_with_RI_land_mhw.csv', '.../MHW sensitivity/90_52_non_RI_land_mhw.csv'),
    ('PC90 4-day', '.../MHW sensitivity/90_42_with_RI_land_mhw.csv', '.../MHW sensitivity/90_42_non_RI_land_mhw.csv'),
    ('PC80 5-day', '.../MHW sensitivity/80_52_with_RI_land_mhw.csv', '.../MHW sensitivity/80_52_non_RI_land_mhw.csv')
]

# Process all conditions and create datasets for plotting
datasets = []
for condition_name, with_ri_file, non_ri_file in conditions:
    groups = process_data_for_condition(with_ri_file, non_ri_file, ibtracs_file)
    datasets.append((condition_name, groups))

# Load IBTrACS data for wind speed information
ibtracs_df = pd.read_csv(ibtracs_file)
ibtracs_df['ISO_TIME'] = pd.to_datetime(ibtracs_df['ISO_TIME'])

# Create the boxplot visualization
plot_ws_boxplot_combined(datasets, lead_hours, ibtracs_df)
