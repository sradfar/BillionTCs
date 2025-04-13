# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script evaluates the sensitivity of precipitation (TP) distributions to 
# different marine heatwave (MHW) definitions. It calculates mean rainfall from 
# ERA5 data for each landfalling cyclone and visualizes the distributions.
#
# Outputs:
# - A combined boxplot figure (8 time points × 4 TC categories × 3 MHW criteria)
# - Rainfall statistics are based on a 5-degree box around landfall
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

def adjust_longitude(lon):
    """Adjusts longitude to 0-360 range if it's in -180 to 180 range."""
    return lon % 360

def nearest_era5_grid_point(lat, lon, ds):
    """Finds the nearest grid points in the dataset for given lat and lon."""
    adjusted_lon = adjust_longitude(lon)
    abs_lat_diff = np.abs(ds['latitude'] - lat)
    abs_lon_diff = np.abs(ds['longitude'] - adjusted_lon)
    nearest_lat = ds['latitude'].isel(latitude=abs_lat_diff.argmin()).values.item()
    nearest_lon = ds['longitude'].isel(longitude=abs_lon_diff.argmin()).values.item()
    return nearest_lat, nearest_lon

def mean_precipitation_around_grid(filename, target_time, lat, lon, lead_hours):
    """Calculates mean precipitation around the given lat and lon for the specified time and lead hours."""
    with xr.open_dataset(filename) as ds:
        adjusted_lon = adjust_longitude(lon)
        nearest_lat, nearest_lon = nearest_era5_grid_point(lat, adjusted_lon, ds)
        results = {}
        for lead in lead_hours:
            adjusted_time = target_time + timedelta(hours=lead)
            try:
                # Select a 2-degree box around the nearest point
                lat_slice = slice(nearest_lat + 5, nearest_lat - 5)
                lon_slice = slice(nearest_lon - 5, nearest_lon + 5)

                # Select data for each of the 3 hours separately using method='nearest' and sum up
                tp_sum = 0
                for hour_offset in range(3):
                    hour_time = adjusted_time + timedelta(hours=hour_offset)
                    selected_data = ds.sel(time=hour_time, method='nearest')
                    subset = selected_data.sel(latitude=lat_slice, longitude=lon_slice)
                    tp_sum += subset.tp.mean().values.item() * 1000  # Convert from m to mm

                results[lead] = tp_sum
            except ValueError:
                results[lead] = np.nan
        return results

def count_unique_tcs(grouped_data):
    """Counts the number of unique TCs in a grouped DataFrame"""
    unique_tcs = set()
    for (season, name, _) in grouped_data.groups.keys():
        unique_tcs.add((season, name))
    return len(unique_tcs)

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

def plot_tp_boxplot_combined(base_dir, datasets, lead_hours):
    """
    Create a single figure with boxplots for all MHW conditions and days.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for ERA5 data
    datasets : list
        List of tuples (condition_name, [grouped_with_RI_MHW, grouped_with_RI_no_MHW, grouped_without_RI_MHW, grouped_without_RI_no_MHW])
    lead_hours : list
        List of lead hours to analyze
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
            
            # Iterate with tqdm for progress tracking
            for (season, name, _), sub_df in tqdm(group, desc=f'Processing {condition_name} - {labels[j]}'):
                representative_row = sub_df.iloc[0]
                year = representative_row['land_time'].year
                filename = os.path.join(base_dir, f"era5_precipitation{year}.nc")
                if not os.path.exists(filename):
                    continue  # Skip if the file does not exist
                lat = representative_row['land_lat']
                lon = representative_row['land_lon']
                target_time = representative_row['land_time']
                tp_data = mean_precipitation_around_grid(filename, target_time, lat, lon, lead_hours)
                
                # Organize data by day
                for day_idx in day_indices:
                    # Find all lead hours that correspond to this day
                    day_lead_hours = [lh for lh in lead_hours if lh // 24 == day_idx]
                    if day_lead_hours:
                        # Average the values for all hours in this day
                        day_values = [tp_data[lh] for lh in day_lead_hours if lh in tp_data and not np.isnan(tp_data[lh])]
                        if day_values:
                            day_data[day_idx][i][j].append(np.mean(day_values))
    
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
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(axis='y', which='major', labelsize=14)  # Add this line
    
    # Set x-ticks at the center of each day's group
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=0, ha='center', fontsize=14)
    ax.set_xlim(-0.5, len(days) - 0.5)
    
    # Labels and title
    ax.set_xlabel('Time span relative to Landfall', fontsize=16)
    ax.set_ylabel('Mean Tp (mm/hr)', fontsize=16)
    
    # Add title as a subtitle positioned above the x-axis
    ax.text(0.5, 0.02, '(d) Tp ranges for MHW conditions', transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add legend with vertical layout at top left, without background box
    ax.legend(handles=legend_handles, fontsize=14, loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig('rainfall_boxplot_combined.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# Define lead_hours and base directory for ERA5 data
lead_hours = list(range(-5 * 24, 2 * 24 + 1, 3))
base_dir = '.../'
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

# Create the boxplot visualization
plot_tp_boxplot_combined(base_dir, datasets, lead_hours)
