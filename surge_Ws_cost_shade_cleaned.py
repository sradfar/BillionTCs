# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script visualizes joint exceedance probabilities of storm surge and wind speed 
# and their relationship to tropical cyclone economic damages. It creates copula-based 
# shaded subplots across four RI/MHW group combinations.
#
# Outputs:
# - A multi-panel PDF figure: 'Surge_Ws_cost.pdf'
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from scipy.stats import rankdata

# Update the default settings for all plots
plt.rcParams.update({
    'font.size': 16,        # Adjust to change the font size globally
    'axes.titlesize': 16,   # Adjust to change the subplot title size
    'axes.labelsize': 14,   # Adjust to change the x and y labels size
    'xtick.labelsize': 12,   # Adjust to change the x tick label size
    'ytick.labelsize': 12    # Adjust to change the y tick label size
})

# Load the main dataset
data = pd.read_csv('.../TC cost/updated_aggregated_with_built_v_agg.csv')

# Load datasets
with_RI_df = pd.read_csv('.../with_RI_landfall_mhw_surge.csv')
non_RI_df = pd.read_csv('.../non_RI_landfall_mhw_surge.csv')
ibtracs_df = pd.read_csv('.../ibtracs_filled.csv')

RI_without_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 0) & (with_RI_df['mhw_hours'] == 0)]
RI_with_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 1) & (with_RI_df['mhw_hours'] > 48)]

# Apply the filtering
non_RI_df = non_RI_df[(non_RI_df['land'] == 1) | (non_RI_df['land'] == 10)]

without_RI_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 1) & (non_RI_df['mhw_hours'] > 48)]
without_RI_no_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 0) & (non_RI_df['mhw_hours'] == 0)]

RI_without_MHW_df['land_time'] = pd.to_datetime(RI_without_MHW_df['land_time'])
RI_with_MHW_df['land_time'] = pd.to_datetime(RI_with_MHW_df['land_time'])
without_RI_MHW_df['land_time'] = pd.to_datetime(without_RI_MHW_df['land_time'])
without_RI_no_MHW_df['land_time'] = pd.to_datetime(without_RI_no_MHW_df['land_time'])

# Define cost levels
cost_levels = [1e6, 10e6, 50e6, 100e6, 500e6, 1e9, 10e9, 50e9, 100e9]

# Normalize the damage data to fit the color range
data['Damage Level'] = np.digitize(data["Total Damage, Adjusted ('000 US$)"] * 1000, cost_levels)

# Function to filter main data based on subset
def filter_data(main_df, subset_df):
    return pd.merge(main_df, subset_df[['NAME', 'SEASON']], on=['NAME', 'SEASON'], how='inner')

# Filter data for each subset
filtered_RI_MHW = filter_data(data, RI_with_MHW_df)
filtered_RI_noMHW = filter_data(data, RI_without_MHW_df)
filtered_noRI_MHW = filter_data(data, without_RI_MHW_df)
filtered_noRI_noMHW = filter_data(data, without_RI_no_MHW_df)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
datasets = [(filtered_RI_MHW, 'Reds'), (filtered_RI_noMHW, 'Reds'),
            (filtered_noRI_MHW, 'Reds'), (filtered_noRI_noMHW, 'Reds')]

# Determine global min and max damage levels for the color range across all subplots
global_min_damage_level = data['Damage Level'].min()
global_max_damage_level = data['Damage Level'].max()

cost_ranges = [
    (100e9, np.inf),  # >100B
    (10e9, 100e9),    # 10B-100B
    (1e9, 10e9),      # 1B-10B
    (100e6, 1e9),     # 100M-1B
    (10e6, 100e6),    # 10-100M
    (1e6, 10e6),      # 1-10M
    (0, 1e6)          # <1M
]

# Function to count unique TCs in each cost range
def count_tcs(data, cost_ranges):
    # Create an empty list for TC counts per range
    counts = [0] * len(cost_ranges)
    # Create a set for tracking unique TCs
    unique_tcs = set()

    for index, row in data.iterrows():
        total_damage = row["Total Damage, Adjusted ('000 US$)"] * 1000
        tc_identifier = (row['NAME'], row['SEASON'])

        if tc_identifier not in unique_tcs:
            for i, (low, high) in enumerate(cost_ranges):
                if low < total_damage <= high:
                    counts[i] += 1
                    unique_tcs.add(tc_identifier)
                    break

    return counts

# Keep a list of colorbar references
cbar_refs = []
marker_size = 100

# Define probabilities for the copula lines
probabilities = [0.50, 0.80, 0.90, 0.95]

# Dataset labels corresponding to each subplot
dataset_labels = [
    'RI with MHW',
    'RI without MHW',
    'No RI with MHW',
    'No RI without MHW'
]

# Colors for empirical and independent copula lines
empirical_color = 'darkgray'
special_color = 'red'

subplot_labels = ['a', 'b', 'c', 'd']  # Labels for the subplots

# Labels now include the new categories, ordered from highest to lowest
labels = [">100B", "10-100B", "1-10B", "100M-1B", "10-100M", "1-10M", "<1M"]

# Colors for the shaded areas in each subplot
shaded_colors = ['goldenrod', 'goldenrod', 'goldenrod', 'goldenrod']
# shaded_colors = ['lightcoral', 'lightcoral', 'goldenrod', 'goldenrod']

subplot_labels = ['a', 'b', 'c', 'd']  # Labels for the subplots

# Adjusted position for the TC count text
text_pos_x = 0.99  # more to the right
text_pos_y_start = 0.98  # a little lower
text_line_spacing = 0.035  # smaller space between lines

for i, (ax, (dataset, cmap), dataset_label, shade_color) in enumerate(zip(axes.flatten(), datasets, dataset_labels, shaded_colors)):
    # Disable grid lines for the current subplot
    ax.grid(False)
    
    # Separate data by source for different markers
    em_dat_data = dataset[dataset['Source'] == 'EM-DAT']
    other_data = dataset[dataset['Source'] != 'EM-DAT']
    
    # Compute ranks
    x = dataset['avg_WS']
    y = dataset['max_surge']
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    # Rank data and transform to [0, 1] (Copula space)
    u_data = rankdata(x) / (len(x) + 1)
    v_data = rankdata(y) / (len(y) + 1)
    
    # Grid for contouring
    u_grid = np.linspace(0, 1, 300)
    v_grid = np.linspace(0, 1, 300)
    U, V = np.meshgrid(u_grid, v_grid)
    
    # Calculate copula values on the grid
    copula_values = np.array([
        [np.mean((u_data >= U[j, k]) & (v_data >= V[j, k])) for k in range(U.shape[1])]
        for j in range(U.shape[0])
    ])
    
    # Define contour levels for exceedance (e.g., 1%, 5%, 10%, 20%)
    contour_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
    
    x_quantiles = np.quantile(x, u_grid)
    y_quantiles = np.quantile(y, v_grid)
    X, Y = np.meshgrid(x_quantiles, y_quantiles)
    
    # Plot the filled contours using the 'Reds' colormap
    contour_filled = ax.contourf(X, Y, copula_values, levels=contour_levels, cmap='Blues_r')
    
    # Plot the contour lines on top for clarity
    # ax.contour(X, Y, copula_values, levels=contour_levels, colors='green', linestyles='-', linewidths=1.5)
    
    # Add the colorbar for filled contours below the upper right text box
    if i == 0:  # Only add the colorbar once for the first subplot
        # Create a new axes instance for the colorbar
        cbar_ax = fig.add_axes([0.23, 0.84, 0.14, 0.01])  # [left, bottom, width, height]
        cbar_filled = fig.colorbar(contour_filled, cax=cbar_ax, orientation='horizontal')
        cbar_filled.set_label('Joint exceedance probability', fontsize=11)
        cbar_filled.ax.tick_params(labelsize=12)
        
    if i == 1:  # Only add the colorbar once for the first subplot
        # Create a new axes instance for the colorbar
        cbar_ax = fig.add_axes([0.724, 0.84, 0.14, 0.01])  # [left, bottom, width, height]
        cbar_filled = fig.colorbar(contour_filled, cax=cbar_ax, orientation='horizontal')
        cbar_filled.set_label('Joint exceedance probability', fontsize=11)
        cbar_filled.ax.tick_params(labelsize=12)
        
    if i == 2:  # Only add the colorbar once for the first subplot
        # Create a new axes instance for the colorbar
        cbar_ax = fig.add_axes([0.23, 0.348, 0.14, 0.01])  # [left, bottom, width, height]
        cbar_filled = fig.colorbar(contour_filled, cax=cbar_ax, orientation='horizontal')
        cbar_filled.set_label('Joint exceedance probability', fontsize=11)
        cbar_filled.ax.tick_params(labelsize=12)
        
    if i == 3:  # Only add the colorbar once for the first subplot
        # Create a new axes instance for the colorbar
        cbar_ax = fig.add_axes([0.724, 0.348, 0.14, 0.01])  # [left, bottom, width, height]
        cbar_filled = fig.colorbar(contour_filled, cax=cbar_ax, orientation='horizontal')
        cbar_filled.set_label('Joint exceedance probability', fontsize=11)
        cbar_filled.ax.tick_params(labelsize=12)
        
    # Reset the titles
    ax.set_title(dataset_label)

    # Calculate the total number of TCs for the current subset
    total_tcs = len(dataset.drop_duplicates(subset=['NAME', 'SEASON']))
    
    # Display the counts
    subset_count = count_tcs(dataset, cost_ranges)
    
    for j, (count, range_label) in enumerate(zip(subset_count, labels)):
        percentage = (count / total_tcs) * 100 if total_tcs > 0 else 0
        ax.text(text_pos_x, text_pos_y_start - (j * text_line_spacing),
                f"{range_label}: {count} TCs ({percentage:.1f}%)",
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontsize=11, color='black', zorder=3)
    
    # # Shade the area where precipitation > 5
    # ax.fill_betweenx(y=[1.5, ax.get_ylim()[1]], x1=0, x2=ax.get_xlim()[1], color=shade_color, alpha=0.1)

    # # Shade the area where wind speed > 100
    # ax.fill_between(x=[100, ax.get_xlim()[1]], y1=0, y2=ax.get_ylim()[1], color=shade_color, alpha=0.1)
    
    # Reset the titles
    ax.set_title(dataset_label)
    
    # Add the subplot label to the top left corner
    ax.text(0.02, 0.98, f'{subplot_labels[i]})', transform=ax.transAxes, fontsize=14, va='top', ha='left')

    #ax.set_title(label)
    ax.set_xlabel('Mean Ws over land [knots]')
    ax.set_ylabel('Maximum surge at landfall [m+MSL]')
    ax.set_xlim(10, 160)
    ax.set_ylim(0, 3.5)
    
    # Plot scatter for each data source
    scatter_em_dat = ax.scatter(
        dataset['avg_WS'], 
        dataset['max_surge'], 
        c=dataset['Damage Level'], 
        cmap=cmap, 
        alpha=0.75,
        s=100,
        vmin=global_min_damage_level,
        vmax=global_max_damage_level,
        marker='h',  # hexagon for EM-DAT
    )
    
    scatter_other = ax.scatter(
        dataset[dataset['Source'] != 'EM-DAT']['avg_WS'], 
        dataset[dataset['Source'] != 'EM-DAT']['avg_TP'], 
        c=dataset[dataset['Source'] != 'EM-DAT']['Damage Level'], 
        cmap=cmap, 
        alpha=0.75,
        s=100,
        vmin=global_min_damage_level,
        vmax=global_max_damage_level,
        marker='*',  # star for other sources
    )
    
    # Create a colorbar for each subplot
    cbar = fig.colorbar(scatter_em_dat, ax=ax, extend='both', ticks=np.arange(1, len(cost_levels)+1))
    cbar.set_label('Damage Levels', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticklabels(['<1M', '10M', '50M', '100M', '500M', '1B', '10B', '50B', '>100B'])

# Adjust layout to prevent overlap
fig.tight_layout()

# Save the plot as a PDF file
plt.savefig('.../Surge_Ws_cost.pdf', format='pdf', bbox_inches='tight')

plt.show()
