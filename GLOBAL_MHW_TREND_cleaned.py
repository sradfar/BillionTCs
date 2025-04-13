# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script quantifies the annual number of rapid intensification (RI) tropical 
# cyclones that occurred during marine heatwaves (MHWs) and non-MHW periods from 
# 1981 to 2023. It fits linear trend lines using OLS regression and annotates 
# the plot with trend significance and percentage increases across decades.
#
# Outputs:
# - A bar chart showing the count of RI events with MHW and without MHW
#   from 1981 to 2023.
# - Overlaid OLS regression trend lines with 90% confidence intervals.
# - Annotated arrows and text showing decadal percentage increases.
# - The plot is saved as 'mhw_total_globe.pdf'
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
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Define basins with their latitude and longitude ranges and colors
basins = {
    'Eastern Pacific': {'range': (0, 40, -180, -100), 'color': '#DD6D67'},
    'Atlantic': {'range': (0, 50, -100, -20), 'color': '#67C4CA'},
    'Northwest Pacific': {'range': (0, 50, 100, 180), 'color': '#67CA86'},
    'North Indian': {'range': (0, 30, 45, 100), 'color': '#CA67AB'},
    'Southwest Indian': {'range': (-40, 0, 30, 90), 'color': '#677ACA'},
    'Australian': {'range': (-40, 0, 90, 180), 'color': '#9E67CA'},
    'East Australian': {'range': (-40, 0, -180, -140), 'color': '#CAB767'}
}

min_lat, max_lat, min_lon, max_lon = (0, 50, -100, -20)

# Read the dataset for events related to rapid intensification
data_mhw_ri = pd.read_csv('.../T26_MHW_RI_input1.csv')
data_all_ri = pd.read_csv('.../new_intensifications_with_season.csv')
data_ibtracs = pd.read_csv('.../ibtracs_filled.csv')
data_ibtracs = data_ibtracs[data_ibtracs['NAME'] != 'NOT_NAMED']


data_ibtracs['ISO_TIME'] = pd.to_datetime(data_ibtracs['ISO_TIME'], errors='coerce')
data_ibtracs['Year'] = data_ibtracs['ISO_TIME'].dt.year

# Filter data to include years 1981 to 2023
years_range = list(range(1981, 2024))
data_mhw_ri = data_mhw_ri[data_mhw_ri['SEASON'].isin(years_range)]
data_all_ri = data_all_ri[data_all_ri['SEASON'].isin(years_range)]
data_ibtracs = data_ibtracs[data_ibtracs['SEASON'].isin(years_range)]

# Find unique MHW RI events
mhw_ri_events = data_mhw_ri[['NAME', 'SEASON']].drop_duplicates()

# Find unique RI events
ri_events = data_all_ri[['NAME', 'SEASON']].drop_duplicates()

# Initialize a dictionary to store the counts per year
mhw_ri_counts_per_year_dict = {year: 0 for year in years_range}

# Iterate through each event in ri_events and check if it matches any row in mhw_ri_events
for _, event in ri_events.iterrows():
    if ((mhw_ri_events['NAME'] == event['NAME']) & (mhw_ri_events['SEASON'] == event['SEASON'])).any():
        mhw_ri_counts_per_year_dict[event['SEASON']] += 1

# Convert the dictionary to a pandas Series
mhw_ri_counts_per_year = pd.Series(mhw_ri_counts_per_year_dict).reindex(years_range, fill_value=0)

# Calculate total RI counts per year
ri_counts_per_year = ri_events.groupby('SEASON')['NAME'].nunique().reindex(years_range, fill_value=0)

# Calculate non-MHW RI counts per year
non_mhw_ri_counts_per_year = ri_counts_per_year - mhw_ri_counts_per_year

# Proportion calculations
proportion_ri_counts = ri_counts_per_year
proportion_non_ri_counts = non_mhw_ri_counts_per_year

# Filling missing values with zero where no data exists
proportion_ri_counts.fillna(0, inplace=True)
proportion_non_ri_counts.fillna(0, inplace=True)

# Fit OLS regression model for RI counts
x = sm.add_constant(np.arange(len(proportion_ri_counts)))  # Numeric x values for years, assuming a simple range is adequate
y = proportion_ri_counts.values  # y-values, ensuring no NaNs
model_ri = sm.OLS(y, x)
results_ri = model_ri.fit()

# Get predictions and confidence intervals for RI counts
predict_ri = results_ri.get_prediction()
frame_ri = predict_ri.summary_frame(alpha=0.1)

# Extracting the data for plotting for RI counts
fitted_values_ri = frame_ri['mean']
conf_lower_ri = frame_ri['mean_ci_lower']
conf_upper_ri = frame_ri['mean_ci_upper']

# Fit OLS regression model for combined counts
y_combined = (proportion_ri_counts + proportion_non_ri_counts).values  # Combined y-values
model_combined = sm.OLS(y_combined, x)
results_combined = model_combined.fit()

# Get predictions and confidence intervals for combined counts
predict_combined = results_combined.get_prediction()
frame_combined = predict_combined.summary_frame(alpha=0.1)

# Extracting the data for plotting for combined counts
fitted_values_combined = frame_combined['mean']
conf_lower_combined = frame_combined['mean_ci_lower']
conf_upper_combined = frame_combined['mean_ci_upper']

# Define the periods for which to calculate averages
periods = {
    '1981-1989': (1981, 1989),
    '1990-1999': (1990, 1999),
    '2000-2009': (2000, 2009),
    '2010-2023': (2010, 2023)
}

# Function to calculate the average counts for specified periods from mhw_ri_counts
def calculate_averages_for_periods(counts, periods):
    averages = {}
    for period, (start_year, end_year) in periods.items():
        period_counts = counts[(counts.index >= start_year) & (counts.index <= end_year)]
        average_count = period_counts.mean() if not period_counts.empty else 0
        averages[period] = average_count
    return averages

# Calculate the averages for the specified periods
averages_for_periods = calculate_averages_for_periods(proportion_ri_counts, periods)

# Calculate percentage increases between periods
percent_increases = []
previous_value = None
for period in sorted(averages_for_periods):
    if previous_value is not None:
        increase = ((averages_for_periods[period] - previous_value) / previous_value) * 100
        percent_increases.append(f"{increase:+.0f}%")
    previous_value = averages_for_periods[period]

# Create a figure for the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the proportion of RI and non-RI events
proportion_ri_counts.plot(kind='bar', ax=ax, color='#DC133C', label='MHW RI')
proportion_non_ri_counts.plot(kind='bar', ax=ax, color='lightgray', label='Non-MHW RI', bottom=proportion_ri_counts)

# Plot the trend line for RI counts
ax.plot(np.arange(len(proportion_ri_counts)), fitted_values_ri, 'r-', label='MHW RI Trend')
ax.fill_between(np.arange(len(proportion_ri_counts)), conf_lower_ri, conf_upper_ri, color='red', alpha=0.1)

# Plot the trend line for combined counts
ax.plot(np.arange(len(proportion_ri_counts)), fitted_values_combined, 'k-', label='All RI Trend')
ax.fill_between(np.arange(len(proportion_ri_counts)), conf_lower_combined, conf_upper_combined, color='black', alpha=0.1)

# Check significance and modify legend labels if necessary
if results_ri.pvalues[1] <= 0.1:
    trend_label_ri = 'MHW RI Trend**'
else:
    trend_label_ri = 'MHW RI Trend'

if results_combined.pvalues[1] <= 0.1:
    trend_label_combined = 'All RI Trend**'
else:
    trend_label_combined = 'All RI Trend'

# Add labels and legend
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Number of Events', fontsize=14)
ax.set_ylim(bottom=0) 
ax.set_title('Global', fontsize=14)

# Customize the legend to include only the specified items in the desired order
handles, labels = ax.get_legend_handles_labels()
desired_labels = ['MHW RI', 'Non-MHW RI', trend_label_ri, trend_label_combined]
desired_handles = [handles[labels.index(label)] for label in ['MHW RI', 'Non-MHW RI']]

# Add trend handles separately to avoid index issues
desired_handles.append(handles[labels.index('MHW RI Trend')])
desired_handles.append(handles[labels.index('All RI Trend')])

# Adjust the spacing in the legend
ax.legend(desired_handles, desired_labels, loc='upper left', framealpha=0.5, fontsize=10, handletextpad=0.4, handlelength=2, borderaxespad=0.3, labelspacing=0.3)

ax.tick_params(axis='both', labelsize=12)

# Customize the x-axis to show every second year
xticks = np.arange(len(proportion_ri_counts))
xtick_labels = [str(year) if i % 2 == 0 else '' for i, year in enumerate(proportion_ri_counts.index)]
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, rotation=90, ha='center')

# Add dashed lines and arrows with text
for i, period in enumerate(sorted(averages_for_periods.keys())):
    start_year, end_year = map(int, period.split('-'))
    avg_value = averages_for_periods[period]
    start_index = proportion_ri_counts.index.get_loc(start_year)
    end_index = proportion_ri_counts.index.get_loc(end_year) + 0.5
    ax.hlines(avg_value, start_index, end_index, colors='black', linestyles='dashed', lw=2)
    
    if i > 0:
        previous_period = list(averages_for_periods)[i-1]
        previous_end_year = int(previous_period.split('-')[1])
        previous_avg = averages_for_periods[previous_period]
        previous_end_index = proportion_ri_counts.index.get_loc(previous_end_year+1)
        ax.annotate('', xy=(start_index, avg_value), xytext=(previous_end_index, previous_avg),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

        # Add text for each arrow
        text_x = previous_end_index + (start_index - previous_end_index) / 2  # Midpoint between the start of this period and the end of the previous one
        text_y = max(avg_value, previous_avg) + 1.35  # Midpoint on the y-axis between the two averages
        ax.text(text_x, text_y, percent_increases[i-1], fontweight='bold', horizontalalignment='center', verticalalignment='center', fontsize=11.5, color='black')

plt.tight_layout()
plt.savefig('.../mhw_total_globe.pdf', format='pdf')
plt.show()

# Print OLS regression results for significance testing
print("RI Trend Line Significance:")
print(results_ri.summary())
print("\nAll TCs Trend Line Significance:")
print(results_combined.summary())
