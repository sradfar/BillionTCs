# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script evaluates the contribution of marine heatwaves (MHWs) to the occurrence 
# of tropical cyclone rapid intensification (RI) on a global scale. It calculates and 
# visualizes the yearly fraction and normalized number of MHW-related RI events.
#
# Outputs:
# - A 2-panel figure:
#   (top) Stacked bar chart of RI events with/without MHWs (proportional)
#   (bottom) Normalized number of MHW RI events with trends and decadal comparisons
#
# For more information, see:
# Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2024).
# *Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide.*
# Science Advances.
#
# Disclaimer: Research only. Provided "as is" with no warranties.
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset for events related to rapid intensification
data_mhw_ri = pd.read_csv('.../T26_MHW_RI_input1.csv')
data_all_ri = pd.read_csv('.../intensifications.csv')
data_ibtracs = pd.read_csv('.../ibtracs_filled.csv')
data_ibtracs = data_ibtracs[data_ibtracs['NAME'] != 'NOT_NAMED']

# Convert 'ISO_TIME' to datetime and extract the year
data_mhw_ri['ISO_TIME'] = pd.to_datetime(data_mhw_ri['ISO_TIME'], errors='coerce')
data_mhw_ri['Year'] = data_mhw_ri['ISO_TIME'].dt.year

data_all_ri['ISO_TIME'] = pd.to_datetime(data_all_ri['ISO_TIME'], errors='coerce')
data_all_ri['Year'] = data_all_ri['ISO_TIME'].dt.year

data_ibtracs['ISO_TIME'] = pd.to_datetime(data_ibtracs['ISO_TIME'], errors='coerce')
data_ibtracs['Year'] = data_ibtracs['ISO_TIME'].dt.year

# Filter data to include years 1981 to 2023
years_range = list(range(1981, 2024))
data_mhw_ri = data_mhw_ri[data_mhw_ri['Year'].isin(years_range)]
data_all_ri = data_all_ri[data_all_ri['Year'].isin(years_range)]
data_ibtracs = data_ibtracs[data_ibtracs['Year'].isin(years_range)]

# Filter out rows where RI is not 1 and 'NAME' is 'NOT_NAMED'
filtered_data = data_mhw_ri[(data_mhw_ri['RI'] == 1)]

# Group by Year and count unique RI events based on 'NAME', 'HI_LAT', and 'HI_LON'
mhw_ri_counts = filtered_data.groupby('Year').apply(lambda x: x.drop_duplicates(subset=['NAME', 'HI_LAT', 'HI_LON']).shape[0])

# Sum the counts for each year
total_unique_ri = mhw_ri_counts.sum()

# Group by Year, HI_LAT, and HI_LON and count all RI events
all_ri_counts = data_all_ri.groupby('Year').apply(lambda x: x.drop_duplicates(subset=['NAME', 'HI_LAT', 'HI_LON']).shape[0])

# Drop the first row of all_ri_counts
all_ri_counts = all_ri_counts.iloc[1:]

# Sum the counts for each year
total_all_ri = all_ri_counts.sum()

# Calculate the number of non-MHW-related RI events
non_mhw_ri_counts = all_ri_counts.sub(mhw_ri_counts, fill_value=0)

# Calculate total number of TCs per year based on unique 'Year' and 'NAME'
total_ri_tcs_per_year = data_all_ri.groupby('Year')['NAME'].nunique()
total_tcs_per_year = data_ibtracs.groupby('Year')['NAME'].nunique()
ri_tcs_per_year = data_all_ri.groupby('Year')['NAME'].nunique()
tc_without_ri_per_year = total_tcs_per_year - ri_tcs_per_year
tc_without_ri_per_year = tc_without_ri_per_year.fillna(0)

# Calculate the proportion of MHW-related and non-MHW-related RI events
total_ri_counts = mhw_ri_counts + non_mhw_ri_counts
proportion_mhw_ri_counts = mhw_ri_counts / total_ri_counts
proportion_non_mhw_ri_counts = non_mhw_ri_counts / total_ri_counts
proportion_ri_counts = ri_tcs_per_year / total_tcs_per_year
proportion_non_ri_counts = tc_without_ri_per_year / total_tcs_per_year

# Normalize MHW-related and non-MHW-related RI counts by total TCs per year
normalized_mhw_ri_counts = mhw_ri_counts / total_ri_tcs_per_year
normalized_non_mhw_ri_counts = non_mhw_ri_counts / total_ri_tcs_per_year

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
averages_for_periods = calculate_averages_for_periods(normalized_mhw_ri_counts, periods)

# Calculate percentage increases between periods
percent_increases = []
previous_value = None
for period in sorted(averages_for_periods):
    if previous_value is not None:
        increase = ((averages_for_periods[period] - previous_value) / previous_value) * 100
        percent_increases.append(f"{increase:+.0f}%")
    previous_value = averages_for_periods[period]

# Create a figure for the plots
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot the proportion of MHW-related and non-MHW-related RI events
proportion_ri_counts.plot(kind='bar', ax=axs[0], color='crimson', label='MHW-related')
proportion_non_ri_counts.plot(kind='bar', ax=axs[0], color='lightgray', label='No MHW', bottom=proportion_ri_counts)
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Proportion of Events')
axs[0].set_title('Contribution of MHWs to RI events (normalized)')
axs[0].legend(title='RI Category', loc='upper left')
axs[0].set_ylim(0, 1)  # Set y-axis limits from 0 to 1 for proportions

# Plot the number of MHW-related and non-MHW-related RI events with a logarithmic y-axis
normalized_mhw_ri_counts.plot(kind='bar', ax=axs[1], color='crimson', label='MHW-related')
normalized_non_mhw_ri_counts.plot(kind='bar', ax=axs[1], color='lightgray', label='No MHW', bottom=normalized_mhw_ri_counts)
axs[1].set_xlabel('Year', fontsize=16)
axs[1].set_ylabel('Number of RI per TC', fontsize=16)
axs[1].set_title('Normalized number of RI events related to MHWs', fontsize=16)
axs[1].legend(loc='upper left', fontsize=14)
axs[1].tick_params(axis='both', labelsize=12)

# Add dashed lines and arrows with text
for i, period in enumerate(sorted(averages_for_periods.keys())):
    start_year, end_year = map(int, period.split('-'))
    avg_value = averages_for_periods[period]
    start_index = normalized_mhw_ri_counts.index.get_loc(start_year)
    end_index = normalized_mhw_ri_counts.index.get_loc(end_year) + 0.5
    axs[1].hlines(avg_value, start_index, end_index, colors='black', linestyles='dashed', lw=2)
    
    if i > 0:
        previous_period = list(averages_for_periods)[i-1]
        previous_end_year = int(previous_period.split('-')[1])
        previous_avg = averages_for_periods[previous_period]
        previous_end_index = normalized_mhw_ri_counts.index.get_loc(previous_end_year+1)
        axs[1].annotate('', xy=(start_index, avg_value), xytext=(previous_end_index, previous_avg),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

        # Add text for each arrow
        text_x = previous_end_index + (start_index - previous_end_index) / 2  # Midpoint between the start of this period and the end of the previous one
        text_y = max(avg_value, previous_avg) + 0.5  # Midpoint on the y-axis between the two averages
        axs[1].text(text_x, text_y, percent_increases[i-1], fontweight='bold', horizontalalignment='center', verticalalignment='center', fontsize=11.5, color='black')
       
plt.tight_layout()

#Save the plot as a PDF
plt.savefig('.../mhw_normal_by_category.pdf', format='pdf')
plt.savefig('.../mhw_normal_by_category.png', dpi=300, format='png')

plt.show()
