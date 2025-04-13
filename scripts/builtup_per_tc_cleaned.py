# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script analyzes the trend of built-up volume per landfalling tropical cyclone 
# from 1980 to 2023. It calculates normalized values, fits an OLS trend line, and 
# visualizes confidence intervals.
#
# Outputs:
# - A time series bar + regression plot: 'builtup_per_tc.pdf'
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
import statsmodels.api as sm
import numpy as np

# Load the updated dataset
updated_df = pd.read_csv('.../unique_landfall_events_builtv_updated.csv')

# Convert Built-up Volume to Billion m³
updated_df['built_v_billion'] = updated_df['built_v'] / 1e9

# Calculate the number of unique TCs per season based on unique combinations of NAME and land_month
unique_tc_count_per_season = updated_df.groupby('SEASON').apply(lambda x: x[['NAME', 'land_month']].drop_duplicates().shape[0]).reset_index(name='tc_count')

# Calculate the total built-up volume per season
total_built_v_per_season = updated_df.groupby('SEASON')['built_v_billion'].sum().reset_index()

# Merge the total built-up volume and unique TC count per season
seasonal_built_v = pd.merge(total_built_v_per_season, unique_tc_count_per_season, on='SEASON')

# Calculate the normalized built-up volume per TC
seasonal_built_v['built_v_per_tc'] = seasonal_built_v['built_v_billion'] / seasonal_built_v['tc_count']

# Prepare data for OLS regression
x = sm.add_constant(seasonal_built_v['SEASON'])
y = seasonal_built_v['built_v_per_tc']

# Fit the OLS model
model = sm.OLS(y, x)
results = model.fit()

# Get predictions and confidence intervals
predict = results.get_prediction(x)
frame = predict.summary_frame(alpha=0.05)  # 95% confidence interval

# Extracting the data for plotting
fitted_values = frame['mean']
conf_lower = frame['mean_ci_lower']
conf_upper = frame['mean_ci_upper']

# Check if the trend is significant
p_value = results.pvalues[1]
significant = p_value < 0.05
trend_label = 'Trend Line'
if significant:
    trend_label += ' **'

# Plot the bar plot
plt.figure(figsize=(8, 5))
plt.bar(seasonal_built_v['SEASON'], seasonal_built_v['built_v_per_tc'], color='skyblue', label='Built-up Volume per TC (Billion m³)')

# Plot the OLS regression line
plt.plot(seasonal_built_v['SEASON'], fitted_values, 'r-', label=trend_label)

# Plot the confidence interval
plt.fill_between(seasonal_built_v['SEASON'], conf_lower, conf_upper, color='red', alpha=0.3, label='95% Confidence Interval')

# Adding labels and title
plt.xlabel('TC Season', fontsize=14)
plt.ylabel('Total Built-up Volume (Billion m³)', fontsize=14)
plt.title('Total Built-up Volume (Billion m³) per TC', fontsize=16)
plt.legend(title='', frameon=False, loc='upper left', fontsize= 12)

plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(1980, 2024)
plt.tight_layout()
plt.savefig('builtup_per_tc.pdf', format='pdf')
plt.show()
