# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script performs multiple linear regression analyses for grouped tropical cyclones 
# based on cost quantiles. It evaluates the effects of built-up volume and TC intensity 
# on economic losses, both with and without interaction terms.
#
# Outputs:
# - Regression summaries by group and cost range, saved as text
# - Output file: 'regression_results_by_cost_range_interaction.txt'
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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

# Load the main dataset
main_df = pd.read_csv('.../TC cost/updated_aggregated_with_built_v_agg.csv')

# Load the additional datasets
with_RI_df = pd.read_csv('.../with_RI_land_info_updated.csv')
non_RI_df = pd.read_csv('.../non_RI_land_info_updated.csv')
ibtracs_df = pd.read_csv('.../ibtracs_filled.csv')

# Apply filtering to the datasets as per the criteria
RI_without_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 0) & (with_RI_df['mhw_hours'] == 0)]
RI_with_MHW_df = with_RI_df[(with_RI_df['land'] == 1) & (with_RI_df['MHW_land'] == 1) & (with_RI_df['mhw_hours'] > 48)]
non_RI_df = non_RI_df[(non_RI_df['land'] == 1) | (non_RI_df['land'] == 10)]
without_RI_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 1) & (non_RI_df['mhw_hours'] > 48)]
without_RI_no_MHW_df = non_RI_df[(non_RI_df['MHW_land'] == 0) & (non_RI_df['mhw_hours'] == 0)]

# Convert 'land_time' to datetime
RI_without_MHW_df['land_time'] = pd.to_datetime(RI_without_MHW_df['land_time'])
RI_with_MHW_df['land_time'] = pd.to_datetime(RI_with_MHW_df['land_time'])
without_RI_MHW_df['land_time'] = pd.to_datetime(without_RI_MHW_df['land_time'])
without_RI_no_MHW_df['land_time'] = pd.to_datetime(without_RI_no_MHW_df['land_time'])

# Add a new column to main_df for group assignment
main_df['group'] = 0

# Assign groups based on the criteria
def assign_group(row):
    if ((RI_with_MHW_df['NAME'] == row['NAME']) & (RI_with_MHW_df['SEASON'] == row['SEASON'])).any():
        return 1
    elif ((RI_without_MHW_df['NAME'] == row['NAME']) & (RI_without_MHW_df['SEASON'] == row['SEASON'])).any():
        return 2
    elif ((without_RI_MHW_df['NAME'] == row['NAME']) & (without_RI_MHW_df['SEASON'] == row['SEASON'])).any():
        return 3
    elif ((without_RI_no_MHW_df['NAME'] == row['NAME']) & (without_RI_no_MHW_df['SEASON'] == row['SEASON'])).any():
        return 4
    else:
        return 0

main_df['group'] = main_df.apply(assign_group, axis=1)

# Function to remove infinities and impute NaNs
def clean_data(df, columns):
    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='mean')
    df[columns] = imputer.fit_transform(df[columns])
    return df

# Standardize TC intensity measures and handle NaNs
tc_columns = ['max_surge', 'avg_WS', 'avg_TP']
main_df = clean_data(main_df, tc_columns)
scaler = StandardScaler()
main_df[tc_columns] = scaler.fit_transform(main_df[tc_columns])

# Define cost ranges (in 1000s of USD)
cost_ranges = {
    # '<10M': (0, 10000),
    # '10-100M': (10000, 100000),
    # '100M-1B': (100000, 1000000),
    '1M-1B': (1000, 1000000),
    '>1B': (1000000, np.inf)
}

# # Function to perform analysis for each cost range
# def analyze_cost_range(df, range_label, cost_range):
    # results = []
    # # Filter data for the cost range
    # cost_range_df = df[(df['Total Damage, Adjusted (\'000 US$)'] >= cost_range[0]) & 
                       # (df['Total Damage, Adjusted (\'000 US$)'] < cost_range[1])]
    
    # if cost_range_df.empty:
        # return f"No data for cost range {range_label}\n"

    # # Analyze each group within the cost range
    # for group in range(1, 5):
        # group_df = cost_range_df[cost_range_df['group'] == group]
        
        # if group_df.empty:
            # results.append(f"No data for group {group} in cost range {range_label}\n")
            # continue

        # # Ensure no NaNs or infinities in built_v_agg and Total Damage
        # group_df = clean_data(group_df, ['built_v_agg', 'Total Damage, Adjusted (\'000 US$)'])

        # # PCA to combine TC intensity measures
        # pca = PCA(n_components=1)
        # group_df['TC_intensity'] = pca.fit_transform(group_df[tc_columns])

        # # Multiple Linear Regression
        # X = group_df[['built_v_agg', 'TC_intensity']]
        # y = group_df['Total Damage, Adjusted (\'000 US$)']

        # # Add constant term for intercept
        # X = sm.add_constant(X)

        # # Fit the model
        # model = sm.OLS(y, X).fit()

        # # Summary of the model
        # results.append(f"Summary for group {group} in cost range {range_label}:\n{model.summary()}\n")
    
    # return ''.join(results)

# Function to perform analysis for each cost range
def analyze_cost_range(df, range_label, cost_range):
    results = []
    # Filter data for the cost range
    cost_range_df = df[(df['Total Damage, Adjusted (\'000 US$)'] >= cost_range[0]) & 
                       (df['Total Damage, Adjusted (\'000 US$)'] < cost_range[1])]
    
    if cost_range_df.empty:
        return f"No data for cost range {range_label}\n"

    # Analyze each group within the cost range
    for group in range(1, 5):
        group_df = cost_range_df[cost_range_df['group'] == group]
        
        if group_df.empty:
            results.append(f"No data for group {group} in cost range {range_label}\n")
            continue

        # Ensure no NaNs or infinities in built_v_agg and Total Damage
        group_df = clean_data(group_df, ['built_v_agg', 'Total Damage, Adjusted (\'000 US$)'])

        # PCA to combine TC intensity measures
        pca = PCA(n_components=1)
        group_df['TC_intensity'] = pca.fit_transform(group_df[tc_columns])

        # Multiple Linear Regression
        
        # 1. WITHOUT interaction term (your original approach)
        X = group_df[['built_v_agg', 'TC_intensity']]
        y = group_df['Total Damage, Adjusted (\'000 US$)']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        results.append(f"Summary for group {group} in cost range {range_label} (WITHOUT interaction):\n{model.summary()}\n")
        
        # 2. WITH interaction term (the new approach)
        # Create interaction term
        group_df['built_TC_interaction'] = group_df['built_v_agg'] * group_df['TC_intensity']
        
        # Define model with interaction
        X_interaction = group_df[['built_v_agg', 'TC_intensity', 'built_TC_interaction']]
        X_interaction = sm.add_constant(X_interaction)
        
        # Fit the model with interaction
        model_interaction = sm.OLS(y, X_interaction).fit()
        
        # Add interaction model results
        results.append(f"Summary for group {group} in cost range {range_label} (WITH interaction):\n{model_interaction.summary()}\n")
    
    return ''.join(results)
    
# Analyze each cost range and save the results
results = ""
for label, range_ in cost_ranges.items():
    results += analyze_cost_range(main_df, label, range_)

# Save results to a text file
with open('.../regression_results_by_cost_range_interaction.txt', 'w') as file:
    file.write(results)
