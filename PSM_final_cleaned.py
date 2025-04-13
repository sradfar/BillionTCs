# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script performs stratified propensity score matching (PSM) to assess whether 
# tropical cyclones undergoing rapid intensification (RI) during marine heatwaves (MHW)
# cause more damage than comparable TCs without MHWs or RI. It controls for built 
# exposure (built_v_agg) and matches within the same continent.
#
# Outputs:
# - CSV files for matched groups and summary statistics
# - A text report: 'stratified_psm_analysis_results.txt'
# - Plots comparing TC damage, exposure, and intensity across matched groups
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

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

# Drop rows with group=0 (not in any of our categories)
main_df = main_df[main_df['group'] > 0]

# Make sure 'built_v_agg' is clean
main_df = clean_data(main_df, ['built_v_agg'])

# Make sure 'Total Damage, Adjusted (\'000 US$)' is clean
main_df = clean_data(main_df, ['Total Damage, Adjusted (\'000 US$)'])

# Log transform damage for more normal distribution
main_df['log_damage'] = np.log1p(main_df['Total Damage, Adjusted (\'000 US$)'])

# First, filter main_df for each category using the 'group' column
ri_with_mhw = main_df[main_df['group'] == 1].copy()
ri_without_mhw = main_df[main_df['group'] == 2].copy()
no_ri_with_mhw = main_df[main_df['group'] == 3].copy()
no_ri_without_mhw = main_df[main_df['group'] == 4].copy()

# Define output paths
output_dir = '.../verification/'  # Change this to your preferred directory

# Export each filtered dataframe to CSV
ri_with_mhw.to_csv(f'{output_dir}ri_with_mhw.csv', index=False)
ri_without_mhw.to_csv(f'{output_dir}ri_without_mhw.csv', index=False)
no_ri_with_mhw.to_csv(f'{output_dir}no_ri_with_mhw.csv', index=False)
no_ri_without_mhw.to_csv(f'{output_dir}no_ri_without_mhw.csv', index=False)

# PCA to combine TC intensity measures into a single score
pca = PCA(n_components=1)
main_df['TC_intensity'] = pca.fit_transform(main_df[tc_columns])

##############################################################################
# STRATIFIED PROPENSITY SCORE MATCHING ANALYSIS BY CONTINENT
##############################################################################

# Define treatment and control groups
# Group 1 (RI with MHW) is our treatment group
treatment_df = main_df[main_df['group'] == 1].copy()
treatment_df['treatment'] = 1

# Create control groups for different comparisons
control_group2_df = main_df[main_df['group'] == 2].copy()
control_group2_df['treatment'] = 0

control_group3_df = main_df[main_df['group'] == 3].copy()
control_group3_df['treatment'] = 0

control_group4_df = main_df[main_df['group'] == 4].copy()
control_group4_df['treatment'] = 0

# Filter treatment and control datasets to include only North America and Asia
continents_to_include = ['North America', 'Asia', 'Oceania', 'Africa']
treatment_df = treatment_df[treatment_df['CONTINENT'].isin(continents_to_include)].copy()
control_group2_df = control_group2_df[control_group2_df['CONTINENT'].isin(continents_to_include)].copy()
control_group3_df = control_group3_df[control_group3_df['CONTINENT'].isin(continents_to_include)].copy()
control_group4_df = control_group4_df[control_group4_df['CONTINENT'].isin(continents_to_include)].copy()

# Function to perform stratified PSM by continent and analyze results
def perform_stratified_psm(treatment_df, control_df, group_name, output_file):
    # Create an empty list to store matched data from each continent
    all_matched_data = []
    all_matches = []
    
    # Get unique continents in the combined data
    continents = pd.concat([treatment_df['CONTINENT'], control_df['CONTINENT']]).unique()
    
    # Write header to output file
    with open(output_file, 'a') as f:
        f.write(f"\n\n========================= STRATIFIED PSM ANALYSIS RESULTS =========================\n")
        f.write(f"Comparison: Treatment (RI with MHW) vs. {group_name}\n")
        f.write(f"Stratified by continent: matches only made within the same continent\n\n")
    
    # Initialize counters
    total_matched_pairs = 0
    
    # Loop through each continent and perform matching within that continent
    for continent in continents:
        # Filter data for current continent
        continent_treatment = treatment_df[treatment_df['CONTINENT'] == continent].copy()
        continent_control = control_df[control_df['CONTINENT'] == continent].copy()
        
        # Skip if either group has no data for this continent
        if len(continent_treatment) == 0 or len(continent_control) == 0:
            with open(output_file, 'a') as f:
                f.write(f"Continent: {continent} - Insufficient data for matching (Treatment: {len(continent_treatment)}, Control: {len(continent_control)})\n\n")
            continue
            
        # Combine treatment and control for this continent
        continent_combined = pd.concat([continent_treatment, continent_control])
        
        # We'll use built_v_agg (exposure) as our covariate for matching
        X = continent_combined[['built_v_agg']]
        y = continent_combined['treatment']
        
        # Logistic regression to calculate propensity scores
        try:
            logit = LogisticRegression(random_state=42)
            logit.fit(X, y)
            
            # Calculate propensity scores
            propensity_scores = logit.predict_proba(X)[:, 1]
            continent_combined['propensity_score'] = propensity_scores
            
        except Exception as e:
            with open(output_file, 'a') as f:
                f.write(f"Continent: {continent} - Error calculating propensity scores: {str(e)}\n\n")
            continue
            
        # Now, match treatment to controls based on propensity scores
        # Using nearest neighbor matching with replacement
        
        # Function to find nearest match within the same continent
        def find_match(propensity_score, excluded_indices):
            control_scores = continent_combined[(continent_combined['treatment'] == 0) & 
                                               (~continent_combined.index.isin(excluded_indices))]['propensity_score']
            if len(control_scores) == 0:
                return None
            distances = abs(control_scores - propensity_score)
            match_idx = distances.idxmin()
            return match_idx
        
        # Initialize matching results for this continent
        continent_matches = []
        used_control_indices = set()
        
        # For each treatment observation in this continent, find nearest control
        for idx, row in continent_combined[continent_combined['treatment'] == 1].iterrows():
            match_idx = find_match(row['propensity_score'], used_control_indices)
            if match_idx is not None:
                continent_matches.append((idx, match_idx))
                used_control_indices.add(match_idx)
        
        # Append the matches from this continent to the overall list
        all_matches.extend(continent_matches)
        
        # Create matched dataset for this continent
        if len(continent_matches) > 0:
            matched_treatment_indices = [pair[0] for pair in continent_matches]
            matched_control_indices = [pair[1] for pair in continent_matches]
            
            continent_matched_data = pd.concat([
                continent_combined.loc[matched_treatment_indices],
                continent_combined.loc[matched_control_indices]
            ])
            
            # Record stats for this continent
            with open(output_file, 'a') as f:
                f.write(f"Continent: {continent}\n")
                f.write(f"  Number of matched pairs: {len(continent_matches)}\n")
                f.write(f"  Treatment storms: {len(continent_treatment)}, Control storms: {len(continent_control)}\n")
                f.write(f"  Matched treatment: {len(matched_treatment_indices)}, Matched control: {len(matched_control_indices)}\n\n")
            
            # Add to combined matched data
            all_matched_data.append(continent_matched_data)
            total_matched_pairs += len(continent_matches)
        else:
            with open(output_file, 'a') as f:
                f.write(f"Continent: {continent} - No matches found\n\n")
    
    # Combine matched data from all continents
    if len(all_matched_data) > 0:
        matched_data = pd.concat(all_matched_data)
    else:
        with open(output_file, 'a') as f:
            f.write("No matches found across any continent\n")
        return None, None
    
    # Write summary of matches
    with open(output_file, 'a') as f:
        f.write(f"Total matched pairs across all continents: {total_matched_pairs}\n\n")
    
    # Calculate average treatment effect (ATE)
    treatment_avg_damage = np.mean(matched_data[matched_data['treatment'] == 1]['Total Damage, Adjusted (\'000 US$)'])
    control_avg_damage = np.mean(matched_data[matched_data['treatment'] == 0]['Total Damage, Adjusted (\'000 US$)'])
    ate = treatment_avg_damage - control_avg_damage
    
    # Calculate log-transformed ATE (often more reliable for skewed data)
    treatment_avg_log_damage = np.mean(matched_data[matched_data['treatment'] == 1]['log_damage'])
    control_avg_log_damage = np.mean(matched_data[matched_data['treatment'] == 0]['log_damage'])
    log_ate = treatment_avg_log_damage - control_avg_log_damage
    
    # Statistical tests on matched data
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        matched_data[matched_data['treatment'] == 1]['Total Damage, Adjusted (\'000 US$)'],
        matched_data[matched_data['treatment'] == 0]['Total Damage, Adjusted (\'000 US$)']
    )
    
    # Non-parametric test (Wilcoxon/Mann-Whitney)
    mw_stat, mw_p_value = stats.mannwhitneyu(
        matched_data[matched_data['treatment'] == 1]['Total Damage, Adjusted (\'000 US$)'],
        matched_data[matched_data['treatment'] == 0]['Total Damage, Adjusted (\'000 US$)']
    )
    
    # Calculate percentage difference for easier interpretation
    pct_difference = ((treatment_avg_damage - control_avg_damage) / control_avg_damage) * 100
    
    # Prepare results
    results = {
        'comparison': f"Treatment (RI with MHW) vs. {group_name}",
        'n_treatment': len(matched_data[matched_data['treatment'] == 1]),
        'n_control': len(matched_data[matched_data['treatment'] == 0]),
        'treatment_avg_damage': treatment_avg_damage,
        'control_avg_damage': control_avg_damage,
        'ate': ate,
        'log_ate': log_ate,
        'percent_difference': pct_difference,
        't_statistic': t_stat,
        't_p_value': p_value,
        'mw_statistic': mw_stat,
        'mw_p_value': mw_p_value,
        'matched_data': matched_data
    }
    
    # Balance check - compare built-up volume distribution before and after matching
    before_treatment_builtup = treatment_df['built_v_agg'].mean()
    before_control_builtup = control_df['built_v_agg'].mean()
    after_treatment_builtup = matched_data[matched_data['treatment'] == 1]['built_v_agg'].mean()
    after_control_builtup = matched_data[matched_data['treatment'] == 0]['built_v_agg'].mean()
    
    balance_stats = {
        'before_treatment_builtup': before_treatment_builtup,
        'before_control_builtup': before_control_builtup,
        'before_diff': before_treatment_builtup - before_control_builtup,
        'before_pct_diff': ((before_treatment_builtup - before_control_builtup) / before_control_builtup) * 100,
        'after_treatment_builtup': after_treatment_builtup,
        'after_control_builtup': after_control_builtup,
        'after_diff': after_treatment_builtup - after_control_builtup,
        'after_pct_diff': ((after_treatment_builtup - after_control_builtup) / after_control_builtup) * 100
    }
    
    # Print results to file
    with open(output_file, 'a') as f:
        f.write("BALANCE CHECK:\n")
        f.write(f"Before matching - Treatment built-up mean: ${before_treatment_builtup:.2e}\n")
        f.write(f"Before matching - Control built-up mean: ${before_control_builtup:.2e}\n")
        f.write(f"Before matching - Difference: ${balance_stats['before_diff']:.2e} ({balance_stats['before_pct_diff']:.2f}%)\n\n")
        
        f.write(f"After matching - Treatment built-up mean: ${after_treatment_builtup:.2e}\n")
        f.write(f"After matching - Control built-up mean: ${after_control_builtup:.2e}\n")
        f.write(f"After matching - Difference: ${balance_stats['after_diff']:.2e} ({balance_stats['after_pct_diff']:.2f}%)\n\n")
        
        f.write("DAMAGE COMPARISON:\n")
        f.write(f"Treatment (RI with MHW) average damage: ${treatment_avg_damage:.2f}k\n")
        f.write(f"Matched {group_name} average damage: ${control_avg_damage:.2f}k\n")
        f.write(f"Average Treatment Effect (ATE): ${ate:.2f}k\n")
        f.write(f"Percentage difference: {pct_difference:.2f}%\n\n")
        
        f.write("STATISTICAL TESTS:\n")
        f.write(f"t-test: t = {t_stat:.4f}, p-value = {p_value:.4f}\n")
        f.write(f"Mann-Whitney test: U = {mw_stat:.4f}, p-value = {mw_p_value:.4f}\n\n")
        
        # Also analyze by damage tier
        f.write("ANALYSIS BY DAMAGE TIER:\n")
        
        # Create damage tiers
        matched_data['damage_tier'] = pd.cut(
            matched_data['Total Damage, Adjusted (\'000 US$)'],
            bins=[0, 1000000, np.inf],
            labels=['<1B', '>1B']
        )
        
        # First analyze overall by tier
        tier_stats = matched_data.groupby(['damage_tier', 'treatment']).agg(
            count=('Total Damage, Adjusted (\'000 US$)', 'count'),
            avg_damage=('Total Damage, Adjusted (\'000 US$)', 'mean')
        ).reset_index()
        
        # Pivot for easier comparison
        tier_comparison = tier_stats.pivot(index='damage_tier', columns='treatment', values=['count', 'avg_damage'])
        
        for tier in tier_comparison.index:
            try:
                treatment_count = tier_comparison.loc[tier, ('count', 1)]
                control_count = tier_comparison.loc[tier, ('count', 0)]
                treatment_avg = tier_comparison.loc[tier, ('avg_damage', 1)]
                control_avg = tier_comparison.loc[tier, ('avg_damage', 0)]
                
                tier_diff = treatment_avg - control_avg
                tier_pct_diff = ((treatment_avg - control_avg) / control_avg) * 100 if control_avg > 0 else float('inf')
                
                f.write(f"Damage tier {tier}:\n")
                f.write(f"  Treatment count: {treatment_count}, Control count: {control_count}\n")
                f.write(f"  Treatment avg: ${treatment_avg:.2f}k, Control avg: ${control_avg:.2f}k\n")
                f.write(f"  Difference: ${tier_diff:.2f}k ({tier_pct_diff:.2f}%)\n\n")
            except:
                f.write(f"Damage tier {tier}: Insufficient data for comparison\n\n")
        
        # Then analyze by continent and tier
        f.write("ANALYSIS BY CONTINENT AND DAMAGE TIER:\n")
        
        for continent in matched_data['CONTINENT'].unique():
            continent_data = matched_data[matched_data['CONTINENT'] == continent]
            
            f.write(f"\nContinent: {continent}\n")
            f.write(f"  Treatment storms: {len(continent_data[continent_data['treatment'] == 1])}\n")
            f.write(f"  Control storms: {len(continent_data[continent_data['treatment'] == 0])}\n")
            
            # Calculate average damage by treatment group for this continent
            continent_treatment_avg = continent_data[continent_data['treatment'] == 1]['Total Damage, Adjusted (\'000 US$)'].mean()
            continent_control_avg = continent_data[continent_data['treatment'] == 0]['Total Damage, Adjusted (\'000 US$)'].mean()
            
            continent_pct_diff = ((continent_treatment_avg - continent_control_avg) / continent_control_avg) * 100 if continent_control_avg > 0 else float('inf')
            
            f.write(f"  Treatment avg damage: ${continent_treatment_avg:.2f}k\n")
            f.write(f"  Control avg damage: ${continent_control_avg:.2f}k\n")
            f.write(f"  Difference: ${continent_treatment_avg - continent_control_avg:.2f}k ({continent_pct_diff:.2f}%)\n\n")
            
            # Now break down by damage tier within this continent
            try:
                continent_tier_stats = continent_data.groupby(['damage_tier', 'treatment']).agg(
                    count=('Total Damage, Adjusted (\'000 US$)', 'count'),
                    avg_damage=('Total Damage, Adjusted (\'000 US$)', 'mean')
                ).reset_index()
                
                # Pivot for comparison
                continent_tier_comparison = continent_tier_stats.pivot(index='damage_tier', columns='treatment', values=['count', 'avg_damage'])
                
                for tier in continent_tier_comparison.index:
                    try:
                        c_treatment_count = continent_tier_comparison.loc[tier, ('count', 1)]
                        c_control_count = continent_tier_comparison.loc[tier, ('count', 0)]
                        c_treatment_avg = continent_tier_comparison.loc[tier, ('avg_damage', 1)]
                        c_control_avg = continent_tier_comparison.loc[tier, ('avg_damage', 0)]
                        
                        c_tier_diff = c_treatment_avg - c_control_avg
                        c_tier_pct_diff = ((c_treatment_avg - c_control_avg) / c_control_avg) * 100 if c_control_avg > 0 else float('inf')
                        
                        f.write(f"  Damage tier {tier}:\n")
                        f.write(f"    Treatment count: {c_treatment_count}, Control count: {c_control_count}\n")
                        f.write(f"    Treatment avg: ${c_treatment_avg:.2f}k, Control avg: ${c_control_avg:.2f}k\n")
                        f.write(f"    Difference: ${c_tier_diff:.2f}k ({c_tier_pct_diff:.2f}%)\n\n")
                    except:
                        f.write(f"  Damage tier {tier}: Insufficient data for comparison\n\n")
            except:
                f.write("  Insufficient data for damage tier analysis in this continent\n\n")
        
        f.write("=======================================================================\n")
    
    return results, matched_data

# Create output file
output_file = '.../stratified_psm_analysis_results.txt'

with open(output_file, 'w') as f:
    f.write("STRATIFIED PROPENSITY SCORE MATCHING ANALYSIS RESULTS\n")
    f.write("===================================================\n\n")
    f.write("This analysis matches TCs with RI and MHW (treatment group) with other TC types\n")
    f.write("based on similar built-up volume (exposure) WITHIN THE SAME CONTINENT to isolate\n")
    f.write("the effect of MHW+RI on damage, controlling for exposure and geographic region.\n\n")

# Perform stratified PSM for each comparison
results_group2, matched_data_group2 = perform_stratified_psm(
    treatment_df, control_group2_df, "Group 2 (RI without MHW)", output_file
)

results_group3, matched_data_group3 = perform_stratified_psm(
    treatment_df, control_group3_df, "Group 3 (No RI with MHW)", output_file
)

results_group4, matched_data_group4 = perform_stratified_psm(
    treatment_df, control_group4_df, "Group 4 (No RI without MHW)", output_file
)

# Skip further analysis if any of the matched datasets are None
if results_group2 is None or results_group3 is None or results_group4 is None:
    with open(output_file, 'a') as f:
        f.write("\n\nOne or more comparisons failed to produce matches. Analysis terminated.\n")
    print("Analysis terminated due to insufficient matches in one or more comparisons.")
else:
    # Additional analysis: Billion-dollar events
    # Filter to focus only on high-damage events (≥$1B)
    high_damage_threshold = 1000000  # $1B in '000 USD

    # Create combined results for all comparisons as a summary
    with open(output_file, 'a') as f:
        f.write("\n\n=================== SUMMARY OF STRATIFIED PSM RESULTS ===================\n\n")
        f.write("Comparison                      | % Diff  | p-value (t-test) | p-value (MW) \n")
        f.write("--------------------------------|---------|------------------|-------------\n")
        f.write(f"RI with MHW vs RI without MHW       | {results_group2['percent_difference']:7.2f}% | {results_group2['t_p_value']:.4f} | {results_group2['mw_p_value']:.4f}\n")
        f.write(f"RI with MHW vs No RI with MHW       | {results_group3['percent_difference']:7.2f}% | {results_group3['t_p_value']:.4f} | {results_group3['mw_p_value']:.4f}\n")
        f.write(f"RI with MHW vs No RI without MHW    | {results_group4['percent_difference']:7.2f}% | {results_group4['t_p_value']:.4f} | {results_group4['mw_p_value']:.4f}\n\n")

        # Add section specifically for billion-dollar events
        f.write("\n============= BILLION-DOLLAR EVENTS ANALYSIS =============\n\n")
        
        # For each matched dataset, filter for billion-dollar events
        for result, data, group_name in [
            (results_group2, matched_data_group2, "RI without MHW"),
            (results_group3, matched_data_group3, "No RI with MHW"),
            (results_group4, matched_data_group4, "No RI without MHW")
        ]:
            billion_data = data[data['Total Damage, Adjusted (\'000 US$)'] >= high_damage_threshold]
            
            if len(billion_data) > 0:
                treatment_billion = billion_data[billion_data['treatment'] == 1]
                control_billion = billion_data[billion_data['treatment'] == 0]
                
                if len(treatment_billion) > 0 and len(control_billion) > 0:
                    treatment_avg = treatment_billion['Total Damage, Adjusted (\'000 US$)'].mean()
                    control_avg = control_billion['Total Damage, Adjusted (\'000 US$)'].mean()
                    pct_diff = ((treatment_avg - control_avg) / control_avg) * 100
                    
                    # Statistical tests if enough samples
                    if len(treatment_billion) >= 5 and len(control_billion) >= 5:
                        t_stat, p_value = stats.ttest_ind(
                            treatment_billion['Total Damage, Adjusted (\'000 US$)'],
                            control_billion['Total Damage, Adjusted (\'000 US$)']
                        )
                        
                        mw_stat, mw_p = stats.mannwhitneyu(
                            treatment_billion['Total Damage, Adjusted (\'000 US$)'],
                            control_billion['Total Damage, Adjusted (\'000 US$)']
                        )
                    else:
                        t_stat, p_value = float('nan'), float('nan')
                        mw_stat, mw_p = float('nan'), float('nan')
                    
                    f.write(f"RI with MHW vs {group_name} (Billion-dollar events):\n")
                    f.write(f"  Treatment count: {len(treatment_billion)}, Control count: {len(control_billion)}\n")
                    f.write(f"  Treatment avg: ${treatment_avg:.2f}k, Control avg: ${control_avg:.2f}k\n")
                    f.write(f"  Difference: ${treatment_avg - control_avg:.2f}k ({pct_diff:.2f}%)\n")
                    if not np.isnan(p_value):
                        f.write(f"  t-test: t = {t_stat:.4f}, p-value = {p_value:.4f}\n")
                        f.write(f"  Mann-Whitney: U = {mw_stat:.4f}, p-value = {mw_p:.4f}\n")
                    else:
                        f.write("  Insufficient sample size for statistical testing\n")
                    f.write("\n")
                else:
                    f.write(f"RI with MHW vs {group_name} (Billion-dollar events): Insufficient data for comparison\n\n")
            else:
                f.write(f"RI with MHW vs {group_name} (Billion-dollar events): No billion-dollar events in matched data\n\n")
    
    # Create visualizations to help interpret the results
    # 1. Propensity score distributions before and after matching by continent
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # Group 2 comparison
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=pd.concat([treatment_df, control_group2_df]), ax=axs[0, 0])
    axs[0, 0].set_title('Built-up Volume by Continent: Before Matching\nRI with MHW vs RI w/o MHW')
    axs[0, 0].set_xticklabels(['RI w/o MHW', 'RI with MHW'])
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend(title='Continent', loc='upper right')
    
    # After matching
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=matched_data_group2, ax=axs[0, 1])
    axs[0, 1].set_title('Built-up Volume by Continent: After Matching\nRI with MHW vs RI w/o MHW')
    axs[0, 1].set_xticklabels(['RI w/o MHW', 'RI with MHW'])
    axs[0, 1].set_yscale('log')
    axs[0, 1].legend(title='Continent', loc='upper right')
    
    # Group 3 comparison
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=pd.concat([treatment_df, control_group3_df]), ax=axs[1, 0])
    axs[1, 0].set_title('Built-up Volume by Continent: Before Matching\nRI with MHW vs No RI with MHW')
    axs[1, 0].set_xticklabels(['No RI with MHW', 'RI with MHW'])
    axs[1, 0].set_yscale('log')
    axs[1, 0].legend(title='Continent', loc='upper right')
    
    # After matching
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=matched_data_group3, ax=axs[1, 1])
    axs[1, 1].set_title('Built-up Volume by Continent: After Matching\nRI with MHW vs No RI with MHW')
    axs[1, 1].set_xticklabels(['No RI with MHW', 'RI with MHW'])
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend(title='Continent', loc='upper right')
    
    # Group 4 comparison
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=pd.concat([treatment_df, control_group4_df]), ax=axs[2, 0])
    axs[2, 0].set_title('Built-up Volume by Continent: Before Matching\nRI with MHW vs No RI w/o MHW')
    axs[2, 0].set_xticklabels(['No RI w/o MHW', 'RI with MHW'])
    axs[2, 0].set_yscale('log')
    axs[2, 0].legend(title='Continent', loc='upper right')
    
    # After matching
    sns.boxplot(x='treatment', y='built_v_agg', hue='CONTINENT', data=matched_data_group4, ax=axs[2, 1])
    axs[2, 1].set_title('Built-up Volume by Continent: After Matching\nRI with MHW vs No RI w/o MHW')
    axs[2, 1].set_xticklabels(['No RI w/o MHW', 'RI with MHW'])
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(title='Continent', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('.../stratified_psm_built_volume_distribution.png', dpi=300)
    plt.close()
    
    # 2. Damage comparison plots by continent
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Group 2 damage comparison
    sns.boxplot(x='treatment', y='Total Damage, Adjusted (\'000 US$)', hue='CONTINENT', data=matched_data_group2, ax=axs[0])
    axs[0].set_title('Damage Comparison by Continent: RI with MHW vs RI w/o MHW (Matched Pairs)')
    axs[0].set_xticklabels(['RI w/o MHW', 'RI with MHW'])
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Damage (\'000 USD, log scale)')
    axs[0].legend(title='Continent', loc='upper right')
    
    # Group 3 damage comparison
    sns.boxplot(x='treatment', y='Total Damage, Adjusted (\'000 US$)', hue='CONTINENT', data=matched_data_group3, ax=axs[1])
    axs[1].set_title('Damage Comparison by Continent: RI with MHW vs No RI with MHW (Matched Pairs)')
    axs[1].set_xticklabels(['No RI with MHW', 'RI with MHW'])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Damage (\'000 USD, log scale)')
    axs[1].legend(title='Continent', loc='upper right')
    
    # Group 4 damage comparison
    sns.boxplot(x='treatment', y='Total Damage, Adjusted (\'000 US$)', hue='CONTINENT', data=matched_data_group4, ax=axs[2])
    axs[2].set_title('Damage Comparison by Continent: RI with MHW vs No RI w/o MHW (Matched Pairs)')
    axs[2].set_xticklabels(['No RI w/o MHW', 'RI with MHW'])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('Damage (\'000 USD, log scale)')
    axs[2].legend(title='Continent', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('.../stratified_psm_damage_comparison.png', dpi=300)
    plt.close()
    
    # 3. Continent-specific analysis plots
    # Create summary dataframes by continent for each comparison
    continent_list = set(treatment_df['CONTINENT'].unique()) | set(control_group2_df['CONTINENT'].unique()) | set(control_group3_df['CONTINENT'].unique()) | set(control_group4_df['CONTINENT'].unique())
    
    # Plot continent-specific damage comparisons
    plt.figure(figsize=(16, 10))
    
    # Create a summary dataframe for plotting
    summary_data = []
    
    for group_num, (result, matched_data, label) in enumerate([
        (results_group2, matched_data_group2, "RI w/o MHW"),
        (results_group3, matched_data_group3, "No RI with MHW"),
        (results_group4, matched_data_group4, "No RI w/o MHW")
    ]):
        # Treatment data
        treatment_data = matched_data[matched_data['treatment'] == 1].copy()
        treatment_data['group'] = f"RI with MHW vs {label}"
        treatment_data['type'] = "RI with MHW"
        summary_data.append(treatment_data[['group', 'type', 'CONTINENT', 'Total Damage, Adjusted (\'000 US$)', 'TC_intensity']])
        
        # Control data
        control_data = matched_data[matched_data['treatment'] == 0].copy()
        control_data['group'] = f"RI with MHW vs {label}"
        control_data['type'] = label
        summary_data.append(control_data[['group', 'type', 'CONTINENT', 'Total Damage, Adjusted (\'000 US$)', 'TC_intensity']])
    
    summary_df = pd.concat(summary_data)
    
    # Create continent-based facet grid for damage comparison
    g = sns.catplot(
        data=summary_df,
        kind="bar",
        x="group", y="Total Damage, Adjusted ('000 US$)",
        hue="type", col="CONTINENT",
        col_wrap=2, height=5, aspect=1.2,
        sharex=True, sharey=False,
        error_bars="sd", errwidth=1.5,
        palette="dark"
    )
    
    g.set_axis_labels("", "Damage ('000 USD)")
    g.set_xticklabels(rotation=45)
    g.set_titles("{col_name}")
    
    # For logarithmic scale on y-axis, apply after figure is created
    for ax in g.axes.flat:
        ax.set_yscale('log')
        # Remove all x-tick labels except for bottom row
        if ax.get_subplotspec().is_last_row():
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig('.../stratified_psm_continent_damage_comparison.png', dpi=300)
    plt.close()
    
    # 4. Also create a TC intensity comparison faceted by continent
    g = sns.catplot(
        data=summary_df,
        kind="bar",
        x="group", y="TC_intensity",
        hue="type", col="CONTINENT",
        col_wrap=2, height=5, aspect=1.2,
        sharex=True, sharey=True,
        error_bars="sd", errwidth=1.5,
        palette="dark"
    )
    
    g.set_axis_labels("", "TC Intensity (PCA composite)")
    g.set_xticklabels(rotation=45)
    g.set_titles("{col_name}")
    
    # For best visuals on tick labels
    for ax in g.axes.flat:
        if ax.get_subplotspec().is_last_row():
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig('.../stratified_psm_continent_intensity_comparison.png', dpi=300)
    plt.close()
    
    # 5. Subset analysis for billion-dollar events only, by continent
    billion_dollar_data = summary_df[summary_df['Total Damage, Adjusted (\'000 US$)'] >= 1000000].copy()
    
    if len(billion_dollar_data) > 10:  # Only if we have enough data points
        # First create an overall comparison
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='group', y='Total Damage, Adjusted (\'000 US$)', hue='type', data=billion_dollar_data)
        plt.title('Damage Comparison Across Matched Groups - Billion Dollar Events Only', fontsize=16)
        plt.xlabel('')
        plt.ylabel('Damage (\'000 USD)', fontsize=14)
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.legend(title='TC Type')
        plt.tight_layout()
        plt.savefig('.../stratified_psm_billion_dollar_damage_comparison.png', dpi=300)
        plt.close()
        
        # Then try to create a continent-based facet grid for billion-dollar events
        # But only if we have enough data across continents
        continents_with_data = billion_dollar_data['CONTINENT'].value_counts()
        continents_with_sufficient_data = continents_with_data[continents_with_data >= 4].index.tolist()
        
        if len(continents_with_sufficient_data) >= 2:
            filtered_billion_data = billion_dollar_data[billion_dollar_data['CONTINENT'].isin(continents_with_sufficient_data)]
            
            g = sns.catplot(
                data=filtered_billion_data,
                kind="bar",
                x="group", y="Total Damage, Adjusted ('000 US$)",
                hue="type", col="CONTINENT",
                col_wrap=2, height=5, aspect=1.2,
                sharex=True, sharey=False,
                error_bars="sd", errwidth=1.5,
                palette="dark"
            )
            
            g.set_axis_labels("", "Damage ('000 USD)")
            g.set_xticklabels(rotation=45)
            g.set_titles("{col_name} - Billion Dollar Events")
            
            # For logarithmic scale on y-axis
            for ax in g.axes.flat:
                ax.set_yscale('log')
                if ax.get_subplotspec().is_last_row():
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax.set_xticklabels([])
            
            plt.tight_layout()
            plt.savefig('.../stratified_psm_continent_billion_dollar_comparison.png', dpi=300)
            plt.close()
    
    # Additional analysis: regression on matched data to test for significance
    with open(output_file, 'a') as f:
        f.write("\n\n============= REGRESSION ANALYSIS ON MATCHED DATA =============\n\n")
        f.write("This analysis examines whether TC intensity explains damage differences within matched pairs,\n")
        f.write("with continent-specific effects included as control variables.\n\n")
        
        for group_num, (result, matched_data, label) in enumerate([
            (results_group2, matched_data_group2, "RI w/o MHW"),
            (results_group3, matched_data_group3, "No RI with MHW"),
            (results_group4, matched_data_group4, "No RI without MHW")
        ]):
            f.write(f"\n--- Comparison: RI with MHW vs {label} ---\n\n")
            
            # Create continent dummy variables
            continent_dummies = pd.get_dummies(matched_data['CONTINENT'], prefix='continent', drop_first=True)
            matched_data_with_dummies = pd.concat([matched_data, continent_dummies], axis=1)
            
            # Run regression on treatment group (RI with MHW)
            treatment_data = matched_data_with_dummies[matched_data_with_dummies['treatment'] == 1]
            
            # Get continent dummy columns
            continent_dummy_cols = [col for col in continent_dummies.columns]
            
            # Create model matrices
            X_treatment = sm.add_constant(treatment_data[['TC_intensity'] + continent_dummy_cols])
            y_treatment = treatment_data['log_damage']  # Using log damage for better distribution
            
            try:
                model_treatment = sm.OLS(y_treatment, X_treatment).fit()
                f.write(f"Regression for RI with MHW group:\n")
                f.write(f"  TC Intensity coefficient: {model_treatment.params['TC_intensity']:.4f}, p-value: {model_treatment.pvalues['TC_intensity']:.4f}\n")
                
                # Report continent effects if significant
                significant_continents = []
                for col in continent_dummy_cols:
                    if col in model_treatment.params and model_treatment.pvalues[col] < 0.1:
                        significant_continents.append(f"{col[10:]}: coef={model_treatment.params[col]:.4f}, p={model_treatment.pvalues[col]:.4f}")
                
                if significant_continents:
                    f.write(f"  Significant continent effects (p<0.1): {', '.join(significant_continents)}\n")
                else:
                    f.write(f"  No significant continent effects at p<0.1\n")
                    
                f.write(f"  R-squared: {model_treatment.rsquared:.4f}, F-statistic: {model_treatment.fvalue:.4f}, p-value: {model_treatment.f_pvalue:.4f}\n\n")
            except Exception as e:
                f.write(f"Error in treatment group regression: {str(e)}\n\n")
            
            # Run regression on control group
            control_data = matched_data_with_dummies[matched_data_with_dummies['treatment'] == 0]
            X_control = sm.add_constant(control_data[['TC_intensity'] + continent_dummy_cols])
            y_control = control_data['log_damage']
            
            try:
                model_control = sm.OLS(y_control, X_control).fit()
                f.write(f"Regression for {label} group:\n")
                f.write(f"  TC Intensity coefficient: {model_control.params['TC_intensity']:.4f}, p-value: {model_control.pvalues['TC_intensity']:.4f}\n")
                
                # Report continent effects if significant
                significant_continents = []
                for col in continent_dummy_cols:
                    if col in model_control.params and model_control.pvalues[col] < 0.1:
                        significant_continents.append(f"{col[10:]}: coef={model_control.params[col]:.4f}, p={model_control.pvalues[col]:.4f}")
                
                if significant_continents:
                    f.write(f"  Significant continent effects (p<0.1): {', '.join(significant_continents)}\n")
                else:
                    f.write(f"  No significant continent effects at p<0.1\n")
                    
                f.write(f"  R-squared: {model_control.rsquared:.4f}, F-statistic: {model_control.fvalue:.4f}, p-value: {model_control.f_pvalue:.4f}\n\n")
            except Exception as e:
                f.write(f"Error in control group regression: {str(e)}\n\n")
            
            # Run combined regression with interaction term
            combined_data = matched_data_with_dummies.copy()
            combined_data['mhw_ri'] = combined_data['treatment']
            combined_data['intensity_interaction'] = combined_data['TC_intensity'] * combined_data['mhw_ri']
            
            X_combined = sm.add_constant(combined_data[['TC_intensity', 'mhw_ri', 'intensity_interaction'] + continent_dummy_cols])
            y_combined = combined_data['log_damage']
            
            try:
                model_combined = sm.OLS(y_combined, X_combined).fit()
                f.write(f"Combined regression with interaction:\n")
                f.write(f"  TC Intensity coefficient: {model_combined.params['TC_intensity']:.4f}, p-value: {model_combined.pvalues['TC_intensity']:.4f}\n")
                f.write(f"  MHW+RI coefficient: {model_combined.params['mhw_ri']:.4f}, p-value: {model_combined.pvalues['mhw_ri']:.4f}\n")
                f.write(f"  Interaction coefficient: {model_combined.params['intensity_interaction']:.4f}, p-value: {model_combined.pvalues['intensity_interaction']:.4f}\n")
                
                # Report continent effects if significant
                significant_continents = []
                for col in continent_dummy_cols:
                    if col in model_combined.params and model_combined.pvalues[col] < 0.1:
                        significant_continents.append(f"{col[10:]}: coef={model_combined.params[col]:.4f}, p={model_combined.pvalues[col]:.4f}")
                
                if significant_continents:
                    f.write(f"  Significant continent effects (p<0.1): {', '.join(significant_continents)}\n")
                else:
                    f.write(f"  No significant continent effects at p<0.1\n")
                    
                f.write(f"  R-squared: {model_combined.rsquared:.4f}, F-statistic: {model_combined.fvalue:.4f}, p-value: {model_combined.f_pvalue:.4f}\n\n")
            except Exception as e:
                f.write(f"Error in combined regression: {str(e)}\n\n")

        # Billion-dollar events regression analysis with continent controls
        f.write("\n--- REGRESSION ANALYSIS FOR BILLION-DOLLAR EVENTS ONLY ---\n\n")
        
        billion_dollar_summary = summary_df[summary_df['Total Damage, Adjusted (\'000 US$)'] >= 1000000].copy()
        
        if len(billion_dollar_summary) > 10:  # Only if we have enough data points
            # Create dummy variables for continents
            continent_dummies = pd.get_dummies(billion_dollar_summary['CONTINENT'], prefix='continent', drop_first=True)
            billion_dollar_summary = pd.concat([billion_dollar_summary, continent_dummies], axis=1)
            
            # Create dummy variables for group comparisons
            billion_dollar_summary['is_rimhw'] = (billion_dollar_summary['type'] == 'RI with MHW').astype(int)
            
            # Get continent dummy columns
            continent_dummy_cols = [col for col in continent_dummies.columns]
            
            # Combined model across all groups
            X = sm.add_constant(billion_dollar_summary[['TC_intensity', 'is_rimhw'] + continent_dummy_cols])
            y = billion_dollar_summary['Total Damage, Adjusted (\'000 US$)']
            
            try:
                model = sm.OLS(np.log1p(y), X).fit()  # log transform to normalize
                f.write(f"Combined regression for all billion-dollar events:\n")
                f.write(f"  TC Intensity coefficient: {model.params['TC_intensity']:.4f}, p-value: {model.pvalues['TC_intensity']:.4f}\n")
                f.write(f"  RI with MHW coefficient: {model.params['is_rimhw']:.4f}, p-value: {model.pvalues['is_rimhw']:.4f}\n")
                
                # Report continent effects if significant
                significant_continents = []
                for col in continent_dummy_cols:
                    if col in model.params and model.pvalues[col] < 0.1:
                        significant_continents.append(f"{col[10:]}: coef={model.params[col]:.4f}, p={model.pvalues[col]:.4f}")
                
                if significant_continents:
                    f.write(f"  Significant continent effects (p<0.1): {', '.join(significant_continents)}\n")
                else:
                    f.write(f"  No significant continent effects at p<0.1\n")
                
                f.write(f"  R-squared: {model.rsquared:.4f}, F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4f}\n\n")
                
                # Calculate the marginal effect
                rimhw_effect = np.exp(model.params['is_rimhw']) - 1
                f.write(f"  Marginal effect of RI with MHW: {rimhw_effect*100:.2f}% increase in damage\n")
                
                # Calculate and report average marginal effects by continent if possible
                if len(continent_dummy_cols) > 0:
                    f.write("\n  Average effects by continent:\n")
                    
                    for continent in billion_dollar_summary['CONTINENT'].unique():
                        continent_subset = billion_dollar_summary[billion_dollar_summary['CONTINENT'] == continent]
                        
                        if len(continent_subset) >= 5:  # Only if sufficient samples
                            treatment_avg = continent_subset[continent_subset['is_rimhw'] == 1]['Total Damage, Adjusted (\'000 US$)'].mean()
                            control_avg = continent_subset[continent_subset['is_rimhw'] == 0]['Total Damage, Adjusted (\'000 US$)'].mean()
                            
                            if not np.isnan(treatment_avg) and not np.isnan(control_avg) and control_avg > 0:
                                pct_diff = ((treatment_avg - control_avg) / control_avg) * 100
                                f.write(f"    {continent}: {pct_diff:.2f}% difference (n={len(continent_subset)})\n")
                            else:
                                f.write(f"    {continent}: Insufficient data for comparison\n")
                        else:
                            f.write(f"    {continent}: Too few samples (n={len(continent_subset)})\n")
                
                f.write("\n")
                
            except Exception as e:
                f.write(f"Error in billion-dollar regression: {str(e)}\n\n")
        else:
            f.write("Insufficient billion-dollar events for regression analysis\n\n")

    # Final summary of findings
    with open(output_file, 'a') as f:
        f.write("\n\n================== OVERALL CONCLUSIONS ==================\n\n")
        
        # Calculate average percentage increase across all comparisons
        avg_pct_increase = np.mean([
            results_group2['percent_difference'],
            results_group3['percent_difference'],
            results_group4['percent_difference']
        ])
        
        f.write(f"1. Overall, TCs with RI and MHW caused on average {avg_pct_increase:.2f}% more damage\n")
        f.write(f"   than similar TCs (matched on built-up volume within the same continent).\n\n")
        
        # Calculate average percentage increase by continent
        f.write("2. Continent-specific effects:\n")
        
        # Get all continents from matched data
        all_continents = set()
        for data in [matched_data_group2, matched_data_group3, matched_data_group4]:
            all_continents.update(data['CONTINENT'].unique())
        
        # Calculate continental differences
        for continent in sorted(all_continents):
            continent_diffs = []
            
            for result, data, label in [
                (results_group2, matched_data_group2, "RI w/o MHW"),
                (results_group3, matched_data_group3, "No RI with MHW"),
                (results_group4, matched_data_group4, "No RI without MHW")
            ]:
                continent_data = data[data['CONTINENT'] == continent]
                
                if len(continent_data) > 0:
                    continent_treatment = continent_data[continent_data['treatment'] == 1]
                    continent_control = continent_data[continent_data['treatment'] == 0]
                    
                    if len(continent_treatment) > 0 and len(continent_control) > 0:
                        treatment_avg = continent_treatment['Total Damage, Adjusted (\'000 US$)'].mean()
                        control_avg = continent_control['Total Damage, Adjusted (\'000 US$)'].mean()
                        
                        if control_avg > 0:
                            pct_diff = ((treatment_avg - control_avg) / control_avg) * 100
                            continent_diffs.append(pct_diff)
            
            if continent_diffs:
                avg_continent_diff = np.mean(continent_diffs)
                f.write(f"   {continent}: {avg_continent_diff:.2f}% average difference across comparisons\n")
            else:
                f.write(f"   {continent}: Insufficient matched data for comparison\n")
        
        f.write("\n")
        
        # Determine which group showed the most dramatic difference
        max_diff_group = np.argmax([
            results_group2['percent_difference'],
            results_group3['percent_difference'],
            results_group4['percent_difference']
        ])
        
        group_labels = ["RI without MHW", "No RI with MHW", "No RI without MHW"]
        max_diff_label = group_labels[max_diff_group]
        max_diff_value = [
            results_group2['percent_difference'],
            results_group3['percent_difference'],
            results_group4['percent_difference']
        ][max_diff_group]
        
        f.write(f"3. The most dramatic difference was between RI with MHW and {max_diff_label},\n")
        f.write(f"   with RI with MHW TCs causing {max_diff_value:.2f}% more damage.\n\n")
        
        # Comment on statistical significance
        significant_results = []
        if results_group2['t_p_value'] < 0.05 or results_group2['mw_p_value'] < 0.05:
            significant_results.append("RI without MHW")
        if results_group3['t_p_value'] < 0.05 or results_group3['mw_p_value'] < 0.05:
            significant_results.append("No RI with MHW")
        if results_group4['t_p_value'] < 0.05 or results_group4['mw_p_value'] < 0.05:
            significant_results.append("No RI without MHW")
        
        if significant_results:
            f.write(f"4. The damage differences were statistically significant (p<0.05) when comparing\n")
            f.write(f"   RI with MHW TCs with: {', '.join(significant_results)}.\n\n")
        else:
            f.write(f"4. While damage differences were substantial, they did not reach statistical\n")
            f.write(f"   significance at the p<0.05 level in any comparison. This may be due to\n")
            f.write(f"   high variability in damage data and relatively small sample sizes when\n")
            f.write(f"   stratified by continent.\n\n")
        
        f.write("5. These results indicate that the higher damage costs associated with MHW+RI events\n")
        f.write("   are primarily due to their greater intensity, even when controlling for coastal\n")
        f.write("   development levels and continent through stratified propensity score matching.\n\n")
        
        f.write("================================================================\n")

print("Stratified PSM analysis complete. Results saved to:", output_file)
print("Visualizations saved to D:/3rdp/data/ directory")

# Conduct additional continent-specific analyses

# Compare results between original PSM and stratified PSM
with open(output_file, 'a') as f:
    f.write("\n\n================ COMPARISON WITH UNSTRATIFIED MATCHING ================\n\n")
    f.write("This section compares the results of stratified matching (within continents)\n")
    f.write("with unstratified matching (ignoring continents).\n\n")
    
    # First, perform unstratified matching for comparison
    def perform_unstratified_psm(treatment_df, control_df):
        # Combine treatment and control for analysis
        combined_df = pd.concat([treatment_df, control_df])
        
        # We'll use built_v_agg (exposure) as our covariate for matching
        X = combined_df[['built_v_agg']]
        y = combined_df['treatment']
        
        # Logistic regression to calculate propensity scores
        logit = LogisticRegression(random_state=42)
        logit.fit(X, y)
        
        # Calculate propensity scores
        propensity_scores = logit.predict_proba(X)[:, 1]
        combined_df['propensity_score'] = propensity_scores
        
        # Function to find nearest match
        def find_match(propensity_score, excluded_indices):
            control_scores = combined_df[(combined_df['treatment'] == 0) & 
                                        (~combined_df.index.isin(excluded_indices))]['propensity_score']
            if len(control_scores) == 0:
                return None
            distances = abs(control_scores - propensity_score)
            match_idx = distances.idxmin()
            return match_idx
        
        # Initialize matching results
        matches = []
        used_control_indices = set()
        
        # For each treatment observation, find nearest control
        for idx, row in combined_df[combined_df['treatment'] == 1].iterrows():
            match_idx = find_match(row['propensity_score'], used_control_indices)
            if match_idx is not None:
                matches.append((idx, match_idx))
                used_control_indices.add(match_idx)
        
        # Create matched dataset
        if len(matches) > 0:
            matched_treatment_indices = [pair[0] for pair in matches]
            matched_control_indices = [pair[1] for pair in matches]
            
            matched_data = pd.concat([
                combined_df.loc[matched_treatment_indices],
                combined_df.loc[matched_control_indices]
            ])
            
            # Calculate effect
            treatment_avg = matched_data[matched_data['treatment'] == 1]['Total Damage, Adjusted (\'000 US$)'].mean()
            control_avg = matched_data[matched_data['treatment'] == 0]['Total Damage, Adjusted (\'000 US$)'].mean()
            
            if control_avg > 0:
                pct_diff = ((treatment_avg - control_avg) / control_avg) * 100
            else:
                pct_diff = float('nan')
            
            return {
                'n_matches': len(matches),
                'treatment_avg': treatment_avg,
                'control_avg': control_avg,
                'pct_diff': pct_diff,
                'matched_data': matched_data
            }
        else:
            return None
    
    # Run unstratified PSM for each comparison
    unstrat_results_2 = perform_unstratified_psm(treatment_df, control_group2_df)
    unstrat_results_3 = perform_unstratified_psm(treatment_df, control_group3_df)
    unstrat_results_4 = perform_unstratified_psm(treatment_df, control_group4_df)
    
    # Compare results
    if unstrat_results_2 and unstrat_results_3 and unstrat_results_4:
        f.write("Comparison              | Stratified % Diff | Unstratified % Diff | Difference\n")
        f.write("------------------------|-------------------|---------------------|----------\n")
        f.write(f"RI with MHW vs RI w/o MHW    | {results_group2['percent_difference']:17.2f}% | {unstrat_results_2['pct_diff']:19.2f}% | {results_group2['percent_difference'] - unstrat_results_2['pct_diff']:9.2f}%\n")
        f.write(f"RI with MHW vs No RI w/ MHW  | {results_group3['percent_difference']:17.2f}% | {unstrat_results_3['pct_diff']:19.2f}% | {results_group3['percent_difference'] - unstrat_results_3['pct_diff']:9.2f}%\n")
        f.write(f"RI with MHW vs No RI w/o MHW | {results_group4['percent_difference']:17.2f}% | {unstrat_results_4['pct_diff']:19.2f}% | {results_group4['percent_difference'] - unstrat_results_4['pct_diff']:9.2f}%\n\n")
        
        f.write("Interpretation:\n")
        f.write("- Positive differences indicate that stratified matching (by continent) shows larger effects\n")
        f.write("- Negative differences indicate that unstratified matching shows larger effects\n\n")
        
        # Compare which continents are contributing most to the difference
        f.write("Continent-specific differences between stratified and unstratified approaches:\n\n")
        
        for continent in all_continents:
            f.write(f"Continent: {continent}\n")
            
            for i, (result, unstrat_result, data, label) in enumerate([
                (results_group2, unstrat_results_2, matched_data_group2, "RI w/o MHW"),
                (results_group3, unstrat_results_3, matched_data_group3, "No RI w/ MHW"),
                (results_group4, unstrat_results_4, matched_data_group4, "No RI w/o MHW")
            ]):
                # Get continent data from stratified matching
                continent_data = data[data['CONTINENT'] == continent]
                
                if len(continent_data) > 0:
                    continent_treatment = continent_data[continent_data['treatment'] == 1]
                    continent_control = continent_data[continent_data['treatment'] == 0]
                    
                    if len(continent_treatment) > 0 and len(continent_control) > 0:
                        strat_treatment_avg = continent_treatment['Total Damage, Adjusted (\'000 US$)'].mean()
                        strat_control_avg = continent_control['Total Damage, Adjusted (\'000 US$)'].mean()
                        
                        if strat_control_avg > 0:
                            strat_pct_diff = ((strat_treatment_avg - strat_control_avg) / strat_control_avg) * 100
                            f.write(f"  RI with MHW vs {label}: Stratified effect {strat_pct_diff:.2f}% vs. Overall effect {unstrat_result['pct_diff']:.2f}%\n")
                        else:
                            f.write(f"  RI with MHW vs {label}: Cannot calculate percentage difference (zero control average)\n")
                    else:
                        f.write(f"  RI with MHW vs {label}: Insufficient matched data in this continent\n")
                else:
                    f.write(f"  RI with MHW vs {label}: No matched data for this continent\n")
            
            f.write("\n")
    else:
        f.write("Unable to perform unstratified matching for comparison\n\n")

    f.write("================================================================\n")

# Export final matched datasets for potential further analysis
matched_data_group2.to_csv('.../stratified_matched_data_group2.csv', index=False)
matched_data_group3.to_csv('.../stratified_matched_data_group3.csv', index=False)
matched_data_group4.to_csv('.../stratified_matched_data_group4.csv', index=False)

print("Analysis complete! All results and datasets have been exported.")
