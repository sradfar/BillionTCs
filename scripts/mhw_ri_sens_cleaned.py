# -----------------------------------------------------------------------------
# Python script developed by Soheil Radfar (sradfar@ua.edu), Postdoctoral Fellow
# Center for Complex Hydrosystems Research
# Department of Civil, Construction, and Environmental Engineering
# The University of Alabama
#
# Last modified on April 12, 2025
#
# This script performs a sensitivity analysis on the spatial-temporal overlap of MHWs
# with high-intensity tropical cyclones over a defined region (e.g., Gulf of Mexico).
# It processes different MHW threshold definitions and saves RI-MHW matches.
#
# Outputs:
# - Separate CSV files for each MHW criterion (e.g., 'MHW_RI_1981_2022_52.csv')
# - Counts and metadata for each storm-MHW pair
#
# Disclaimer:
# This script is intended for research and educational purposes only. It is provided 'as is' 
# without warranty of any kind, express or implied. The developer assumes no responsibility for 
# errors or omissions in this script. No liability is assumed for damages resulting from the use 
# of the information contained herein.
#
# -----------------------------------------------------------------------------
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# define a function to calculate the distance between two coordinates in km
def calc_dist(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371  # radius of the earth in km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

def process_mhw_data(data):
    csv_file, hi_df, output_file = data
    mhw_df = pd.read_csv(csv_file).dropna()

    result_df_list = []

    for hi_idx, hi_row in tqdm(hi_df.iterrows(), total=len(hi_df)):
        hi_lat = hi_row['HI_LAT']
        hi_lon = hi_row['HI_LON']
        hi_date = pd.to_datetime(hi_row['ISO_TIME'], infer_datetime_format=True, errors='coerce')
        observation_start = hi_date - pd.DateOffset(days=10)
        observation_end = hi_date
        mhw_df['date_start'] = pd.to_datetime(mhw_df['date_start'], infer_datetime_format=True, errors='coerce')
        mhw_df['date_end'] = pd.to_datetime(mhw_df['date_end'], infer_datetime_format=True, errors='coerce')

        mhw_df_filtered = mhw_df[
            (calc_dist(hi_lat, hi_lon, mhw_df['lat'], mhw_df['lon']) <= 200) 
            &
            (
                (mhw_df['date_start'] <= observation_end) &
                (mhw_df['date_end'] >= observation_start)
            )
        ]

        if not mhw_df_filtered.empty:
            hi_data_repeated = pd.concat([hi_row.to_frame().T] * len(mhw_df_filtered), ignore_index=True)
            combined_df = pd.concat([hi_data_repeated, mhw_df_filtered.reset_index(drop=True)], axis=1)
            result_df_list.append(combined_df)

    if result_df_list:
        result_df = pd.concat(result_df_list, ignore_index=True)
        return (result_df, output_file)
    else:
        # Return an empty DataFrame if no matching mhw events were found for any hi events
        return (pd.DataFrame(), output_file)

def filter_tc_by_region(df, min_lat=-15, max_lat=31, min_lon=-100, max_lon=-78):
    """
    Filter TCs to only include those that pass through the specified geographic box.
    
    Args:
        df: DataFrame containing TC track data
        min_lat, max_lat, min_lon, max_lon: Geographic boundaries
        
    Returns:
        DataFrame with only the TCs that pass through the box
    """
    # Create a unique identifier for each storm using NAME and SEASON
    df['STORM_ID'] = df['NAME'] + '_' + df['SEASON'].astype(str)
    
    # Group by storm ID and check if any point is within the box
    storm_ids = []
    
    for storm_id, group in df.groupby('STORM_ID'):
        within_box = ((group['HI_LAT'] >= min_lat) & 
                      (group['HI_LAT'] <= max_lat) & 
                      (group['HI_LON'] >= min_lon) & 
                      (group['HI_LON'] <= max_lon))
        
        if within_box.any():
            storm_ids.append(storm_id)
    
    # Filter the DataFrame to only include the identified storms
    return df[df['STORM_ID'].isin(storm_ids)]

def main():
    # Use available CPUs if not running on SLURM
    try:
        num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    except (TypeError, ValueError):
        num_cores = multiprocessing.cpu_count()

    hi_df = pd.read_csv('ibtracs_with_RI.csv')
    
    # Filter TCs to only include those that pass through the specified geographic box
    min_lat, max_lat = -15, 31
    min_lon, max_lon = -100, -78
    
    print(f"Original number of TC records: {len(hi_df)}")
    hi_df = filter_tc_by_region(hi_df, min_lat, max_lat, min_lon, max_lon)
    print(f"Number of TC records after filtering for geographic box: {len(hi_df)}")
    
    # Define specific MHW files and corresponding output files
    file_mappings = [
        ('MHW_1981_2022_52.csv', 'MHW_RI_1981_2022_52.csv'),
        ('MHW_1981_2022_80_52.csv', 'MHW_RI_1981_2022_80_52.csv')
        ('MHW_1981_2022_90_42.csv', 'MHW_RI_1981_2022_90_42.csv')
    ]
    
    data_to_process = [(input_file, hi_df, output_file) for input_file, output_file in file_mappings]

    # Create a progress bar for tracking file processing
    with tqdm(total=len(data_to_process), desc="Files Processed") as pbar_files:
        def update_progress(_):
            pbar_files.update(1)

        with multiprocessing.Pool(num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(process_mhw_data, data_to_process), 
                           total=len(data_to_process), desc="Rows Processed", position=0, leave=False))
    
    # Save each result to its corresponding output file
    for result_df, output_file in results:
        if not result_df.empty:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        else:
            print(f"No results found for {output_file}")

if __name__ == "__main__":
    main()
