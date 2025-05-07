import json
import numpy as np
import pandas as pd
from pandas import Timestamp

def save_dataset_to_json(data, filename, start_date):
    """
    Save a numpy array of time series data to a JSON file with GluonTS compatible format, 
    where each line in the JSON file is a separate dictionary.

    Args:
    data (np.array): 2D numpy array where each row is a time series.
    filename (str): Path to the JSON file where the dataset will be saved.
    start_date (str): The start date in 'YYYY-MM-DD hh:mm:ss' format.
    """
    start_ts = Timestamp(start_date)

    with open(filename, 'w') as f:
        for idx, series in enumerate(data):
            entry = {
                'target': series.tolist(),  
                'start': start_ts.isoformat(), 
                'feat_static_cat': [idx],
                'item_id': idx
            }
            json.dump(entry, f)  
            f.write('\n')  


def convert_hourly(df, output_name):
    # Convert Start and End to datetime
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    hourly_data = []

    for _, row in df.iterrows():
        start = row['Start']
        end = row['End']
        energy = row['Energy']
        
        # Iterate through all hours spanned by the charging session
        current_time = start
        while current_time < end:
            # Calculate the start and end of the current hour
            hour_start = current_time.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + pd.Timedelta(hours=1)
            
            # Calculate the overlap duration between the session and the current hour
            overlap_start = max(current_time, hour_start)
            overlap_end = min(end, hour_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600  # Convert to hours
            
            # Calculate the energy contributed to this hour
            hourly_energy = energy * overlap_duration / ((end - start).total_seconds() / 3600)
            
            # Append the hour and the calculated energy to the hourly data
            hourly_data.append({
                'Hour': hour_start,
                'Energy': hourly_energy
            })
            
            # Move to the next hour
            current_time = hour_end

    hourly_df = pd.DataFrame(hourly_data)

    # Aggregate the energy values by hour
    hourly_df = hourly_df.groupby('Hour', as_index=False).sum()

    # Generate a complete range of hours from the earliest to the latest timestamp
    all_hours = pd.date_range(start=hourly_df['Hour'].min(), 
                            end=hourly_df['Hour'].max(), 
                            freq='H')

    # Merge the hourly data with the full range of hours
    all_hours_df = pd.DataFrame({'Hour': all_hours})
    hourly_df = pd.merge(all_hours_df, hourly_df, on='Hour', how='left')

    # Fill missing energy values with 0
    hourly_df['Energy'] = hourly_df['Energy'].fillna(0)

    print(hourly_df)
    hourly_df.to_csv(output_name, index=False)


if __name__ == "__main__":

    datasets = ['caltech', 'sap', 'boulder']
    for dataset in datasets:
        df = pd.read_csv(dataset + '.csv')
        convert_hourly(df, dataset + '_hourly.csv')