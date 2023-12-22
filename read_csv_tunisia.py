import os
import pandas as pd
import dask.dataframe as dd

import dask
dask.config.set({'temporary_directory': '../../../data/iroshanij/greendata/'})

def cleanup_temp_files(filepaths):
    for filepath in filepaths:
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error occurred while deleting file {filepath}: {e}")

def read_csv_file(file_path): 
    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        return pd.DataFrame()

    # Read the raw content of the file line by line
    with open(file_path, 'r') as f:
        all_lines = f.readlines()

    # Identify the locations of the different header lines
    header_indices = [i for i, line in enumerate(all_lines) if "DATETIME,TimeStamp,GMT" in line]

    # Separate the content based on the header lines
    sections = [all_lines[header_indices[i]:header_indices[i+1]] if i+1 < len(header_indices) else all_lines[header_indices[i]:] 
                for i in range(len(header_indices))]

    # Convert each section into a Dask dataframe
    dataframes = []
    temp_files = []  # Keep track of temporary files to clean up later
    for idx, section in enumerate(sections):
        # Save the section to a temporary CSV file
        temp_filename = f"temp_section_tu_{idx}.csv"
        with open(temp_filename, 'w') as temp_file:
            temp_file.writelines(section)
        
        temp_files.append(temp_filename)  # Add to the list of temporary files
        
        # Read the temporary CSV file using Dask
        df = dd.read_csv(temp_filename, assume_missing=True)
        
        # Rename columns to avoid collisions
        prefix = f"sec_{idx}_"
        df = df.rename(columns={col: prefix + col if col not in ['DATETIME', 'TimeStamp', 'GMT'] else col for col in df.columns})
        df = df.reset_index(drop=True)  # Reset index and realign metadata
        
        dataframes.append(df)

    # Merge the Dask dataframes one by one
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = dd.merge(merged_df, df, on=['DATETIME', 'TimeStamp', 'GMT'], how='outer')

    # Compute the result to get a Pandas dataframe
    final_df = merged_df.compute()
    
    # Clean up the temporary files after computation
    cleanup_temp_files(temp_files)
    
    return final_df

def process_dataframe(df):
    # Extract unique column suffixes e.g., A1_Pin, A2_Pin, A10_Pin etc.
    suffixes = set(tuple(col.split('_')[-2:]) for col in df.columns if col.startswith('sec_'))
    suffixes = ['_'.join(s) for s in suffixes]

    # Store the aggregated series in a list
    aggregated_columns = []

    # include the columns 'DATETIME', 'TimeStamp', 'GMT'
    aggregated_columns.append(df[['DATETIME', 'TimeStamp', 'GMT']])

    # Aggregate data for each suffix and add to the list
    for suffix in suffixes:
        cols_to_aggregate = [col for col in df.columns if col.endswith(suffix)]
        aggregated_column = df[cols_to_aggregate].sum(axis=1)
        aggregated_column.name = suffix
        aggregated_columns.append(aggregated_column)

    # Concatenate all the aggregated columns
    agg_df = pd.concat(aggregated_columns, axis=1)

    # Drop unnecessary columns (Vin, Iin, and RSSI)
    agg_df.drop(columns=agg_df.filter(regex='.*Vin|.*Iin|.*RSSI').columns, inplace=True)

    return agg_df

# Main function and other parts remain unchanged


def main():
    """
    The main function of the read_csv.py script.
    """
    # Read all the csv files in the directory
    folder_path = "../../../data/iroshanij/greendata/"
    #paths = ["Tunisia/", "Oslo/"]
    paths = ["Tunisia/"]
    for path in paths:
        
        final = pd.DataFrame()
        for filename in os.listdir(folder_path + path):
            if filename.endswith(".csv"):
                print(os.path.join(folder_path + path, filename))
                df = read_csv_file(os.path.join(folder_path + path, filename))
                # Process the dataframe
                df = process_dataframe(df)
                # concatenate the dataframes
                final = pd.concat([final, df])

            else:
                continue
            #break

        # save the final dataframe to a csv file
        final.to_csv(folder_path + "final_with_panels_"+path.split("/")[0]+".csv")
        
    
        
    


    


if __name__ == "__main__":
    main()