import pandas as pd
import matplotlib.pyplot as plt
# write a finction to read .csv files from following path and plot all the operating hours in a month
folder_path =   folder_path = "../../../data/iroshanij/greendata/"
file_1 = 'operating_hours_tunisia.csv'
file_2 = 'operating_times_tunisia.csv'
file_12 = 'operating_hours_oslo.csv'
file_22 = 'operating_times_oslo.csv'
file_13 = 'operating_hours_spain.csv'
file_23 = 'operating_times_spain.csv'

import matplotlib.pyplot as plt
import numpy as np

def plot_hours(df):
    # Ensure the index is of a type that allows addition (like numeric or datetime)
    # If the index is not of a type that allows this, convert it to a range index.
    df.reset_index(inplace=True)

    # Determine the width of each bar and the positions
    bar_width = 0.25
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create the figure and the axes
    plt.figure(figsize=(20, 10))
    
    # Create each bar. Note that you'll need to use the 'r1', 'r2', 'r3' as the x positions
    plt.bar(r1, df['Oslo'], width=bar_width, label='Oslo', edgecolor='grey')
    plt.bar(r2, df['Touzer'], width=bar_width, label='Touzer', edgecolor='grey')
    plt.bar(r3, df['Sevilla'], width=bar_width, label='Sevilla', edgecolor='grey')

    # Add xticks on the middle of the group bars
    plt.xlabel('Month', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(df))], df['Month'])  # Replace 'Month' with the actual name of your month column

    # Create labels and a title
    plt.ylabel('Operating Hours', fontweight='bold')
    
    # rotate x axis labels
    plt.xticks(rotation=90)
    # Create a legend
    plt.legend()

    # Save the plot
    plt.savefig('operating_hours.png')
    plt.show()  # If you want to display it



def main():
    # Assuming 'folder_path', 'file_1', 'file_12', and 'file_13' are predefined with correct file paths
    # Read the CSV files
    df_tunisia= pd.read_csv(folder_path + file_1)
    df_oslo= pd.read_csv(folder_path + file_12)
    df_spain = pd.read_csv(folder_path + file_13)
    
    # Verify the contents of each dataframe
    print(df_oslo.head())
    # add column names to the dataframe month and avg_operating_hours
    df_oslo.columns = ['Month', 'Oslo']
    df_oslo.set_index('Month', inplace=True)
    print(df_tunisia.head())
    # add column names to the dataframe month and avg_operating_hours
    df_tunisia.columns = ['Month', 'Touzer']
    df_tunisia.set_index('Month', inplace=True)
    print(df_spain.head())
    # add column names to the dataframe month and avg_operating_hours
    df_spain.columns = ['Month', 'Sevilla']
    df_spain.set_index('Month', inplace=True)
   

    # Combine the dataframes
    df_final = pd.concat([df_oslo, df_tunisia, df_spain], axis=1)
    print(df_final.head())

    # do the same for file_2 , file_22 and file_23 and combine all the dataframes into one dataframe
    # Assuming 'folder_path', 'file_2', 'file_22', and 'file_23' are predefined with correct file paths
    # Read the CSV files
    df_tunisia_time= pd.read_csv(folder_path + file_2)
    df_oslo_time= pd.read_csv(folder_path + file_22)
    df_spain_time = pd.read_csv(folder_path + file_23)

    # Verify the contents of each dataframe
    print(df_oslo_time.head())
    # add column names to the dataframe date and times
    df_oslo_time.columns = [Month         Day           Start Time             End Time  Operating Hours]
    df_oslo_time.set_index('Date', inplace=True)
    print(df_tunisia_time.head())
    # add column names to the dataframe Date and times
    df_tunisia_time.columns = [Month         Day           Start Time             End Time  Operating Hours]
    df_tunisia_time.set_index('Date', inplace=True)
    print(df_spain_time.head())
    # add column names to the dataframe Date and times
    df_spain_time.columns = [Month         Day           Start Time             End Time  Operating Hours]
    df_spain_time.set_index('Date', inplace=True)

    # Combine the dataframes
    df_final_time = pd.concat([df_oslo_time, df_tunisia_time, df_spain_time], axis=1)
    print(df_final_time.head())

    # Call the plot_hours function if needed - this is just an example call
    plot_hours(df_final)  # Assuming you have a function that takes the combined dataframe


if __name__ == "__main__":
    main()