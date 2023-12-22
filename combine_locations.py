# read two .csv files combined_data_tunisia.csv and combined_data_oslo.csv and combine them into one file with primary key period_end 
# and save it to combined_data.csv file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def combine_weather_power(df_oslo, df_tunisia, df_spain):
    # get the columns of the two dataframes
    columns_oslo = df_oslo.columns
    columns_tunisia = df_tunisia.columns
    columns_spain = df_spain.columns

    print("Oslo: ", columns_oslo)
    print("Tozeur: ", columns_tunisia) 
    print("Sevilla: ", columns_spain)

    # remove the unnamed column
    columns_spain = columns_spain[1:]

    # get the common columns in all three dataframes
    common_columns = list(set(columns_oslo).intersection(columns_tunisia))
    common_columns = list(set(common_columns).intersection(columns_spain))
   

    # get the columns that are not common in all three dataframes
    uncommon_columns = list(set(columns_oslo).symmetric_difference(columns_tunisia))
    uncommon_columns = list(set(uncommon_columns).symmetric_difference(columns_spain))


    # drop the uncommon columns from the two dataframes
    #df_oslo.drop(uncommon_columns, axis=1, inplace=True)
    #df_tunisia.drop(uncommon_columns, axis=1, inplace=True)
    #df_spain.drop(uncommon_columns, axis=1, inplace=True)

    # combine two dataframes vertically, assuming period_end is the primary key
    combined_df = pd.concat([df_oslo, df_tunisia, df_spain], ignore_index=True)
    # set the period_end as the index
    combined_df.set_index('period_end', inplace=True)




    return combined_df



def plot(df):
    # plot columns soliteck_mono, where location is Oslo and Tunisia in seperate plots
    df_oslo = df[df['location'] == 'Oslo']
    df_tunisia = df[df['location'] == 'Tozeur']
    df_spain = df[df['location'] == 'Sevilla']
    # plot the data
    plt.plot(df_oslo['period_end'], df_oslo['soliteck_mono'], label='Oslo')
    plt.plot(df_tunisia['period_end'], df_tunisia['soliteck_mono'], label='Tozeur')
    plt.plot(df_spain['period_end'], df_spain['soliteck_mono'], label='Sevilla')
    plt.xlabel('period_end')
    plt.ylabel('Wh')
    plt.legend()
    plt.show()

    # save the plot to a file
    plt.savefig("soliteck_mono.png")
   


if __name__ == "__main__":
    # read the csv files
    folder_path = "../../../data/iroshanij/greendata/"
    df_oslo = pd.read_csv(folder_path + "combined_data_oslo.csv")
    df_tunisia = pd.read_csv(folder_path + "combined_data_tunisia.csv")
    df_spain = pd.read_csv(folder_path + "combined_data_spain.csv")
   
    # combine the two dataframes
    combined_df = combine_weather_power(df_oslo, df_tunisia, df_spain)
    # describe the data
    print(combined_df.describe())
    print(combined_df.columns)
    # find the missing values for each column
    print(combined_df.isnull().sum())
    print(combined_df.head())

    #plot(combined_df)
    # print the period_end range
    # convert the period_end to datetime
    #combined_df['period_end'] = pd.to_datetime(combined_df['period_end'])
    #print("period_end range: ", combined_df['period_end'].min(), " to ", combined_df['period_end'].max())

    # save the combined dataframe to a csv file
    combined_df.to_csv(folder_path + "combined_data.csv")