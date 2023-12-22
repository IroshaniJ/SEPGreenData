import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_5min_average_power(df):
    """
    Calculates the 5-minute average power from a DataFrame.

    :param df: DataFrame with columns including time ('DATETIME', 'TimeStamp', 'GMT') and power ('A_Pin', 'B_Pni', 'C_Pin', 'D_Pin', 'E_Pin', 'F_Pin', 'G_Pin', 'H_Pin').
    :return: DataFrame with 5-minute average power, including a datetime column.
    """

    # Determine the time column used in the DataFrame
    time_columns = ['DATETIME', 'TimeStamp', 'GMT']
    time_column = next((col for col in time_columns if col in df.columns), None)
    if not time_column:
        raise ValueError("No valid time column found in the DataFrame.")

    # Convert the time column to datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Set the time column as the index
    df.set_index(time_column, inplace=True)

    # Power columns for Tunisia
    #power_columns = ['A_Pin', 'B_Pni', 'C_Pin', 'D_Pin', 'E_Pin', 'F_Pin', 'G_Pin', 'H_Pin']
    print(df.columns)
    # consider all the columns with the sufix _Pin
    power_columns = df.filter(regex='_Pin').columns
    # Power columns for Oslo
    #power_columns = ['A_Pin', 'B_Pin', 'C_Pin', 'D_Pin', 'H_Pin', 'I_Pin', 'J_Pin']
    power_columns = [col for col in power_columns if col in df.columns]

    if not power_columns:
        raise ValueError("No power columns found in the DataFrame.")

    # Group by one hour and calculate the sum
    avg_power_df = df[power_columns].resample('5T').sum()

    # convert the power to kWh
    avg_power_df = avg_power_df*(5/60)

    # Reset index to include datetime column in the DataFrame
    avg_power_df.reset_index(inplace=True)

    return avg_power_df

def calculate_panel_type_average_power(df):
    # aggregate the columns with the same suffix and store the result in a new column
    # this includes A1_Pin, A2_Pin, A3_Pin, A4_Pin, A5_Pin, A6_Pin, A7_Pin, A8_Pin, A9_Pin, A10_Pin
    
    A_Pi = df.filter(regex='A\d+_Pin').sum(axis=1)
    # count the number of columns with the sufix A\d_Pin
    A_count = max(df.filter(regex='A\d+_Pin').count(axis=1))
    print("A_count: ", A_count)
    # print the columns with the sufix A\d_Pin
    print(df.filter(regex='A\d+_Pin').columns)
    # remove the columns with the sufix A\d_Pin
    df.drop(df.filter(regex='A\d+_Pin').columns, axis=1, inplace=True)
    
    B_Pi = df.filter(regex='B\d+_Pin').sum(axis=1)
    # count the number of columns with the sufix B\d_Pin
    B_count = max(df.filter(regex='B\d+_Pin').count(axis=1))
    print("B_count: ", B_count)
    # print the columns with the sufix B\d_Pin
    print(df.filter(regex='B\d+_Pin').columns)
    # remove the columns with the sufix B\d_Pin
    df.drop(df.filter(regex='B\d+_Pin').columns, axis=1, inplace=True)
    df['A_Pin'] = A_Pi + B_Pi
   

    C_Pi = df.filter(regex='C\d+_Pin').sum(axis=1)
    # count the number of columns with the sufix C\d_Pin
    C_count = max(df.filter(regex='C\d+_Pin').count(axis=1))
    print("C_count: ", C_count)
    # print the columns with the sufix C\d_Pin
    print(df.filter(regex='C\d+_Pin').columns)
    # remove the columns with the sufix C\d_Pin
    df.drop(df.filter(regex='C\d+_Pin').columns, axis=1, inplace=True)
   
    D_Pi = df.filter(regex='D\d+_Pin').sum(axis=1)
    # count the number of columns with the sufix D\d_Pin
    D_count = max(df.filter(regex='D\d+_Pin').count(axis=1))
    print("D_count: ", D_count)
    # print the columns with the sufix D\d_Pin
    print(df.filter(regex='D\d+_Pin').columns)
    # remove the columns with the sufix D\d_Pin
    df.drop(df.filter(regex='D\d+_Pin').columns, axis=1, inplace=True)
    df['C_Pin'] = C_Pi + D_Pi
    

    # E1, E2, E3, E5, E6, E10 are manufacturer1_inn2_HFB
    # E4, E7, E8, E9, E11 are manufacturer1_inn1_HFC
    wp_E = [296.97, 295.13, 293.69, 295.16, 295.89, 295.72, 295.16, 293.26, 296.11, 293.46, 295.21]
    EHFB_Pi = df['E1_Pin']/wp_E[0] + df['E2_Pin']/wp_E[1] + df['E3_Pin']/wp_E[2] + df['E5_Pin']/wp_E[4] + df['E6_Pin']/wp_E[5] + df['E10_Pin']/wp_E[9]
    EHFL_Pi = df['E4_Pin']/wp_E[3] + df['E7_Pin']/wp_E[6] + df['E8_Pin']/wp_E[7] + df['E11_Pin']/wp_E[10]

    # count the number of columns with the sufix E\d_Pin
    E_count = max(df.filter(regex='E\d+_Pin').count(axis=1))
    print("E_count: ", E_count)
    # print the columns with the sufix E\d_Pin
    print(df.filter(regex='E\d+_Pin').columns)
    # remove the columns with the sufix E\d_Pin
    df.drop(df.filter(regex='E\d+_Pin').columns, axis=1, inplace=True)
    df['EHFB_Pin'] = EHFB_Pi
    df['EHFL_Pin'] = EHFL_Pi
    

    #F_Pi = df.filter(regex='F\d+_Pin').sum(axis=1)
    #df['F_Pin'] = F_Pi
    # filter only F1_Pin and F2_Pin
    wp_F = [302.47,304.26,303.50,302.89,303.45,302.36,303.29,301.96,302.63,302.52]
    F_Pi = df['F1_Pin']/wp_F[0] + df['F2_Pin']/wp_F[1] + df['F3_Pin']/wp_F[2] + df['F4_Pin']/wp_F[3] + df['F5_Pin']/wp_F[4] + df['F6_Pin']/wp_F[5] + df['F7_Pin']/wp_F[6] + df['F8_Pin']/wp_F[7] + df['F9_Pin']/wp_F[8] + df['F10_Pin']/wp_F[9]
    df['F_Pin'] = F_Pi
    # count the number of columns with the sufix F\d_Pin
    F_count = max(df.filter(regex='F\d+_Pin').count(axis=1))
    print("F_count: ", F_count)
    # print the columns with the sufix F\d_Pin
    print(df.filter(regex='F\d+_Pin').columns)
    # remove the columns with the sufix F\d_Pin
    df.drop(df.filter(regex='F\d+_Pin').columns, axis=1, inplace=True)

    #G_Pi = df.filter(regex='G\d+_Pin').sum(axis=1)
    #df['G_Pin'] = G_Pi
    # filter only G1_Pin and G2_Pin
    G_Pi = df.filter(regex='G[1,2]_Pin').sum(axis=1)
    df['G_Pin'] = G_Pi
    # count the number of columns with the sufix G\d_Pin
    G_count = max(df.filter(regex='G\d+_Pin').count(axis=1))
    print("G_count: ", G_count)
    # print the columns with the sufix G\d_Pin
    print(df.filter(regex='G\d+_Pin').columns)
    # remove the columns with the sufix G\d_Pin
    df.drop(df.filter(regex='G\d+_Pin').columns, axis=1, inplace=True)

    #H_Pi = df.filter(regex='H\d+_Pin').sum(axis=1)
    #df['H_Pin'] = H_Pi
    # filter only H1_Pin and H2_Pin
    wp_H =[302.47,304.26,303.50,302.89,303.45,302.36,303.29,301.96,302.63,302.52,302.51]
    H_Pi = df['H1_Pin']/wp_H[0] + df['H2_Pin']/wp_H[1] + df['H3_Pin']/wp_H[2] + df['H4_Pin']/wp_H[3] + df['H5_Pin']/wp_H[4] + df['H6_Pin']/wp_H[5] + df['H7_Pin']/wp_H[6] + df['H8_Pin']/wp_H[7] + df['H9_Pin']/wp_H[8] + df['H10_Pin']/wp_H[9] + df['H11_Pin']/wp_H[10]
    df['H_Pin'] = H_Pi
    # count the number of columns with the sufix H\d_Pin
    H_count = max(df.filter(regex='H\d+_Pin').count(axis=1))
    print("H_count: ", H_count)
    # print the columns with the sufix H\d_Pin
    print(df.filter(regex='H\d+_Pin').columns)
    # remove the columns with the sufix H\d_Pin
    df.drop(df.filter(regex='H\d+_Pin').columns, axis=1, inplace=True)

    I_Pi = df.filter(regex='I\d+_Pin').sum(axis=1)
    df['I_Pin'] = I_Pi
    # count the number of columns with the sufix I\d_Pin
    I_count = max(df.filter(regex='I\d+_Pin').count(axis=1))
    print("I_count: ", I_count)
    # print the columns with the sufix I\d_Pin
    print(df.filter(regex='I\d+_Pin').columns)
    # remove the columns with the sufix I\d_Pin
    df.drop(df.filter(regex='I\d+_Pin').columns, axis=1, inplace=True)

    wp_J = [300.8, 300.9, 300.5, 300.8, 302, 301.7]
    J_Pi = df['J1_Pin']/wp_J[0] + df['J2_Pin']/wp_J[1] + df['J3_Pin']/wp_J[2] + df['J4_Pin']/wp_J[3] + df['J5_Pin']/wp_J[4] + df['J6_Pin']/wp_J[5]
    df['J_Pin'] = J_Pi
    # count the number of columns with the sufix J\d_Pin
    J_count = max(df.filter(regex='J\d+_Pin').count(axis=1))
    print("J_count: ", J_count)
    # print the columns with the sufix J\d_Pin
    print(df.filter(regex='J\d+_Pin').columns)
    # remove the columns with the sufix J\d_Pin
    df.drop(df.filter(regex='J\d+_Pin').columns, axis=1, inplace=True)

    K_Pi = df.filter(regex='K\d+_Pin').sum(axis=1)
    df['K_Pin'] = K_Pi
    # count the number of columns with the sufix K\d_Pin
    K_count = max(df.filter(regex='K\d+_Pin').count(axis=1))
    print("K_count: ", K_count)
    # print the columns with the sufix K\d_Pin
    print(df.filter(regex='K\d+_Pin').columns)
    # remove the columns with the sufix K\d_Pin
    df.drop(df.filter(regex='K\d+_Pin').columns, axis=1, inplace=True)

    wp_L = [295.9, 294.7, 301.7, 293.3, 291.8, 292.6, 292.4]
    LHFB_Pi = df['L1_Pin']/wp_L[0] + df['L2_Pin']/wp_L[1] + df['L7_Pin']/wp_L[6]
    df['LHFB_Pin'] = LHFB_Pi
    LHLB_Pi = df['L3_Pin']/wp_L[2] + df['L4_Pin']/wp_L[3] + df['L5_Pin']/wp_L[4] + df['L6_Pin']/wp_L[5]
    df['LHFL_Pin'] = LHLB_Pi
    # count the number of columns with the sufix L\d_Pin
    L_count = max(df.filter(regex='L\d+_Pin').count(axis=1))
    print("L_count: ", L_count)
    # print the columns with the sufix L\d_Pin
    print(df.filter(regex='L\d+_Pin').columns)
    # remove the columns with the sufix L\d_Pin
    df.drop(df.filter(regex='L\d+_Pin').columns, axis=1, inplace=True)

   
    return df



def combine_weather_power(weather_df, power_df):
    """
    Combines weather and power data based on overlapping time periods.

    :param weather_df: DataFrame containing weather data with a 'period_end' column.
    :param power_df: DataFrame containing power data with a 'DATETIME' column.
    :return: Combined DataFrame with overlapping rows from both dataframes.
    """
    
    # Convert 'period_end' and 'DATETIME' to datetime if they aren't already
    weather_df['period_end'] = pd.to_datetime(weather_df['period_end'])
    power_df['DATETIME'] = pd.to_datetime(power_df['DATETIME'])

    # Set 'period_end' and 'DATETIME' as indices
    weather_df.set_index('period_end', inplace=True)
    power_df.set_index('DATETIME', inplace=True)

    weather_df.index = weather_df.index.tz_localize(None)
    # drop the column period
    weather_df.drop('period', axis=1, inplace=True)
    # calculate the average of the weather data for each 1 hour
    #weather_df = weather_df.resample('1H').mean()




    # Find overlapping time range
    common_start = max(weather_df.index.min(), power_df.index.min())
    common_end = min(weather_df.index.max(), power_df.index.max())

    print("common_start: ", common_start)
    print("common_end: ", common_end)

    # Filter both DataFrames to the common time range
    weather_df_common = weather_df.loc[common_start:common_end]
    power_df_common = power_df.loc[common_start:common_end]

    # Combine the two DataFrames keeping period_end as the index
    combined_df = weather_df_common.join(power_df_common)

    # Reset the index to include period_end as a column
    combined_df.reset_index(inplace=True)

    
    return combined_df

def plot_compare_and_improvement(df, type1, type2):
    # create a copy of the dataframe
    df = df.copy()
    
    # remove the rows with missing values
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0))]
    # remove the rows with null values 
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()))]

    # remove the data 2021-05
    df = df[~((df['period_end'].dt.month == 5) & (df['period_end'].dt.year == 2021))]
    # remove data from 2021-06-14, 2021-07-30, 2021-12-03, 2022-01-11, 2022-01-25,
    # 2022-02-03, 2022-02-14, 
    # 2022-06-06, 2022-06-20, 2022-06-21
    # 2022-07-12, 2022-08-13, 2023-05-18
    df = df.loc[~((df['period_end'] == '2021-06-14') | 
    (df['period_end'] == '2021-07-30') | 
    (df['period_end'] == '2021-12-03') | 
    (df['period_end'] == '2022-01-11') | 
    (df['period_end'] == '2022-01-25') | 
    (df['period_end'] == '2022-02-03') | 
    (df['period_end'] == '2022-02-14') | 
    (df['period_end'] == '2022-06-06') | 
    (df['period_end'] == '2022-06-20') | 
    (df['period_end'] == '2022-06-21') |
     (df['period_end'] == '2022-07-12') | 
     (df['period_end'] == '2022-08-13') | 
     (df['period_end'] == '2023-05-18'))] 

    # Convert 'period_end' to datetime and set as index
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    df.set_index('period_end', inplace=True)

    # Create a 'month-year' column for aggregation
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')
    df[type1] = df[type1]
    df[type2] = df[type2]

    # Aggregate data by 'month-year'
    df_grouped = df.groupby('month-year').agg({type1: 'sum', type2: 'sum'}).reset_index()

    # Calculate the difference (improvement) between type2 and type1
    df_grouped['improvement'] = df_grouped[type2] - df_grouped[type1]

    # Calculate total improvement as a percentage
    total_improvement_percentage = (df_grouped['improvement'].sum() / df_grouped[type1].sum()) * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(20, 10))
    sns.set(style="darkgrid")

    # Offset for side-by-side bars
    bar_width = 0.35
    positions = np.arange(len(df_grouped))

    # Plot for type1 and type2 with offset
    ax1.bar(positions - bar_width/2, df_grouped[type1], bar_width, label=type1)
    ax1.bar(positions + bar_width/2, df_grouped[type2], bar_width, label=type2)

    # Adding month-year labels to x-axis
    ax1.set_xticks(positions)
    ax1.set_xticklabels(df_grouped['month-year'], rotation=90)

    # Create a second y-axis for improvement plot
    ax2 = ax1.twinx()
    sns.lineplot(x=positions, y='improvement', data=df_grouped, marker='o', color='green', label='Deviation', ax=ax2)

    # Setting labels and titles
    ax1.set_xlabel('Month-Year', fontsize=20)
    ax1.set_ylabel('kWh/kWp', fontsize=20)
    ax2.set_ylabel('Deviation (kWh/KWp)', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'plots/Spain/{type1}_{type2}_improvement.png')
    
    return total_improvement_percentage

# Assume df is your DataFrame
def plot_compare_three(df, type1, type2, type3):
    # create a copy of the dataframe
    df = df.copy()
     # drop the columns with missing values and zero values only for the colunms 'manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL'
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0) |  (df[type3] == 0))]
    # drop null values only for the colunms 'manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL'
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()) | (df[type3].isnull()))]
    # remove data from 2021-05-01 to 2021-05-31
    df = df[~((df['period_end'].dt.month == 5) & (df['period_end'].dt.year == 2021))]
    # remove data from 2021-06-14, 2021-07-30, 2021-12-03, 2022-01-11, 2022-01-25,
    # 2022-02-03, 2022-02-14, 
    # 2022-06-06, 2022-06-20, 2022-06-21
    # 2022-07-12, 2022-08-13, 2023-05-18
    #df = df.loc[~((df['period_end'] == '2021-06-14') | 
    #(df['period_end'] == '2021-07-30') | 
   # (df['period_end'] == '2021-12-03') | 
   # (df['period_end'] == '2022-01-11') | 
   # (df['period_end'] == '2022-01-25') | 
   # (df['period_end'] == '2022-02-03') | 
   # (df['period_end'] == '2022-02-14') | 
    #(df['period_end'] == '2022-06-06') | 
    #(df['period_end'] == '2022-06-20') | 
    #(df['period_end'] == '2022-06-21') |
    # (df['period_end'] == '2022-07-12') | 
    # (df['period_end'] == '2022-08-13') | 
    # (df['period_end'] == '2023-05-18'))] 

    df = df.loc[~((df['period_end'] == '2022-01-11') | (df['period_end'] == '2022-01-25') 
                   | (df['period_end'] == '2022-01-04') | (df['period_end'] == '2022-01-10')
                   | (df['period_end'] == '2022-01-24') | (df['period_end'] == '2022-01-27'))]

    # Create a 'month-year' column for aggregation
    # Convert 'period_end' to datetime and set as index
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    df.set_index('period_end', inplace=True)

    # save the dataframe to a csv file incuding index, type1, type2, and type3
    df.to_csv('plots/Spain/spain.csv', index=True, columns=[type1, type2, type3])
    
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')
    df[type1] = df[type1]
    df[type2] = df[type2]
    df[type3] = df[type3]

    # Aggregate data by 'month-year'
    df_grouped = df.groupby('month-year').agg({type1: 'sum', type2: 'sum', type3: 'sum'}).reset_index()

    # Calculate differences
    diff_1_2 = (df_grouped[type2] - df_grouped[type1]) 
    diff_1_3 = (df_grouped[type3] - df_grouped[type1])

    # Calculate average percentage differences
    total_improvement_percentage_12 = (diff_1_2.sum() / df_grouped[type1].sum()) 
    total_improvement_percentage_13 = (diff_1_3.sum() / df_grouped[type1].sum()) 

    # Plotting
    fig, ax1 = plt.subplots(figsize=(20, 10))
    sns.set(style="darkgrid")

    # Offset for side-by-side bars
    bar_width = 0.25
    positions = np.arange(len(df_grouped))

    # Plot for type1, type2, and type3 with offset
    ax1.bar(positions - bar_width, df_grouped[type1], bar_width, label=type1, color='orange')
    ax1.bar(positions, df_grouped[type2], bar_width, label=type2, color='purple')
    ax1.bar(positions + bar_width, df_grouped[type3], bar_width, label=type3, color='green')

    # Deviation plots
    ax2 = ax1.twinx()
    ax2.plot(positions, diff_1_2, color='purple', marker='o', label=f'{type2} vs {type1} Deviation')
    ax2.plot(positions, diff_1_3, color='green', marker='o', label=f'{type3} vs {type1} Deviation')

    # Adding month-year labels to x-axis
    ax1.set_xticks(positions)
    ax1.set_xticklabels(df_grouped['month-year'], rotation=90)

    # Setting labels and titles
    ax1.set_xlabel('Month-Year', fontsize=20)
    ax1.set_ylabel('kWh/kWp', fontsize=20)
    ax2.set_ylabel('Deviation (kWh/kWp)', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'plots/Spain/{type1}_{type2}_{type3}.png')
   
    
    

    # Return percentage differences
    return total_improvement_percentage_12, total_improvement_percentage_13
    
import pandas as pd

def remove_anomalies(df, column, window='1H', std_devs=200):
    """
    Remove anomalies from a PV dataset based on rolling mean and standard deviation.

    :param df: DataFrame containing the PV data
    :param column: Name of the column to process
    :param window: Size of the rolling window (default is '5T' for 5 minutes)
    :param std_devs: Number of standard deviations to use for anomaly detection
    :return: DataFrame with anomalies removed
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
    rolling_std = df[column].rolling(window=window, min_periods=1).std()

    # Identify anomalies as values that are a certain number of standard deviations from the mean
    lower_bound = rolling_mean - (rolling_std * std_devs)
    upper_bound = rolling_mean + (rolling_std * std_devs)

    # Filter out anomalies
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered

def plot_daily_power_variation(df, columns):
    date_column = 'period_end'
    # set the index to date_column
    df = df.set_index(date_column)
    # Create a month-year column for grouping
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')

    # Get the unique month-year combinations
    month_years = df['month-year'].unique()

    for my in month_years:
        # Filter data for the specific month-year
        df_my = df[df['month-year'] == my]

        # Melt the DataFrame for the columns to be plotted
        df_melted = df_my.reset_index().melt(id_vars=date_column, value_vars=columns, var_name='PanelType', value_name='Power')

        # Determine the number of days in the month
        num_days = df_my.index.day.max()

        # Create subplots for each day of the month
        fig, axs = plt.subplots(1, num_days, figsize=(30, 15), sharey=True)

        for i in range(1, num_days + 1):
            # Filter data for the specific day
            df_day = df_melted[df_melted[date_column].dt.day == i]

            # Plot the daily power variation
            sns.lineplot(x=df_day[date_column].dt.hour, y='Power', hue='PanelType', data=df_day, ax=axs[i-1])

            # Set the title for each subplot
            axs[i-1].set_title(f'Day {i}')
            axs[i-1].set_xlabel('Hour')

        # Set the title for the figure and adjust layout
        fig.suptitle(f'Daily Power Variation for {my}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        plt.savefig(f'plots/Spain/daily_power_variation_{my}.png')
        plt.close(fig)  # Close the figure to free memory

def plot_monthly_average_operating_hours(df, panel_type, power_threshold=0.1):
    # Ensure that the 'period_end' column is in datetime format
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Filter the DataFrame for the given panel type
    panel_data = df[['period_end', panel_type]]

    # Determine if the panel is operating (power > threshold)
    panel_data['Operating'] = panel_data[panel_type] > power_threshold

    # Group by month
    monthly_data = panel_data.set_index('period_end').groupby(pd.Grouper(freq='M'))

    # Initialize a dictionary to store average operating hours for each month
    avg_operating_hours_monthly = {}
    monthly_operating_times = []

    for month, group in monthly_data:
        # Group by day within the month
        daily_data = group.groupby(group.index.date)

        # List to store operating hours for each day
        daily_operating_hours = []

        for day, daily_group in daily_data:
            operating_periods = daily_group[daily_group['Operating']]
            if not operating_periods.empty:
                start_time = operating_periods.index.min()
                end_time = operating_periods.index.max()
                operating_duration = (end_time - start_time).total_seconds() / 3600  # Convert to hours
                daily_operating_hours.append(operating_duration)
                monthly_operating_times.append({
                    'Month': month.strftime('%Y-%m'),
                    'Day': day,
                    'Start Time': start_time,
                    'End Time': end_time,
                    'Operating Hours': operating_duration
                })

        # Calculate average operating hours for the month
        if daily_operating_hours:
            avg_operating_hours = sum(daily_operating_hours) / len(daily_operating_hours)
            avg_operating_hours_monthly[month.strftime('%Y-%m')] = avg_operating_hours

    # Convert the average operating hours dictionary to a Series for plotting
    avg_operating_hours_series = pd.Series(avg_operating_hours_monthly)

    # Plot
    plt.figure(figsize=(12, 6))
    avg_operating_hours_series.plot(kind='bar')
    plt.title(f'Average Monthly Operating Hours of {panel_type} Panel')
    plt.xlabel('Month')
    plt.ylabel('Average Operating Hours')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/Spain/average_operating_hours_{panel_type}.png')
    plt.close()

    # Convert monthly operating times to DataFrame
    operating_times_df = pd.DataFrame(monthly_operating_times)

    return avg_operating_hours_series, operating_times_df

def main():
   
    # Read th power data file
    folder_path = "../../../data/iroshanij/greendata/"
    power_df = pd.read_csv(folder_path + "final_with_panels_Spain.csv")
    weather_df1 = pd.read_csv(folder_path + "weather_spain.csv")
    weather_df2 = pd.read_csv(folder_path + "weather_spain_2.csv")
    weather_df = pd.concat([weather_df1, weather_df2])

    # remove duplicate rows from weather_df
    # Removing duplicates (keeping the first occurrence)
    weather_df = weather_df[~weather_df.index.duplicated(keep='first')]
    
    # describe the data
    print(power_df.describe())
    print(power_df.columns)
    print(weather_df.describe())
    print(weather_df.columns)

    # get the columns with missing values
    missing_cols = power_df.columns[power_df.isnull().any()].tolist()
    print(missing_cols)

    # count the number of missing values for each column in each month-year
    # set the index to DATETIME
    # power_df['DATETIME'] = pd.to_datetime(power_df['DATETIME'])
    # power_df.set_index('DATETIME', inplace=True)
   
    # replace the missing values with previous values
    # power_df.fillna(method='ffill', inplace=True)
   
    # count the number of missing values for each column in each month-year
    # power_df.groupby([power_df.index.year, power_df.index.month])[missing_cols].count()
    # plot the missing values for each column in each month-year
    # for col in missing_cols:
    #     plt.figure()
    #     power_df.groupby([power_df.index.year, power_df.index.month])[col].count().plot()
    #     change the x label to month-year format
    #     plt.xlabel('month-year')
    #     plt.ylabel('number of missing values')
    #     plt.title('Missing values for '+col)
    #     plt.savefig('plots/Spain/'+col+'.png')


    # remove the missing column if it has more than 20% missing values out of the total number of rows
    for col in missing_cols:
        if power_df[col].isnull().sum() > 0.25*len(power_df[col]):
            #power_df.drop(col, axis=1, inplace=True)
            #print("Dropped column: ", col)
            print("Column: ", col, " has more than 25% missing values out of the total number of rows")


    # remove the columns with missing values, dropped only E8_Pin
    #missing_cols = ['E8_Pin']
    #power_df.drop(missing_cols, axis=1, inplace=True)
    
    avg_power_df = calculate_5min_average_power(power_df)

    # aggregate the columns with the same suffix and store the result in a new column
    avg_power_df = calculate_panel_type_average_power(avg_power_df)

    # Spain
    # solitek_mono = ['G_Pin']
    # manufacturer1_bi = ['E_Pin']
    # manufacturer1_inn1 = ['I_Pin']
    # manufacturer1_inn2 = ['H_Pin']
    # manufacturer2_inn1 = ['K_Pin']
    # manufacturer2_inn2 = ['L_Pin']
    # manufacturer2_bi-facial = ['J_Pin']
    # CIGS_inn1 = ['D_Pin']
    # CIGS_inn2 = ['F_Pin']
    # CIGS_bi-facial = ['B_Pin']

    avg_power_df.rename(columns={'F_Pin': 'manufacturer1_mono-facial', 
                                'H_Pin': 'manufacturer1_bi-facial', 
                                'G_Pin': 'manufacturer1_inn1',
                                'EHFB_Pin': 'manufacturer1_inn2_HFB', 
                                'EHFL_Pin': 'manufacturer1_inn2_HFL',
                                'LHFB_Pin': 'manufacturer2_inn2_HFB', 
                                'LHFL_Pin': 'manufacturer2_inn2_HFL',
                                'J_Pin': 'manufacturer2_bi-facial', 
                                'K_Pin': 'manufacturer2_inn1',
                                'A_Pin': 'CIGS_bi-facial',
                                'C_Pin': 'CIGS_inn1',
                                'I_Pin': 'CIGS_inn2'}, inplace=True)
    
    # drop the columns with siffix CIGS_, manufacturer2_inn1, manufacturer1_inn1
    avg_power_df.drop(['CIGS_inn1', 'CIGS_inn2', 'CIGS_bi-facial', 'manufacturer2_inn1', 'manufacturer1_inn1'], axis=1, inplace=True)


    # add a new column with the location name and set it to Tunisia
    avg_power_df['location'] = 'Sevilla'

    # calculate averages of power for each panel type in Tunisia
    # soltek_mono = avg_power_df['manufacturer1_mono']/10
    # soltek_bi = avg_power_df['manufacturer1_bi']/11
    # soltek_inn1 = avg_power_df['manufacturer1_inn1']/10
    # soltek_inn2 = avg_power_df['manufacturer1_inn2']/11
    # manufacturer2_inn1 = avg_power_df['manufacturer2_inn1']/7
    # manufacturer2_inn2 = avg_power_df['manufacturer2_inn2']/7
    # manufacturer2_bi-facial = avg_power_df['manufacturer2_bi-facial']/6
    # CIGS_bi-facial = avg_power_df['CIGS_bi-facial']/10 (20)
    # CIGS_inn1 = avg_power_df['CIGS_inn1']/8 (16)
    # CIGS_inn2 = avg_power_df['CIGS_inn2']/5 (10)

    #     A_count:  10
    # Index(['A9_Pin', 'A1_Pin', 'A3_Pin', 'A5_Pin', 'A4_Pin', 'A6_Pin', 'A10_Pin',
    #        'A8_Pin', 'A7_Pin', 'A2_Pin'],
    #       dtype='object')
    # B_count:  10
    # Index(['B9_Pin', 'B6_Pin', 'B4_Pin', 'B3_Pin', 'B10_Pin', 'B8_Pin', 'B2_Pin',
    #        'B1_Pin', 'B7_Pin', 'B5_Pin'],
    #       dtype='object')
    # C_count:  8
    # Index(['C2_Pin', 'C1_Pin', 'C3_Pin', 'C4_Pin', 'C6_Pin', 'C8_Pin', 'C5_Pin',
    #        'C7_Pin'],
    #       dtype='object')
    # D_count:  8
    # Index(['D2_Pin', 'D4_Pin', 'D7_Pin', 'D5_Pin', 'D8_Pin', 'D6_Pin', 'D3_Pin',
    #        'D1_Pin'],
    #       dtype='object')
    # E_count:  11
    # Index(['E9_Pin', 'E2_Pin', 'E3_Pin', 'E5_Pin', 'E1_Pin', 'E4_Pin', 'E7_Pin',
    #        'E6_Pin', 'E11_Pin', 'E10_Pin', 'E8_Pin'],
    #       dtype='object')
    # F_count:  10
    # Index(['F5_Pin', 'F1_Pin', 'F3_Pin', 'F8_Pin', 'F10_Pin', 'F9_Pin', 'F2_Pin',
    #        'F4_Pin', 'F6_Pin', 'F7_Pin'],
    #       dtype='object')
    # G_count:  10
    # Index(['G4_Pin', 'G1_Pin', 'G8_Pin', 'G3_Pin', 'G7_Pin', 'G2_Pin', 'G6_Pin',
    #        'G5_Pin', 'G10_Pin', 'G9_Pin'],
    #       dtype='object')
    # H_count:  11
    # Index(['H4_Pin', 'H8_Pin', 'H6_Pin', 'H2_Pin', 'H9_Pin', 'H11_Pin', 'H3_Pin',
    #        'H1_Pin', 'H5_Pin', 'H7_Pin', 'H10_Pin'],
    #       dtype='object')
    # I_count:  10
    # Index(['I7_Pin', 'I10_Pin', 'I2_Pin', 'I8_Pin', 'I6_Pin', 'I3_Pin', 'I4_Pin',
    #        'I5_Pin', 'I9_Pin', 'I1_Pin'],
    #       dtype='object')
    # J_count:  6
    # Index(['J6_Pin', 'J5_Pin', 'J2_Pin', 'J4_Pin', 'J3_Pin', 'J1_Pin'], dtype='object')
    # K_count:  7
    # Index(['K1_Pin', 'K5_Pin', 'K6_Pin', 'K3_Pin', 'K2_Pin', 'K4_Pin', 'K7_Pin'], dtype='object')
    # L_count:  7
    # Index(['L3_Pin', 'L5_Pin', 'L2_Pin', 'L4_Pin', 'L6_Pin', 'L7_Pin', 'L1_Pin'], dtype='object')

    print(avg_power_df.describe())

    avg_power_df['manufacturer1_mono-facial'] = avg_power_df['manufacturer1_mono-facial']/10
    avg_power_df['manufacturer1_bi-facial'] = avg_power_df['manufacturer1_bi-facial']/11
    avg_power_df['manufacturer1_inn2_HFB'] = avg_power_df['manufacturer1_inn2_HFB']/6
    avg_power_df['manufacturer1_inn2_HFL'] = avg_power_df['manufacturer1_inn2_HFL']/4
    avg_power_df['manufacturer2_inn2_HFB'] = avg_power_df['manufacturer2_inn2_HFB']/3
    avg_power_df['manufacturer2_inn2_HFL'] = avg_power_df['manufacturer2_inn2_HFL']/4
    avg_power_df['manufacturer2_bi-facial'] = avg_power_df['manufacturer2_bi-facial']/6

    
    combined_df = combine_weather_power(weather_df, avg_power_df)
    # remove the rows with missing values
    #combined_df = combined_df.dropna()
    # remove the rows with zeros
    #combined_df = combined_df.loc[~((combined_df['manufacturer1_mono'] == 0) | (combined_df['manufacturer1_bi'] == 0)| (combined_df['manufacturer1_inn2_HFB'] == 0) | (combined_df['manufacturer1_inn2_HFL'] == 0) | (combined_df['manufacturer2_inn2'] == 0) | (combined_df['manufacturer2_bi-facial'] == 0))]
    #combined_df = avg_power_df
    
    # rename the columns DATETIME to period_end
    combined_df.rename(columns={'DATETIME': 'period_end'}, inplace=True)

    #plot_daily_power_variation(combined_df, ['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL'])
    

    #print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL', 'manufacturer2_inn1', 'manufacturer2_inn2', 'manufacturer2_bi-facial']].describe())

   
    nominal_plant_output = (combined_df['ghi']*(5/60))/1000
    combined_df['manufacturer1_mono-facial_pr'] = (combined_df['manufacturer1_mono-facial'])/nominal_plant_output
    combined_df['manufacturer1_bi-facial_pr'] = (combined_df['manufacturer1_bi-facial'])/nominal_plant_output
    combined_df['manufacturer1_inn2_HFB_pr'] = (combined_df['manufacturer1_inn2_HFB'])/nominal_plant_output
    combined_df['manufacturer1_inn2_HFL_pr'] = (combined_df['manufacturer1_inn2_HFL'])/nominal_plant_output
    combined_df['manufacturer2_inn2_HFB_pr'] = (combined_df['manufacturer2_inn2_HFB'])/nominal_plant_output
    combined_df['manufacturer2_inn2_HFL_pr'] = (combined_df['manufacturer2_inn2_HFL'])/nominal_plant_output
    combined_df['manufacturer2_bi-facial_pr'] = (combined_df['manufacturer2_bi-facial'])/nominal_plant_output


    solar_radiation = combined_df['ghi']*1.89072
    combined_df['manufacturer1_mono-facial_eff'] = combined_df['manufacturer1_mono-facial']/solar_radiation
    combined_df['manufacturer1_bi-facial_eff'] = combined_df['manufacturer1_bi-facial']/solar_radiation
    combined_df['manufacturer1_inn2_HFB_eff'] = combined_df['manufacturer1_inn2_HFB']/solar_radiation
    combined_df['manufacturer1_inn2_HFL_eff'] = combined_df['manufacturer1_inn2_HFL']/solar_radiation
    combined_df['manufacturer2_inn2_HFB_eff'] = combined_df['manufacturer2_inn2_HFB']/solar_radiation
    combined_df['manufacturer2_inn2_HFL_eff'] = combined_df['manufacturer2_inn2_HFL']/solar_radiation
    combined_df['manufacturer2_bi-facial_eff'] = combined_df['manufacturer2_bi-facial']/solar_radiation
   
    # remove anomulous data for each panel type
    #panel_types = ['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL']
    #std_devs = [3, 3, 3, 3, 3]
    #for panel_type in panel_types:
    #    combined_df = remove_anomalies(combined_df, panel_type, std_devs=std_devs[panel_types.index(panel_type)])

    print(combined_df.columns)
    # remove rows with zeros, negetive values, and missing values
    #print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL', 'manufacturer2_inn1', 'manufacturer2_inn2', 'manufacturer2_bi-facial']].describe())
    
    improvement = plot_compare_and_improvement(combined_df, 'manufacturer1_mono-facial', 'manufacturer1_bi-facial')
    print("Improvement in bi in Spain: ", improvement)
    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer1_bi-facial', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL')
    print("Improvement in inn2_HFB in Spain: ", diff1)
    print("Improvement in inn2_HFL in Spain: ", diff2)
    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer2_bi-facial', 'manufacturer2_inn2_HFB', 'manufacturer2_inn2_HFL')
    print("Improvement in inn2_HFB in Spain: ", diff1)
    print("Improvement in inn2_HFL in Spain: ", diff2)

    combined_df.reset_index(inplace=True)
    operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_mono-facial', power_threshold=0.1)
     # save operating hours and operating times to csv files
    operating_hours_series.to_csv(folder_path + "operating_hours_spain.csv")
    operating_times_df.to_csv(folder_path + "operating_times_spain.csv")
    #print(operating_hours_series)
    #print(operating_times_df)
    #operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_bi', power_threshold=0.1)
    #print(operating_hours_series)
    #print(operating_times_df)

    # save the combined data to csv file
    combined_df.to_csv(folder_path + "combined_data_spain.csv")
    

if __name__ == "__main__":
    main()