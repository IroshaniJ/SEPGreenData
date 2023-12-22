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

    # Group by 5-minute intervals and calculate the sum of power values each 5 minute interval
    avg_power_df = df[power_columns].resample('5T').sum()

    # convert the power values to kWh
    avg_power_df = avg_power_df*(5/60)

    # Reset index to include datetime column in the DataFrame
    avg_power_df.reset_index(inplace=True)

    return avg_power_df

def calculate_panel_type_average_power(df):
    # aggregate the columns with the same suffix and store the result in a new column
    # this includes A1_Pin, A2_Pin, A3_Pin, A4_Pin, A5_Pin, A6_Pin, A7_Pin, A8_Pin, A9_Pin, A10_Pin
    
    #A_Pi = df.filter(regex='A\d+_Pin').sum(axis=1)
    # filter only A1_Pin, A2_Pin
    A_Pi = df.filter(['A1_Pin', 'A2_Pin']).sum(axis=1)
    df['A_Pin'] = A_Pi
    # count the number of columns with the sufix A\d_Pin
    A_count = max(df.filter(regex='A\d+_Pin').count(axis=1))
    print("A_count: ", A_count)
    # print the columns with the sufix A\d_Pin
    print(df.filter(regex='A\d+_Pin').columns)
    # remove the columns with the sufix A\d_Pin
    df.drop(df.filter(regex='A\d+_Pin').columns, axis=1, inplace=True)
    
    wp_B = [292.8, 292.6, 293.9, 297.1]
    print()
    B_Pi = df['B1_Pin']/wp_B[0] + df['B2_Pin']/wp_B[1] + df['B3_Pin']/wp_B[2] + df['B4_Pin']/wp_B[3]
    df['B_Pin'] = B_Pi
    # count the number of columns with the sufix B\d_Pin
    B_count = max(df.filter(regex='B\d+_Pin').count(axis=1))
    print("B_count: ", B_count)
    # print the columns with the sufix B\d_Pin
    print(df.filter(regex='B\d+_Pin').columns)
    # remove the columns with the sufix B\d_Pin
    df.drop(df.filter(regex='B\d+_Pin').columns, axis=1, inplace=True)
   

    #C_Pi = df.filter(regex='C\d+_Pin').sum(axis=1)
    #df['C_Pin'] = C_Pi
    # filter only C1_Pin and C2_Pin
    Wp_C = [302.68, 301.45, 303.37, 301.47]
    C_Pi = df['C1_Pin']/Wp_C[0] + df['C2_Pin']/Wp_C[1] + df['C3_Pin']/Wp_C[2] + df['C4_Pin']/Wp_C[3]
    df['C_Pin'] = C_Pi
    # count the number of columns with the sufix C\d_Pin
    C_count = max(df.filter(regex='C\d+_Pin').count(axis=1))
    print("C_count: ", C_count)
    # print the columns with the sufix C\d_Pin
    print(df.filter(regex='C\d+_Pin').columns)
    # remove the columns with the sufix C\d_Pin
    df.drop(df.filter(regex='C\d+_Pin').columns, axis=1, inplace=True)
   
    #D_Pi = df.filter(regex='D\d+_Pin').sum(axis=1)
    #df['D_Pin'] = D_Pi
    # filter only D1_Pin, D2_Pin
    wp_D = [302.68, 301.45, 303.37, 301.47, 301.47]
    D_Pi = df['D1_Pin']/wp_D[0] + df['D2_Pin']/wp_D[1] + df['D3_Pin']/wp_D[2] + df['D4_Pin']/wp_D[3] + df['D5_Pin']/wp_D[3]
    df['D_Pin'] = D_Pi
    # count the number of columns with the sufix D\d_Pin
    D_count = max(df.filter(regex='D\d+_Pin').count(axis=1))
    print("D_count: ", D_count)
    # print the columns with the sufix D\d_Pin
    print(df.filter(regex='D\d+_Pin').columns)
    # remove the columns with the sufix D\d_Pin
    df.drop(df.filter(regex='D\d+_Pin').columns, axis=1, inplace=True)
    

    E_Pi = df.filter(regex='E\d+_Pin').sum(axis=1)
    df['E_Pin'] = E_Pi
    # count the number of columns with the sufix E\d_Pin
    E_count = max(df.filter(regex='E\d+_Pin').count(axis=1))
    print("E_count: ", E_count)
    # print the columns with the sufix E\d_Pin
    print(df.filter(regex='E\d+_Pin').columns)
    # remove the columns with the sufix E\d_Pin
    df.drop(df.filter(regex='E\d+_Pin').columns, axis=1, inplace=True)
    

    F_Pi = df.filter(regex='F\d+_Pin').sum(axis=1)
    df['F_Pin'] = F_Pi
    # count the number of columns with the sufix F\d_Pin
    F_count = max(df.filter(regex='F\d+_Pin').count(axis=1))
    print("F_count: ", F_count)
    # print the columns with the sufix F\d_Pin
    print(df.filter(regex='F\d+_Pin').columns)
    # remove the columns with the sufix F\d_Pin
    df.drop(df.filter(regex='F\d+_Pin').columns, axis=1, inplace=True)

    G_Pi = df.filter(regex='G\d+_Pin').sum(axis=1)
    df['G_Pin'] = G_Pi
    # count the number of columns with the sufix G\d_Pin
    G_count = max(df.filter(regex='G\d+_Pin').count(axis=1))
    print("G_count: ", G_count)
    # print the columns with the sufix G\d_Pin
    print(df.filter(regex='G\d+_Pin').columns)
    # remove the columns with the sufix G\d_Pin
    df.drop(df.filter(regex='G\d+_Pin').columns, axis=1, inplace=True)
    
    # only for Oslo data set we have H, I, J, K
    # combine H1_Pin, H2_Pin seperately to HFB_Pin
    # combine H3_Pin, H4_Pin, H5_Pin seperately to HFL_Pin
    wp_H = [297.03, 295.10, 292.47, 297.00, 296.71]
    HFB_Pin =  df['H4_Pin']/wp_H[3] + df['H5_Pin']/wp_H[4]
    df['HFB_Pin'] = HFB_Pin
    HFL_Pin = df['H1_Pin']/wp_H[0] + df['H2_Pin']/wp_H[1] + df['H3_Pin']/wp_H[2] 
    df['HFL_Pin'] = HFL_Pin
    # count the number of columns with the sufix HFB_Pin
    H_count = max(df.filter(regex='H\d+_Pin').count(axis=1))
    print("H_count: ", H_count)
    # remove the columns with the sufix H\d_Pin
    df.drop(df.filter(regex='H\d+_Pin').columns, axis=1, inplace=True)
    
   

    #I_Pi = df.filter(regex='I\d+_Pin').sum(axis=1)
    #df['I_Pin'] = I_Pi
    # filter only I1_Pin, I2_Pin
    wp_I = [291.70, 296.00, 293.60, 294.70]
    IFB_Pin = df['I1_Pin']/wp_I[0] + df['I2_Pin']/wp_I[1] 
    df['IFB_Pin'] = IFB_Pin
    IFL_Pin = df['I3_Pin']/wp_I[2] + df['I4_Pin']/wp_I[3]
    df['IFL_Pin'] = IFL_Pin
    # count the number of columns with the sufix I\d_Pin
    I_count = max(df.filter(regex='I\d+_Pin').count(axis=1))
    print("I_count: ", I_count)
    # print the columns with the sufix I\d_Pin
    print(df.filter(regex='I\d+_Pin').columns)
    # remove the columns with the sufix I\d_Pin
    df.drop(df.filter(regex='I\d+_Pin').columns, axis=1, inplace=True)
   
    J_Pi = df.filter(regex='J\d+_Pin').sum(axis=1)
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



    # Find overlapping time range
    common_start = max(weather_df.index.min(), power_df.index.min())
    common_end = min(weather_df.index.max(), power_df.index.max())

    # Filter both DataFrames to the common time range
    weather_df_common = weather_df.loc[common_start:common_end]
    power_df_common = power_df.loc[common_start:common_end]

    # Combine the two DataFrames
    combined_df = weather_df_common.join(power_df_common, how='inner')
    combined_df.reset_index(inplace=True)
    # Ensure 'period_end' is a datetime format and set as index
    combined_df['period_end'] = pd.to_datetime(combined_df['period_end'])
    combined_df.set_index('period_end', inplace=True)


    return combined_df



def plot_compare_and_improvement(df, type1, type2):
    # create a copy of the dataframe
    df = df.copy()
    # remove the rows with missing values for type1 and type2
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()))]
    
    # remove the rows with zeros for type1 and type2
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0))]

    # Create a 'month-year' column for aggregation
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')
    df[type2] = df[type2]
    df[type1] = df[type1]


    # Aggregate data by 'month-year'
    df_grouped = df.groupby('month-year').agg({type1: 'sum', type2: 'sum'}).reset_index()
    print(df_grouped)

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
    ax2.set_ylabel('Deviation (kWh/kWp)', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    #ax1.set_title(f'Comparison and Deviation of {type1} and {type2} by Month-Year')

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'plots/Oslo/{type1}_{type2}_oslo.png')
    plt.show()

    return total_improvement_percentage

# Example usage
# improvement_percentage = plot_compare_and_improvement(your_dataframe, 'manufacturer1_mono', 'manufacturer1_bi')
# print(f"Total improvement: {improvement_percentage:.2f}%")

    

# write a function to compare three types of panels
def plot_compare_three(df, type1, type2, type3):
    # create a copy of the dataframe
    df = df.copy()
    # drop the rows with missing values for type1, type2, and type3
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()) | (df[type3].isnull()))]
    # drop the rows with zeros for type1, type2, and type3
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0) | (df[type3] == 0))]

    # save the dataframe to a csv file incuding index, type1, type2, and type3
    df.to_csv('plots/Oslo/oslo.csv', index=True, columns=[type1, type2, type3])

    # Create a 'month-year' column for aggregation
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')
    df[type2] = df[type2]
    df[type3] = df[type3]
    df[type1] = df[type1]

    # Aggregate data by 'month-year'
    df_grouped = df.groupby('month-year').agg({type1: 'sum', type2: 'sum', type3: 'sum'}).reset_index()

    # Calculate differences
    diff_1_2 = (df_grouped[type2] - df_grouped[type1]) 
    diff_1_3 = (df_grouped[type3] - df_grouped[type1])

    # Calculate average percentage differences
    total_improvement_percentage_12 = (diff_1_2.sum() / df_grouped[type1].sum()) * 100
    total_improvement_percentage_13 = (diff_1_3.sum() / df_grouped[type1].sum()) * 100

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
    ax2.plot(positions, diff_1_2, color='purple', marker='o', label=f'{type2} vs {type1} Deviation (kWh/kWp)')
    ax2.plot(positions, diff_1_3, color='green', marker='o', label=f'{type3} vs {type1} Deviation (kWh/kWp)')

    # Adding month-year labels to x-axis
    ax1.set_xticks(positions)
    ax1.set_xticklabels(df_grouped['month-year'], rotation=90)

    # Setting labels and titles
    ax1.set_xlabel('Month-Year', fontsize=20)
    ax1.set_ylabel('Energy (kWh/kWp)',  fontsize=20)
    ax2.set_ylabel('Deviation (kWh/kWp)',   fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.legend(fontsize=20)

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'plots/Oslo/{type1}_{type2}_{type3}_oslo.png')

  

    # Return percentage differences
    return total_improvement_percentage_12, total_improvement_percentage_13
    

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


import itertools

def plot_daily_power_variation(df, panel_types):
    date_column = 'period_end'
    df['month-year'] = df.index.to_period('M').strftime('%Y-%m')
    month_years = df['month-year'].unique()

    for my in month_years:
        df_my = df[df['month-year'] == my]
        df_melted = df_my.reset_index().melt(id_vars=date_column, value_vars=panel_types, var_name='PanelType', value_name='Power')
        num_days = df_my.index.day.max()

        fig, axs = plt.subplots(1, num_days, figsize=(30, 15), sharey=True)

        for i in range(1, num_days + 1):
            df_day = df_melted[df_melted[date_column].dt.day == i]
            if df_day.empty or 'PanelType' not in df_day.columns:
                continue  # Skip empty dataframes or missing 'PanelType'

            # Additional check for missing values
            if df_day[date_column].isna().any() or df_day['Power'].isna().any():
                continue  # Skip if there are NaNs in essential columns

            sns.lineplot(x=df_day[date_column].dt.hour, y='Power', hue='PanelType', data=df_day, ax=axs[i-1])
            axs[i-1].set_title(f'Day {i}')
            axs[i-1].set_xlabel('Hour')

        fig.suptitle(f'Daily Power Variation for {my}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plots/Oslo/daily_power_variation_{my}.png')
        plt.close(fig)



import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_operating_hours(df, panel_type, power_threshold=0):
    """
    Plots the operating hours of a given panel type.

    :param df: Pandas DataFrame containing the data
    :param panel_type: String, the name of the column representing the panel type to analyze
    :param power_threshold: Numeric, the threshold above which the panel is considered to be operating
    """
    # Ensure that the date column is in datetime format
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Filter the DataFrame for the given panel type
    panel_data = df[['period_end', panel_type]]

    # Determine if the panel is operating (power > threshold)
    panel_data['Operating'] = panel_data[panel_type] > power_threshold

    # Group by date and count operating hours
    operating_hours = panel_data.groupby(panel_data['period_end'].dt.date)['Operating'].sum()

    # Plot
    plt.figure(figsize=(12, 6))
    operating_hours.plot(kind='bar')
    plt.title(f'Operating Hours of {panel_type} Panel')
    plt.xlabel('Date')
    plt.ylabel('Operating Hours')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
    plt.savefig(f'plots/Oslo/average_operating_hours_{panel_type}.png')
    plt.close()

    # Convert monthly operating times to DataFrame
    operating_times_df = pd.DataFrame(monthly_operating_times)

    return avg_operating_hours_series, operating_times_df

# Example usage
# operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_mono', power_threshold=0.1)
# print(operating_hours_series)
# print(operating_times_df)




def main():
   
    # Read th power data file
    folder_path = "../../../data/iroshanij/greendata/"
    power_df = pd.read_csv(folder_path + "final_with_panels_Oslo.csv")
    weather_df = pd.read_csv(folder_path + "weather_oslo.csv")

    # describe the data
    print(power_df.head())
    print(power_df.describe())
    print(power_df.columns)

    # get the columns with missing values
    missing_cols = power_df.columns[power_df.isnull().any()].tolist()
    #print(missing_cols)

    # count the number of missing values for each column in each month-year
    # set the index to DATETIME
    #power_df['DATETIME'] = pd.to_datetime(power_df['DATETIME'])
    #power_df.set_index('DATETIME', inplace=True)
   
    # replace the missing values with previous values
    #power_df.fillna(method='ffill', inplace=True)
   
    # count the number of missing values for each column in each month-year
    #power_df.groupby([power_df.index.year, power_df.index.month])[missing_cols].count()
    # plot the missing values for each column in each month-year
    #for col in missing_cols:
    #    plt.figure()
    #    power_df.groupby([power_df.index.year, power_df.index.month])[col].count().plot()
        # change the x label to month-year format
    #    plt.xlabel('month-year')
    #    plt.ylabel('number of missing values')
    #    plt.title('Missing values for '+col)
    #    plt.savefig('plots/Oslo/'+col+'.png')


    # select the columns where missing values are greater that 10% of the total number of rows
    #missing_cols = power_df.columns[power_df.isnull().mean() > 0.10].tolist()
    #['F2_Pin', 'F3_Pin', 'F5_Pin', 'E9_Pin', 'F1_Pin', 'F4_Pin', 'K9_Pin', 'J4_Pin']
    #missing_cols = ['F2_Pin', 'F3_Pin', 'F5_Pin', 'E9_Pin', 'F1_Pin', 'F4_Pin', 'K9_Pin', 'J4_Pin']
    #print(missing_cols)

   
    
    # remove the columns with missing values
    #missing_cols = ['F2_Pin', 'F3_Pin', 'F5_Pin', 'E9_Pin', 'F1_Pin', 'F4_Pin', 'K9_Pin', 'J4_Pin']
    #power_df.drop(missing_cols, axis=1, inplace=True)
    # remove the missing column if it has more than 20% missing values out of the total number of rows
    for col in missing_cols:
        if power_df[col].isnull().sum() > 0.25*len(power_df[col]):
            power_df.drop(col, axis=1, inplace=True)
            print("Dropped column: ", col)
            #print("column: ", col, " has more than 25% missing values")
    
    avg_power_df = calculate_5min_average_power(power_df)

    # aggregate the columns with the same suffix and store the result in a new column
    avg_power_df = calculate_panel_type_average_power(avg_power_df)

    # rename the column names to match the mapping of panel types
    # Oslo
    # solitek_mono = ['D_Pin']
    # manufacturer1_bi = ['C_Pin']
    # manufacturer1_inn1 = ['A_Pin']
    # manufacturer1_inn2 = ['H_Pin']
    # manufacturer2_inn1 = ['I_Pin']
    # manufacturer2_inn2 = ['J_Pin']
    # manufacturer2_bi-facial = ['B_Pin']

    avg_power_df.rename(columns={'A_Pin': 'manufacturer1_inn1', 
                                 'B_Pin': 'manufacturer2_bi-facial', 
                                 'C_Pin': 'manufacturer1_bi-facial',
                                 'D_Pin': 'manufacturer1_mono-facial',
                                 'HFB_Pin': 'manufacturer1_inn2_HFB',
                                 'HFL_Pin': 'manufacturer1_inn2_HFL', 
                                 'IFB_Pin': 'manufacturer2_inn2_HFB', 
                                'IFL_Pin': 'manufacturer2_inn2_HFL',
                                 'J_Pin': 'manufacturer2_inn1',
                                 'E_Pin': 'CIGS_inn1',
                                 'G_Pin': 'CIGS_inn2',
                                 'K_Pin': 'CIGS_bi-facial'}, inplace=True)
    # add a new column with the location name and set it to Oslo
    avg_power_df['location'] = 'Oslo'

    # drop the columns with siffix CIGS_, manufacturer2_inn1, manufacturer1_inn1
    avg_power_df.drop(['CIGS_inn1', 'CIGS_inn2', 'CIGS_bi-facial', 'manufacturer2_inn1', 'manufacturer1_inn1'], axis=1, inplace=True)

    # calculate averages of power for each panel type in Oslo
    #soltek_mono = avg_power_df['manufacturer1_mono']/5
    #soltek_bi = avg_power_df['manufacturer1_bi']/4
    #soltek_inn1 = avg_power_df['manufacturer1_inn1']/4
    #soltek_inn2 = avg_power_df['manufacturer1_inn2']/5
    #manufacturer2_inn1 = avg_power_df['manufacturer2_inn1']/4
    #manufacturer2_inn2 = avg_power_df['manufacturer2_inn2']/3
    #manufacturer2_bi-facial = avg_power_df['manufacturer2_bi-facial']/4

    #     A_count:  4
    # Index(['A3_Pin', 'A2_Pin', 'A1_Pin', 'A4_Pin'], dtype='object')
    # B_count:  4
    # Index(['B2_Pin', 'B1_Pin', 'B3_Pin', 'B4_Pin'], dtype='object')
    # C_count:  4
    # Index(['C4_Pin', 'C1_Pin', 'C2_Pin', 'C3_Pin'], dtype='object')
    # D_count:  5
    # Index(['D5_Pin', 'D3_Pin', 'D1_Pin', 'D4_Pin', 'D2_Pin'], dtype='object')
    # E_count:  8
    # Index(['E2_Pin', 'E7_Pin', 'E4_Pin', 'E3_Pin', 'E6_Pin', 'E5_Pin', 'E1_Pin',
    #        'E8_Pin'],
    #       dtype='object')
    # F_count:  0
    # Index([], dtype='object')
    # G_count:  10
    # Index(['G7_Pin', 'G10_Pin', 'G6_Pin', 'G1_Pin', 'G8_Pin', 'G3_Pin', 'G9_Pin',
    #        'G2_Pin', 'G5_Pin', 'G4_Pin'],
    #       dtype='object')
    # H_count:  5
    # Index(['H4_Pin', 'H2_Pin', 'H1_Pin', 'H3_Pin', 'H5_Pin'], dtype='object')
    # I_count:  4
    # Index(['I2_Pin', 'I3_Pin', 'I4_Pin', 'I1_Pin'], dtype='object')
    # J_count:  3
    # Index(['J2_Pin', 'J3_Pin', 'J1_Pin'], dtype='object')
    # K_count:  9
    # Index(['K1_Pin', 'K6_Pin', 'K4_Pin', 'K7_Pin', 'K3_Pin', 'K8_Pin', 'K2_Pin',
    #        'K5_Pin', 'K10_Pin'],

    avg_power_df['manufacturer1_mono-facial'] = avg_power_df['manufacturer1_mono-facial']/5
    avg_power_df['manufacturer1_bi-facial'] = avg_power_df['manufacturer1_bi-facial']/4
    avg_power_df['manufacturer1_inn2_HFL'] = avg_power_df['manufacturer1_inn2_HFL']/3
    avg_power_df['manufacturer1_inn2_HFB'] = avg_power_df['manufacturer1_inn2_HFB']/2
    avg_power_df['manufacturer2_inn2_HFL'] = avg_power_df['manufacturer2_inn2_HFL']/2
    avg_power_df['manufacturer2_inn2_HFB'] = avg_power_df['manufacturer2_inn2_HFB']/2
    avg_power_df['manufacturer2_bi-facial'] = avg_power_df['manufacturer2_bi-facial']/4
   

    combined_df = combine_weather_power(weather_df, avg_power_df)
    # remove the rows with missing values
    #combined_df = combined_df.dropna()
    # remove the rows with zeros
    #combined_df = combined_df.loc[~((combined_df['manufacturer1_mono'] == 0) | (combined_df['manufacturer1_bi'] == 0)| (combined_df['manufacturer1_inn2_HFB'] == 0) | (combined_df['manufacturer1_inn2_HFL'] == 0) | (combined_df['manufacturer2_inn2'] == 0) | (combined_df['manufacturer2_bi-facial'] == 0))]
    #combined_df = avg_power_df
    print(combined_df.head())
    # reset the index
    #combined_df.reset_index(inplace=True)
    #plot_operating_hours(combined_df, 'manufacturer1_mono')
    
    #panel_types = ['manufacturer1_bi', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL']
    #plot_daily_power_variation(combined_df, panel_types)


    #print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL', 'manufacturer2_inn1', 'manufacturer2_inn2', 'manufacturer2_bi-facial']].describe())

    # calculate the power yield for each panel type in Oslo
    # avg_power is measured in Wh
    # ghi is measured in Wh/m2
    # area is measured in m2 = 1.890702
    # power_yield is measured in Wh
    nominal_plant_output = (combined_df['ghi'])*(5/60)/1000
    combined_df['manufacturer1_mono-facial_pr'] = (combined_df['manufacturer1_mono-facial'])/nominal_plant_output
    combined_df['manufacturer1_bi-facial_pr'] = (combined_df['manufacturer1_bi-facial'])/nominal_plant_output
    combined_df['manufacturer1_inn2_HFB_pr'] = (combined_df['manufacturer1_inn2_HFB'])/nominal_plant_output
    combined_df['manufacturer1_inn2_HFL_pr'] = (combined_df['manufacturer1_inn2_HFL'])/nominal_plant_output
    combined_df['manufacturer2_inn2_HFB_pr'] = (combined_df['manufacturer2_inn2_HFB'])/nominal_plant_output
    combined_df['manufacturer2_inn2_HFL_pr'] = (combined_df['manufacturer2_inn2_HFL'])/nominal_plant_output
    combined_df['manufacturer2_bi-facial_pr'] = (combined_df['manufacturer2_bi-facial'])/nominal_plant_output

    #print(combined_df[['manufacturer1_mono_pr', 'manufacturer1_bi_pr', 'manufacturer1_inn1_pr', 'manufacturer1_inn2_FB_pr', 'manufacturer1_inn2_FL_pr', 'manufacturer2_inn1_pr', 'manufacturer2_inn2_FL_pr', 'manufacturer2_inn2_FB_pr','manufacturer2_bi-facial_pr']].describe())

    solar_radiation = combined_df['ghi']*1.89072
    combined_df['manufacturer1_mono-facial_eff'] = combined_df['manufacturer1_mono-facial']/solar_radiation
    combined_df['manufacturer1_bi-facial_eff'] = combined_df['manufacturer1_bi-facial']/solar_radiation
    combined_df['manufacturer1_inn2_HFB_eff'] = combined_df['manufacturer1_inn2_HFB']/solar_radiation
    combined_df['manufacturer1_inn2_HFL_eff'] = combined_df['manufacturer1_inn2_HFL']/solar_radiation
    combined_df['manufacturer2_inn2_HFL_eff'] = combined_df['manufacturer2_inn2_HFL']/solar_radiation
    combined_df['manufacturer2_inn2_HFB_eff'] = combined_df['manufacturer2_inn2_HFB']/solar_radiation
    combined_df['manufacturer2_bi-facial_eff'] = combined_df['manufacturer2_bi-facial']/solar_radiation

    # delete the columns ['F_Pin', 'snow_soiling_rooftop']
    combined_df.drop(['F_Pin'], axis=1, inplace=True) 

    # describe the data
    print(combined_df.describe())
    print(combined_df.columns)
    # find the missing values for each column
    print(combined_df.isnull().sum())

    
    # find the datetime ranges the missing values and give the min max values of continous missing values
    #print(combined_df['manufacturer1_mono'].isnull().groupby(combined_df['manufacturer1_mono'].notnull().cumsum()).sum())
    
    # remove anomulous data for each panel type
    #panel_types = ['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL']
    #std_devs = [3, 3, 3, 3, 3]
    #for panel_type in panel_types:
    #    combined_df = remove_anomalies(combined_df, panel_type, std_devs=std_devs[panel_types.index(panel_type)])

    # remove rows with zeros, negetive values, and missing values
    #print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL', 'manufacturer2_inn1', 'manufacturer2_inn2', 'manufacturer2_bi-facial']].describe())
    
    improvement = plot_compare_and_improvement(combined_df, 'manufacturer1_mono-facial', 'manufacturer1_bi-facial')
    print("Improvement in bi in Oslo: ", improvement)
    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer1_bi-facial', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL')
    print("Improvement in inn2_HFB in Oslo: ", diff1)
    print("Improvement in inn2_HFL in Oslo: ", diff2)

    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer2_bi-facial', 'manufacturer2_inn2_HFB', 'manufacturer2_inn2_HFL')
    print("Improvement in inn2_HFB in Oslo: ", diff1)
    print("Improvement in inn2_HFL in Oslo: ", diff2)


    #print(combined_df.columns)
    # reset the index
    #combined_df.reset_index(inplace=True)
    #operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_mono', power_threshold=0.1)
    # save operating hours and operating times to csv files
    #operating_hours_series.to_csv(folder_path + "operating_hours_oslo.csv")
    #operating_times_df.to_csv(folder_path + "operating_times_oslo.csv")

    #print(operating_hours_series)
    #print(operating_times_df)
    #operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_bi', power_threshold=0.1)
    #print(operating_hours_series)
    #print(operating_times_df)
    
    #plot_ghi(combined_df)
    # save the combined data to csv file
    combined_df.to_csv(folder_path + "combined_data_oslo.csv")
    

if __name__ == "__main__":
    main()