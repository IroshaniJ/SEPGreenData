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
    # Power columns for Oslo
    print(df.columns)
    # consider all the columns with the sufix _Pin
    power_columns = df.filter(regex='_Pin').columns
    power_columns = [col for col in power_columns if col in df.columns]

    if not power_columns:
        raise ValueError("No power columns found in the DataFrame.")

    # Group by 5-minute intervals and calculate the average
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
    #df['A_Pin'] = A_Pi
    wp_A = [301.82, 302.52, 302.94, 302.70, 302.40, 300.80, 302.88, 300.93, 301.32, 304.87]
    df['A_Pin'] = df['A1_Pin']/wp_A[0] + df['A2_Pin']/wp_A[1] + df['A3_Pin']/wp_A[2] + df['A4_Pin']/wp_A[3] + df['A5_Pin']/wp_A[4] + df['A6_Pin']/wp_A[5] + df['A7_Pin']/wp_A[6] +df['A8_Pin']/wp_A[7] + df['A9_Pin']/wp_A[8] + df['A10_Pin']/wp_A[9]

    # count the number of columns with the sufix A\d_Pin
    A_count = max(df.filter(regex='A\d+_Pin').count(axis=1))
    print("A_count: ", A_count)
    # print the columns with the sufix A\d_Pin
    print(df.filter(regex='A\d+_Pin').columns)
    # remove the columns with the sufix A\d_Pin
    df.drop(df.filter(regex='A\d+_Pin').columns, axis=1, inplace=True)
    
    #B_Pi = df.filter(regex='B\d+_Pin').sum(axis=1)
    #df['B_Pin'] = B_Pi
    # filter only B1_Pin and B2_Pin
    wp_B = [302.52, 301.82, 302.88, 300.93, 301.32, 304.87, 300.80, 302.40, 302.70, 301.94]
    df['B_Pin'] = df['B1_Pin']/wp_B[0] + df['B2_Pin']/wp_B[1] + df['B3_Pin']/wp_B[2] + df['B4_Pin']/wp_B[3] + df['B5_Pin']/wp_B[4] + df['B6_Pin']/wp_B[5] + df['B7_Pin']/wp_B[6] + df['B8_Pin']/wp_B[7] + df['B9_Pin']/wp_B[8] + df['B10_Pin']/wp_B[9]
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
    df['C_Pin'] = df.filter(regex='C\d+_Pin').sum(axis=1)
    # count the number of columns with the sufix C\d_Pin
    C_count = max(df.filter(regex='C\d+_Pin').count(axis=1))
    print("C_count: ", C_count)
    # print the columns with the sufix C\d_Pin
    print(df.filter(regex='C\d+_Pin').columns)
    # remove the columns with the sufix C\d_Pin
    df.drop(df.filter(regex='C\d+_Pin').columns, axis=1, inplace=True)
   
    #HFB: D4, D5, D7, D8, D9.
    #HFL: D1, D2, D3, D6, D10.
    wp_D = [290.95, 292.87, 293.15, 297.52, 295.22, 294.01, 295.67, 297.15, 299.05, 296.35]
    DFB_Pi = df['D4_Pin']/wp_D[3] + df['D5_Pin']/wp_D[4] + df['D7_Pin']/wp_D[6] + df['D8_Pin']/wp_D[7] + df['D9_Pin']/wp_D[8]
    df['DFB_Pin'] = DFB_Pi
    DFL_Pi = df['D1_Pin']/wp_D[0] + df['D2_Pin']/wp_D[1] + df['D3_Pin']/wp_D[2] + df['D6_Pin']/wp_D[5] + df['D10_Pin']/wp_D[9]
    df['DFL_Pin'] = DFL_Pi
    # count the number of columns with the sufix D\d_Pin
    D_count = max(df.filter(regex='D\d+_Pin').count(axis=1))
    print("D_count: ", D_count)
    # remove the columns with the sufix D\d_Pin
    df.drop(df.filter(regex='D\d+_Pin').columns, axis=1, inplace=True)
 
    

    wp_E = [293.4, 292.4, 293.9, 292.6, 294.0, 293.2, 293.1, 292.0]
    # FB = E3, E2, E4, E1
    # FL = E6, E5, E7, E8
    EFB_Pi = df['E1_Pin']/wp_E[0] + df['E2_Pin']/wp_E[1] + df['E3_Pin']/wp_E[2] + df['E4_Pin']/wp_E[3] 
    EFL_Pi =  df['E5_Pin']/wp_E[4] + df['E6_Pin']/wp_E[5] + df['E7_Pin']/wp_E[6] 
    df['EFB_Pin'] = EFB_Pi
    df['EFL_Pin'] = EFL_Pi

    # count the number of columns with the sufix E\d_Pin
    E_count = max(df.filter(regex='E\d+_Pin').count(axis=1))
    print("E_count: ", E_count)
    # print the columns with the sufix E\d_Pin
    print(df.filter(regex='E\d+_Pin').columns)
    # remove the columns with the sufix E\d_Pin
    df.drop(df.filter(regex='E\d+_Pin').columns, axis=1, inplace=True)
    

    wp_F = [301.8, 302.2, 301.1, 302.5, 302.1, 300.4]
    df['F_Pin'] = df['F1_Pin']/wp_F[0] + df['F2_Pin']/wp_F[1] + df['F3_Pin']/wp_F[2] + df['F4_Pin']/wp_F[3] + df['F5_Pin']/wp_F[4] + df['F6_Pin']/wp_F[5]
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
    # remove the rows with missing values
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0))]
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()))]

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
    ax1.set_xlabel('Month-Year')
    ax1.set_ylabel('kWh/kWp')
    ax2.set_ylabel('Deviation (kWh/KWp)')
    ax1.tick_params(axis='y', labelsize=20)

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(f'plots/Tunisia/{type1}_{type2}_tunisia.png')
    
    return total_improvement_percentage

# Assume df is your DataFrame
def plot_compare_three(df, type1, type2, type3):
    # create a copy of the dataframe
    df = df.copy()
     # drop the columns with missing values and zero values only for the colunms 'manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL'
    df = df.loc[~((df[type1] == 0) | (df[type2] == 0) |  (df[type3] == 0))]
    # remove the rows with missing values
    df = df.loc[~((df[type1].isnull()) | (df[type2].isnull()) | (df[type3].isnull()))]

   
    # save the dataframe to a csv file incuding index, type1, type2, and type3
    df.to_csv('plots/Tunisia/tunisia.csv', index=True, columns=[type1, type2, type3])

    # Create a 'month-year' column for aggregation
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
    ax1.bar(positions - bar_width, df_grouped[type1], bar_width, label=type1, color = 'orange')
    ax1.bar(positions, df_grouped[type2], bar_width, label=type2, color = 'purple')
    ax1.bar(positions + bar_width, df_grouped[type3], bar_width, label=type3, color = 'green')

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

    plt.savefig(f'plots/Tunisia/{type1}_{type2}_{type3}_tunisia.png')

    # set period_end as index
    
    
    

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


def plot_daily_power_variation(df, columns):
    date_column = 'period_end'
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
        plt.savefig(f'plots/Tunisia/daily_power_variation_{my}.png')
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
    plt.savefig(f'plots/Tunisia/average_operating_hours_{panel_type}.png')
    plt.close()

    # Convert monthly operating times to DataFrame
    operating_times_df = pd.DataFrame(monthly_operating_times)

    # save the monthly operating times to csv file
    operating_times_df.to_csv(f'plots/Tunisia/monthly_operating_times_{panel_type}.csv')

    return avg_operating_hours_series, operating_times_df

# Example usage
# operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_mono', power_threshold=0.1)
# print(operating_hours_series)
# print(operating_times_df)

def main():
   
    # Read th power data file
    folder_path = "../../../data/iroshanij/greendata/"
    power_df = pd.read_csv(folder_path + "final_with_panels_Tunisia.csv")
    weather_df = pd.read_csv(folder_path + "weather_tunisia.csv")

    # describe the data
    print(power_df.head())
    print(power_df.describe())
    print(power_df.columns)

    

    # get the columns with missing values
    missing_cols = power_df.columns[power_df.isnull().any()].tolist()
    print(missing_cols)

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
    #    plt.savefig('plots/Tunisia/'+col+'.png')


    # select the columns where missing values are greater that 10% of the total number of rows
    #missing_cols = power_df.columns[power_df.isnull().mean() > 0.10].tolist()
    #['F2_Pin', 'F3_Pin', 'F5_Pin', 'E9_Pin', 'F1_Pin', 'F4_Pin', 'K9_Pin', 'J4_Pin']
    #missing_cols = ['F2_Pin', 'F3_Pin', 'F5_Pin', 'E9_Pin', 'F1_Pin', 'F4_Pin', 'K9_Pin', 'J4_Pin']
    #print(missing_cols)
    # ['E8_Pin']


    # remove the columns with missing values, dropped only E8_Pin
    # remove the missing column if it has more than 20% missing values out of the total number of rows
    for col in missing_cols:
        if power_df[col].isnull().sum() > 0.25*len(power_df[col]):
            #power_df.drop(col, axis=1, inplace=True)
            #print("Dropped column: ", col)
            print("column: ", col, " has more than 25% missing values")

    
    
    avg_power_df = calculate_5min_average_power(power_df)

    # # aggregate the columns with the same suffix and store the result in a new column
    avg_power_df = calculate_panel_type_average_power(avg_power_df)

    
    # Tunisia
    # solitek_mono = ['A_Pin']
    # manufacturer1_bi = ['B_Pin']
    # manufacturer1_inn1 = ['C_Pin']
    # manufacturer1_inn2 = ['D_Pin']
    # manufacturer2_inn1 = ['G_Pin']
    # manufacturer2_inn2 = ['E_Pin']
    # manufacturer2_bi-facial = ['F_Pin']

    avg_power_df.rename(columns={'A_Pin': 'manufacturer1_mono-facial', 
                                'B_Pin': 'manufacturer1_bi-facial', 
                                'C_Pin': 'manufacturer1_inn1',
                                'DFB_Pin': 'manufacturer1_inn2_HFB',
                                'DFL_Pin': 'manufacturer1_inn2_HFL', 
                                'EFL_Pin': 'manufacturer2_inn2_HFL', 
                                'EFB_Pin': 'manufacturer2_inn2_HFB',
                                'F_Pin': 'manufacturer2_bi-facial', 
                                'G_Pin': 'manufacturer2_inn1'}, inplace=True)

    # # # # add a new column with the location name and set it to Tunisia
    avg_power_df['location'] = 'Tozeur'

    # drop the columns with pbi-facialix 'manufacturer1_inn1' and 'manufacturer2_inn1'
    avg_power_df.drop(avg_power_df.filter(regex='manufacturer1_inn1').columns, axis=1, inplace=True)
    avg_power_df.drop(avg_power_df.filter(regex='manufacturer2_inn1').columns, axis=1, inplace=True)

    # calculate averages of power for each panel type in Tunisia
    # soltek_mono = avg_power_df['manufacturer1_mono']/10
    # soltek_bi = avg_power_df['manufacturer1_bi']/10
    # soltek_inn1 = avg_power_df['manufacturer1_inn1']/9
    # soltek_inn2 = avg_power_df['manufacturer1_inn2']/10
    # manufacturer2_inn1 = avg_power_df['manufacturer2_inn1']/6
    # manufacturer2_inn2 = avg_power_df['manufacturer2_inn2']/7
    # manufacturer2_bi-facial = avg_power_df['manufacturer2_bi-facial']/6

#     A_count:  10
# Index(['A6_Pin', 'A4_Pin', 'A1_Pin', 'A5_Pin', 'A8_Pin', 'A2_Pin', 'A7_Pin',
#        'A3_Pin', 'A9_Pin', 'A10_Pin'],
#       dtype='object')
# B_count:  10
# Index(['B4_Pin', 'B8_Pin', 'B9_Pin', 'B3_Pin', 'B7_Pin', 'B10_Pin', 'B6_Pin',
#        'B5_Pin', 'B2_Pin', 'B1_Pin'],
#       dtype='object')
# C_count:  9
# Index(['C1_Pin', 'C7_Pin', 'C9_Pin', 'C3_Pin', 'C4_Pin', 'C2_Pin', 'C6_Pin',
#        'C10_Pin', 'C5_Pin'],
#       dtype='object')
# D_count:  10
# Index(['D5_Pin', 'D10_Pin', 'D3_Pin', 'D6_Pin', 'D4_Pin', 'D7_Pin', 'D2_Pin',
#        'D9_Pin', 'D1_Pin', 'D8_Pin'],
#       dtype='object')
# E_count:  7
# Index(['E6_Pin', 'E2_Pin', 'E5_Pin', 'E1_Pin', 'E4_Pin', 'E3_Pin', 'E7_Pin'], dtype='object')
# F_count:  6
# Index(['F1_Pin', 'F6_Pin', 'F4_Pin', 'F2_Pin', 'F3_Pin', 'F5_Pin'], dtype='object')
# G_count:  6
# Index(['G5_Pin', 'G6_Pin', 'G1_Pin', 'G4_Pin', 'G2_Pin', 'G3_Pin'], dtype='object')

    avg_power_df['manufacturer1_mono-facial'] = avg_power_df['manufacturer1_mono-facial']/10
    avg_power_df['manufacturer1_bi-facial'] = avg_power_df['manufacturer1_bi-facial']/10
    avg_power_df['manufacturer1_inn2_HFB'] = avg_power_df['manufacturer1_inn2_HFB']/5
    avg_power_df['manufacturer1_inn2_HFL'] = avg_power_df['manufacturer1_inn2_HFL']/5
    avg_power_df['manufacturer2_inn2_HFB'] = avg_power_df['manufacturer2_inn2_HFB']/4
    avg_power_df['manufacturer2_inn2_HFL'] = avg_power_df['manufacturer2_inn2_HFL']/3
    avg_power_df['manufacturer2_bi-facial'] = avg_power_df['manufacturer2_bi-facial']/6
    
    combined_df = combine_weather_power(weather_df, avg_power_df)
     # remove the rows with missing values
    #combined_df = combined_df.dropna()
    # remove the rows with zeros
    #combined_df = combined_df.loc[~((combined_df['manufacturer1_mono'] == 0) | (combined_df['manufacturer1_bi'] == 0)| (combined_df['manufacturer1_inn2_HFB'] == 0) | (combined_df['manufacturer1_inn2_HFL'] == 0) | (combined_df['manufacturer2_inn2'] == 0) | (combined_df['manufacturer2_bi-facial'] == 0))]
    #combined_df = avg_power_df
    
    # """ print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL', 'manufacturer2_inn1', 'manufacturer2_inn2', 'manufacturer2_bi-facial']].describe())

   

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

    # describe the data
    #print(combined_df.describe())
    #print(combined_df.columns)
    # find the missing values for each column
    #print(combined_df.isnull().sum())
    #print(combined_df.head())
    # describe the given columns
    
    # remove anomulous data for each panel type
    #panel_types = ['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL']
    #std_devs = [3, 3, 3, 3, 3]
    #for panel_type in panel_types:
     #   combined_df = remove_anomalies(combined_df, panel_type, std_devs=std_devs[panel_types.index(panel_type)])

    # remove rows with zeros, negetive values, and missing values
    #print(combined_df[['manufacturer1_mono', 'manufacturer1_bi', 'manufacturer1_inn1', 'manufacturer1_inn2_FB', 'manufacturer1_inn2_FL', 'manufacturer2_inn1', 'manufacturer2_inn2_FL', 'manufacturer2_inn2_FB', 'manufacturer2_bi-facial']].describe())
    # reset the index
    #combined_df.reset_index(inplace=True)
    #plot_operating_hours(combined_df, 'manufacturer1_mono')
    improvement = plot_compare_and_improvement(combined_df, 'manufacturer1_mono-facial', 'manufacturer1_bi-facial')
    print("Improvement in bi in Tunisia: ", improvement)
    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer1_bi-facial', 'manufacturer1_inn2_HFB', 'manufacturer1_inn2_HFL')
    print("Improvement in inn2_HFB in Tunisia: ", diff1)
    print("Improvement in inn2_HFL in Tunisia: ", diff2)
    diff1, diff2 = plot_compare_three(combined_df, 'manufacturer2_bi-facial', 'manufacturer2_inn2_HFB', 'manufacturer2_inn2_HFL')
    print("Improvement in inn2_HFB in Tunisia: ", diff1)
    print("Improvement in inn2_HFL in Tunisia: ", diff2)

    #improvement = plot_compare_and_improvement(combined_df, 'manufacturer1_bi', 'manufacturer1_inn1')
    #print("Improvement in inn1 in Tunisia: ", improvement)

    # reset the index
    combined_df.reset_index(inplace=True)
    operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_mono-facial', power_threshold=0.1)
     # save operating hours and operating times to csv files
    operating_hours_series.to_csv(folder_path + "operating_hours_tunisia.csv")
    operating_times_df.to_csv(folder_path + "operating_times_tunisia.csv")
    #print(operating_hours_series)
    #print(operating_times_df)
    #operating_hours_series, operating_times_df = plot_monthly_average_operating_hours(combined_df, 'manufacturer1_bi', power_threshold=0.1)
    #print(operating_hours_series)
    #print(operating_times_df)

    # save the combined data to csv file
    combined_df.to_csv(folder_path + "combined_data_tunisia.csv")
    

if __name__ == "__main__":
    main()