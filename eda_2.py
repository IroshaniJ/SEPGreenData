
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


# Load the dataset
file_path = '../../../data/iroshanij/greendata/combined_data.csv'
data = pd.read_csv(file_path)

# Dropping the 'Unnamed: 0' column as it seems to be an index column without context
data_cleaned = data.drop(columns=['Unnamed: 0'])

# Convert 'period_end' to datetime
data_cleaned['period_end'] = pd.to_datetime(data_cleaned['period_end'])

# Descriptive statistics for the dataset
descriptive_stats = data_cleaned.describe(include='all')

# Checking for missing values
missing_values = data_cleaned.isnull().sum()
# replace NAN values with 0
data_cleaned = data_cleaned.fillna(0)

# plot cpr for each solar panel type
def plot_cpr(df):
    columns = ['soliteck_mono_pr', 'soliteck_bi_pr', 'soliteck_inn1_pr', 'soliteck_inn2_pr', 'apolon_inn1_pr', 'apolon_inn2_pr', 'apolon_ref_pr']
    
    # consider only the common data points for two locations
    
    for c in columns:
        # plot yearly cpr for each solar panel type for locations Oslo and Tunisa
        plt.clf()
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        # create  a new column for year
        df['year'] = df['period_end'].dt.year
        sns.lineplot(x="year", y=c, hue="location", data=df)
        # rotate x-axis labels
        plt.xticks(rotation=90)
        # reduce the font size of x-axis labels, y-axis labels, and legend
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.show()
        plt.savefig('plots/pr/y_cpr_'+str(c)+'.png')

         # plot monthly cpr
        # clear the previous plot
            
        plt.clf()
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        # create  a new column for month
        df['month'] = df['period_end'].dt.month
        sns.lineplot(x="month", y=c, hue="location", data=df)
        # rotate x-axis labels
        plt.xticks(rotation=90)
        # reduce the font size of x-axis labels, y-axis labels, and legend
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.show()
        plt.savefig('plots/pr/m_cpr_'+str(c)+'.png')

def analyse_missing_data(df):

    # do this for all the solar panel types
    columns = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2', 'apolon_inn1', 'apolon_inn2', 'apolon_ref']
    columns_pr = ['soliteck_mono_pr', 'soliteck_bi_pr', 'soliteck_inn1_pr', 'soliteck_inn2_pr', 'apolon_inn1_pr', 'apolon_inn2_pr', 'apolon_ref_pr']
    # Convert 'period_end' to datetime if it's not already
    df['period_end'] = pd.to_datetime(df['period_end'])

    # Set 'period_end' as the DataFrame index
    df = df.set_index('period_end')
    
    # remove the rows with missing values
    #df = df.dropna()
    # also remove NaN values and infinity values
    #df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # remove the rows if there is no common period_end for both locations
    #df = df.groupby('period_end').filter(lambda x: len(x) == 3)

    # give the count of rows for each location
    print(df.groupby('location').count()) 

    # give the count of rows for each month in each location
    print(df.groupby(['location', df.index.month]).count())

    # print the maximum and minimum period_end for each column at each location
    #print(df.groupby(['location']).agg({'period_end': ['min', 'max']}))


    # reset the index
    df = df.reset_index()
    return df

def plot_pr_power_air_temp(df):
    columns = ['soliteck_mono_pr', 'soliteck_bi_pr', 'soliteck_inn1_pr', 'soliteck_inn2_pr', 'apolon_inn1_pr', 'apolon_inn2_pr', 'apolon_ref_pr']
    
    # Assuming 'air_temp_column' is the name of your column that contains air temperature values
    air_temp_column = 'air_temp'  # Replace with your actual column name for air temperature if different

    # Set the style outside the loop
    sns.set(style="whitegrid")

    # Create the 'month' column if it's not already there
    if 'month' not in df.columns:
        df['month'] = df['period_end'].dt.month

    # Loop through each column intended for comparison with air temperature
    for c in columns:  # 'columns' should be a list of your solar panel output column names
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(11.7, 8.27))

        # Plot the solar panel output on ax1
        sns.lineplot(data=df, x='month', y=c, hue="location", ax=ax1, legend=True)  # 'c' is the column name
        ax1.set_ylabel(c, fontsize=10)  # Set the label for the y-axis
        ax1.tick_params(axis='y', labelsize=10)  # Set the font size for the y-axis ticks

        # Create the second y-axis for GHI with twinx
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x='month', y=air_temp_column, hue="location", ax=ax2, legend=True, color='r', linestyle='--')
        ax2.set_ylabel(air_temp_column, fontsize=10)  # Set the label for the y-axis
        ax2.tick_params(axis='y', labelcolor='r', labelsize=10)  # Set the font size for the y-axis ticks

        # To make the legend work for both axes, we need to manually create the legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # avoid overlapping the legend
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=2)

        # Save the plot before showing
        plt.savefig(f'plots/pr_power_vs_air_temp/air_temp_vs_{c}.png')  
    
def plot_pr_power_ghi(df):
    columns = ['soliteck_mono_pr', 'soliteck_bi_pr', 'soliteck_inn1_pr', 'soliteck_inn2_pr', 'apolon_inn1_pr', 'apolon_inn2_pr', 'apolon_ref_pr']
    
    ghi_column = 'ghi'  # Replace with your actual column name for air temperature if different

    # Set the style outside the loop
    sns.set(style="whitegrid")

    # Create the 'month' column if it's not already there
    if 'month' not in df.columns:
        df['month'] = df['period_end'].dt.month

    # Loop through each column intended for comparison with air temperature
    for c in columns:  # 'columns' should be a list of your solar panel output column names
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(11.7, 8.27))

        # Plot the solar panel output on ax1
        sns.lineplot(data=df, x='month', y=c, hue="location", ax=ax1, legend=True)  # 'c' is the column name
        ax1.set_ylabel(c, fontsize=10)  # Set the label for the y-axis
        ax1.tick_params(axis='y', labelsize=10)  # Set the font size for the y-axis ticks

        # Create the second y-axis for GHI with twinx
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x='month', y=ghi_column, hue="location", ax=ax2, legend=True, color='r', linestyle='--')
        ax2.set_ylabel(ghi_column, fontsize=10)  # Set the label for the y-axis
        ax2.tick_params(axis='y', labelcolor='r', labelsize=10)  # Set the font size for the y-axis ticks

        # To make the legend work for both axes, we need to manually create the legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        # avoid overlapping the legend
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=2)

        # Save the plot before showing
        plt.savefig(f'plots/pr_power_vs_ghi/ghi_vs_{c}.png')

def plot_power_air_temp(df):
    columns = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2', 'apolon_inn1', 'apolon_inn2', 'apolon_ref']

    # Assuming 'air_temp_column' is the name of your column that contains air temperature values
    air_temp_column = 'air_temp'  # Replace with your actual column name for air temperature if different

    # Set the style outside the loop
    sns.set(style="whitegrid")

    # Create the 'month' column if it's not already there
    if 'month' not in df.columns:
        df['month'] = df['period_end'].dt.month

    # Loop through each column intended for comparison with air temperature
    for c in columns:
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(11.7, 8.27))

        # Plot the solar panel output on ax1
        sns.lineplot(data=df, x='month', y=c, hue="location", ax=ax1, legend=True)
        ax1.set_ylabel(c, fontsize=10)  # Set the label for the y-axis
        ax1.tick_params(axis='y', labelsize=10)

        # Create the second y-axis for GHI with twinx
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x='month', y=air_temp_column, hue="location", ax=ax2, legend=True, color='r', linestyle='--')
        ax2.set_ylabel(air_temp_column, fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=10)

        # To make the legend work for both axes, we need to manually create the legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        # avoid overlapping the legend
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=2)

        # Save the plot before showing
        plt.savefig(f'plots/power_vs_air_temp/air_temp_vs_{c}.png')

def plot_power_ghi(df):
    columns = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2', 'apolon_inn1', 'apolon_inn2', 'apolon_ref']

    # Assuming 'ghi_column' is the name of your column that contains GHI values
    ghi_column = 'ghi'  # Replace with your actual column name for GHI if different

    # Set the style
    sns.set(style="whitegrid")

    # Create the 'month' column for each year
    df['month'] = df['period_end'].dt.year.astype(str) + ' ' + df['period_end'].dt.month.astype(str)

    # Loop through each column intended for comparison with GHI
    for c in columns:
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(11.7, 8.27))

        # Plot the solar panel output on ax1
        bar_plot = sns.lineplot(data=df, x='month', y=c, hue="location", ax=ax1)
        ax1.set_ylabel(c, fontsize=10)
        plt.xticks(rotation=90)  # Set the x-axis labels to rotate 90 degrees

        # Create the second y-axis for GHI with twinx
        ax2 = ax1.twinx()

        # Plot GHI on the secondary y-axis
        line_plot = sns.lineplot(data=df, x='month', y=ghi_column, hue="location", ax=ax2, color='r', linestyle='--')
        ax2.set_ylabel(ghi_column, fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=10)
        plt.xticks(rotation=90)  # Set the x-axis labels to rotate 90 degrees

        # Manually handle the legends
        handles1, labels1 = bar_plot.get_legend_handles_labels()
        handles2, labels2 = line_plot.get_legend_handles_labels()
        # Merge and add legend
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

        # Save the plot before showing
        plt.savefig(f'plots/power_vs_ghi/ghi_vs_{c}.png')

        # Clear the plot for the next iteration
        plt.clf()

# plot all panel types generation at two locations in subplots

def plot_panel_generation(df):
    panel_types = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2', 'apolon_inn1', 'apolon_inn2', 'apolon_ref']
    locations = df['location'].unique()
    #locations = ['Oslo', 'Tunisia']

    # Calculate the number of rows and columns for the subplots
    n_panels = len(panel_types)
    n_cols = 2
    n_rows = (n_panels + 1) // n_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

    # Iterate over each panel type and create a subplot
    for i, panel_type in enumerate(panel_types):
        ax = axes[i // n_cols, i % n_cols]
        
        # Filter data for current panel type
        panel_data = df[['period_end', 'location', panel_type]]

        # define the year and month columns and time column
        panel_data['year'] = panel_data['period_end'].dt.year
        panel_data['month'] = panel_data['period_end'].dt.month
        panel_data['time'] = panel_data['year'].astype(str)[2:] + '-' + panel_data['month'].astype(str)

        # Plot generation data for each location
        for location in locations:
            location_data = panel_data[panel_data['location'] == location]
            sns.lineplot(data=location_data, x='time', y=panel_type, label=location, ax=ax)

        ax.set_title(panel_type)
        ax.legend()
        # rotate x-axis labels for each subplot to avoid overlapping
        ax.tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'plots/panel_generation_Spain.png')

# Usage
def plot_panel_pr(df):
    df_new = df.copy()
    panel_types = ['manufacturer1_mono-facial', 'manufacturer1_bi-facial', 'manufacturer1_inn2_HFL', 'manufacturer1_inn2_HFB','manufacturer2_inn2_HFL','manufacturer2_inn2_HFL','manufacturer2_ref']
    locations = df_new['location'].unique()
    
    # Calculate the number of rows and columns for the subplots
    n_panels = len(panel_types)
    n_cols = 2
    n_rows = (n_panels + 1) // n_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

    # Iterate over each panel type and create a subplot
    for i, panel_type in enumerate(panel_types):
        ax = axes[i // n_cols, i % n_cols]

        # remove the rows with 0 values for the panel type
        df_new = df_new[df_new[panel_type] > 0]
        
        # Filter data for the coulumns period_end, location, and panel_type
        panel_data = df_new[['period_end', 'location', panel_type]]

        # x axis is year and month from period_end
        panel_data['year'] = panel_data['period_end'].dt.year
        panel_data['month'] = panel_data['period_end'].dt.month
        panel_data['time'] = panel_data['year'].astype(str)[2:] + '-' + panel_data['month'].astype(str)

        # Plot generation data for each location
        # ignore ploting the data with zero generation
        panel_data = panel_data[panel_data[panel_type] > 0]
        for location in locations:
            location_data = panel_data[panel_data['location'] == location]
            sns.lineplot(data=location_data, x='time', y=panel_type, ax=ax, label=location )

        ax.set_title(panel_type)
        ax.legend()
        ax.tick_params(axis='x', rotation=90)

        # Adjust layout
        plt.tight_layout()
        # Save the plot before showing
        plt.savefig(f'plots/panel_pr.png')

def plot_yearly_ghi(df):
    # Assuming 'ghi_column' is the name of your column that contains GHI values
    ghi_column = 'ghi'  # Replace with your actual column name for GHI if different

    # Set the style
    sns.set(style="whitegrid")

    # define the year and month columns
    df['year'] = df['period_end'].dt.year
    df['month'] = df['period_end'].dt.month


    # plot the monthly ghi for each location in each year
    # include locations in subplots
    locations = df['location'].unique()
    for location in locations:
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(11.7, 8.27))

        # Plot the solar panel output on ax1
        sns.lineplot(data=df[df['location'] == location], x='month', y=ghi_column, hue="year", ax=ax1, legend=True)
        ax1.set_ylabel(" W/m^2", fontsize=15)  # Set the label for the y-axis
        ax1.tick_params(axis='y', labelsize=15)
        # set the x-axis labels
        ax1.set_xlabel("Month", fontsize=15)
        ax1.tick_params(axis='x', labelsize=15)
        #ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=15)

        # Save the plot before showing
        plt.savefig(f'plots/ghi_vs_{location}.png')

        # Clear the plot for the next iteration
        plt.clf()
    
    # plot the monthly ghi for each location in all years as a time series
    # Create a new figure and axis for each plot
    fig, ax1 = plt.subplots(figsize=(20, 20))
    plt.xticks(rotation=90)
    # Plot the solar panel output on ax1
    #set x as month-year
    df['month_year'] = df['period_end'].dt.strftime('%b-%Y')
    sns.lineplot(data=df, x='month_year', y=ghi_column, hue="location", ax=ax1, legend=True)
    # increase legend font size
    ax1.legend(fontsize=20)
    ax1.set_ylabel(" W/m^2", fontsize=20)  # Set the label for the y-axis
    ax1.tick_params(axis='y', labelsize=20)
    # set the x-axis labels
    ax1.set_xlabel("Month-Year", fontsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    # rotate x-axis labels
    plt.xticks(rotation=90)
    #change the legend names Tunisa to Touezeur and Spain to Sevilla
    plt.savefig(f'plots/ghi_vs_all.png')

def plot_yearly_temp(df):
    # Assuming 'ghi_column' is the name of your column that contains GHI values
    temp_column = 'air_temp'  # Replace with your actual column name for GHI if different

    # Set the style
    sns.set(style="whitegrid")

    # define the year and month columns
    df['year'] = df['period_end'].dt.year
    df['month'] = df['period_end'].dt.month


    # plot the monthly ghi for each location in each year
    # include locations in subplots
    locations = df['location'].unique()
    for location in locations:
        # Create a new figure and axis for each plot
        fig, ax1 = plt.subplots(figsize=(20, 20))

        # Plot the solar panel output on ax1
        sns.lineplot(data=df[df['location'] == location], x='month', y=temp_column, hue="year", ax=ax1, legend=True)
        ax1.set_ylabel("C^0", fontsize=20)  # Set the label for the y-axis
        ax1.tick_params(axis='y', labelsize=20)
        # set the x-axis labels
        ax1.set_xlabel("Month", fontsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        # increase legend font size
        ax1.legend(fontsize=20)

        # Save the plot before showing
        plt.savefig(f'plots/temp_vs_{location}.png')

        # Clear the plot for the next iteration
        plt.clf()
    
     # plot the monthly ghi for each location in all years as a time series
    # Create a new figure and axis for each plot
  
    fig, ax1 = plt.subplots(figsize=(20, 20))
    plt.xticks(rotation=90)
    # Plot the solar panel output on ax1
    #set x as month-year
    df['month_year'] = df['period_end'].dt.strftime('%b-%Y')
    sns.lineplot(data=df, x='month_year', y=temp_column, hue="location", ax=ax1, legend=True)
    ax1.set_ylabel("C^0", fontsize=20)  # Set the label for the y-axis
    ax1.tick_params(axis='y', labelsize=20)
    # set the x-axis labels
    ax1.set_xlabel("Month-Year", fontsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.legend(fontsize=20)
    # rotate x-axis labels
    plt.xticks(rotation=90)
    plt.savefig(f'plots/temp_vs_all.png')

def plot_trend_seasonality(df, panel_types, locations, resample_freq, decompose_freq):
    df = df.copy()
   
    df = df[df['ghi'] > 0]
    # remove october 2023
    df = df[df['period_end'] < '2023-10-01']
    # remove may 2023
    df = df[df['period_end'] > '2021-06-01']
    # Ensure 'period_end' is in datetime format
    df['period_end'] = pd.to_datetime(df['period_end'])
    
    # Create a month-year column for grouping
    df['month_year'] = df['period_end'].dt.to_period('M')
    
    # Sort the data by month_year (chronologically) and location
    df.sort_values(['location'], inplace=True)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(figsize=(15, 15), sharex=True)

    # We'll store the results for each panel type in this list for further use if needed
    decompose_results = []

    for panel in panel_types:
        df = df[df[panel] > 0]
        # Calculate generation for each month-year for each location
        monthly_data = df.groupby(['location', 'month_year']).agg({panel: 'sum', 'ghi': 'sum'}).reset_index()
    
        # Convert 'month_year' to timestamps for plotting
        monthly_data['month_year'] = monthly_data['month_year'].dt.to_timestamp()

        # filter  the location
        monthly_data = monthly_data[monthly_data['location'] == 'Tozeur']
       
        # Calculate PR for each location
        monthly_data['PR'] = monthly_data[panel] / (monthly_data['ghi'] / 1000) * 100

        # Assuming 'df' is indexed by DateTime and 'panel' is the column name for panel type data
        result = seasonal_decompose(monthly_data['PR'], model='additive', period=12)  # or your chosen frequency
        decompose_results.append(result)
        
        # Plotting the trend
        #axes.plot(monthly_data['month_year'], result.trend, label=panel)
        # Plotting the seasonality
        axes.plot(monthly_data['month_year'], result.seasonal, label=panel)
        
        
        # Plotting the residuals
        #axes[2].plot(result.resid, label=panel)

    # Set titles and legends
    axes.set_title('Trend', fontsize=20)
    # yaxis
    axes.set_ylabel("PR %", fontsize=20)  # Set the label for the y-axis
    axes.legend(loc='best', fontsize=20)
    axes.tick_params(axis='both', which='major', labelsize=13)
    # xaxis
    axes.set_xlabel("Month-Year", fontsize=20)  # Set the label for the x-axis

    #axes[1].set_title('Seasonality')
    # yaxis
    #axes[1].set_ylabel("PR %", fontsize=20)  # Set the label for the y-axis
    # xaxis
    #axes[1].set_xlabel("Month-Year", fontsize=20)  # Set the label for the x-axis
    #axes[1].legend(loc='best')

    #axes[2].set_title('Residuals')
    #axes[2].legend(loc='best')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig('panels_decomposition_seasonility_tozeur.png')


        

def plot_trend_seasonality_sperate(df, panel_types, locations, resample_freq, decompose_freq):
    for location in locations:
        # Create a figure for subplots for each location
        n_panels = len(panel_types)
        fig, axes = plt.subplots(n_panels, 3, figsize=(15, n_panels * 3))

        for i, panel in enumerate(panel_types):
            # remove the zero values for the panel type
            df_new = df[df[panel] > 0]

            # Filter data by location and panel type
            panel_data = df[['period_end', 'location', panel]]

            # Filter the data by location
            panel_data = panel_data[panel_data['location'] == location]
            panel_data = panel_data.drop(columns=['location'])

            # Set 'period_end' as the index and resample
            panel_data = panel_data.set_index('period_end').resample(resample_freq).mean()

            # Check if there is enough data for decomposition
            if panel_data[panel].dropna().shape[0] < 2 * decompose_freq:
                print(f"Not enough data for seasonal decomposition of {panel} at {location}")
                continue

            # Seasonal decomposition
            result = seasonal_decompose(panel_data[panel].dropna(), model='additive', period=decompose_freq)

            # Plotting the trend
            axes[i, 0].plot(result.trend, label=panel)
            axes[i, 0].set_title(f'Trend for {panel}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('PR')

            # Plotting the seasonality
            axes[i, 1].plot(result.seasonal, label=panel)
            axes[i, 1].set_title(f'Seasonality for {panel}')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('PR')

            # plot the residuals
            axes[i, 2].plot(result.resid, label=panel)
            axes[i, 2].set_title(f'Residuals for {panel}')
            axes[i, 2].set_xlabel('Time')
            axes[i, 2].set_ylabel('PR')


            # Add legends to the subplots
            axes[i, 0].legend()
            axes[i, 1].legend()
            axes[i,2].legend()
            # rotate x-axis labels for each subplot to avoid overlapping
            axes[i, 0].tick_params(axis='x', rotation=90)
            axes[i, 1].tick_params(axis='x', rotation=90)
            axes[i, 2].tick_params(axis='x', rotation=90)

        plt.suptitle(f"Trend and Seasonality Analysis for {location}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to accommodate the title
        # increas the x and y axis ticks font size
        axes[i, 0].tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'plots/trend_seasonality_{location}_pr.png')
        plt.show()

# Custom color palette
custom_palette = {
    'Oslo': 'blue',
    'Tozeur': 'orange',
    'Sevilla': 'green'
}

# plot monthly performance ratio for each panel type
# create a single plot for panel type, include three locations in the same plot
def plot_PR(data, panel_type):
    df = data.copy()
    # Remove the rows with 0 generation for each location and 0 ghi
    df = df[df[panel_type] > 0]
    df = df[df['ghi'] > 0]
    # remove october 2023
    df = df[df['period_end'] < '2023-10-01']
    # remove may 2023
    df = df[df['period_end'] > '2021-06-01']
    df['ghi'] = df['ghi']


    # Create a month-year column for grouping
    df['month_year'] = df['period_end'].dt.to_period('M')
   

    # Sort the data by month_year (chronologically) and location
    df.sort_values(['location'], inplace=True)

    # Calculate generation for each month-year for each location
    monthly_data = df.groupby(['location', 'month_year']).agg({panel_type: 'sum', 'ghi': 'sum'}).reset_index()
    # sort the location Oslo, Touezeur, and Sevilla
    

    # Calculate PR for each location
    monthly_data['PR'] = monthly_data[panel_type] / (monthly_data['ghi'] / 1000) * 100

    # Plot the monthly PR for each location in all years as a time series
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Convert month_year to string format for plotting
    monthly_data['month_year_str'] = monthly_data['month_year'].dt.strftime('%b-%Y')

    # Plot the solar panel output on ax1 using the string formatted month_year
    sns.lineplot(data=monthly_data, x='month_year_str', y='PR', hue="location", ax=ax1, palette=custom_palette)



    ax1.set_ylabel("Performance Ratio (%)", fontsize=20)  # Set the label for the y-axis
    ax1.set_xlabel("Month-Year", fontsize=20)  # Set the label for the x-axis
    ax1.legend(title='Location', fontsize=20)  # Add a legend
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to fit everything nicely
    # increas the font in x and y axis ticks
    ax1.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig(f'plots/PR_{panel_type}.png')  # Save the figure


# plot monthly generation for each panel type
# create a single plot for panel type, include three locations in the same plot
def plot_generation(df, panel_type):
    df = data.copy()
    # Remove the rows with 0 generation for each location and 0 ghi
    df = df[df[panel_type] > 0]
    # remove may 2023
    df = df[df['period_end'] > '2021-06-01']

     # First, ensure 'period_end' is of datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df['period_end']):
        df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')

    # Create a month-year column for grouping
    df['month_year'] = df['period_end'].dt.to_period('M')

    # Sort the data by month_year (chronologically)
    df.sort_values(['location'], inplace=True)

    # Calculate generation for each month-year for each location
    monthly_data = df.groupby(['location', 'month_year']).agg({panel_type: 'sum', 'ghi': 'sum'}).reset_index()
    # Plot the monthly PR for each location in all years as a time series
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Convert month_year to string format for plotting
    monthly_data['month_year_str'] = monthly_data['month_year'].dt.strftime('%b-%Y')

    # Plot the solar panel output on ax1 using the string formatted month_year
    sns.lineplot(data=monthly_data, x='month_year_str', y=panel_type, hue="location", ax=ax1,   palette=custom_palette)

    ax1.set_ylabel("kWh/kWp", fontsize=20)  # Set the label for the y-axis
    ax1.set_xlabel("Month-Year", fontsize=20)  # Set the label for the x-axis
    ax1.legend(title='Location', fontsize=20)  # Add a legend
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to fit everything nicely
    ax1.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig(f'plots/generation_{panel_type}.png')  # Save the figure

# find the optimal temperature and ghi for each panel type
def find_correlation(data, panel_type):
    df = data.copy()
    # Remove the rows with 0 generation for each location and 0 ghi
    df = df[df[panel_type] > 0]
    df = df[df['ghi'] > 0]
    # remove october 2023
    df = df[df['period_end'] < '2023-10-01']
    # remove may 2023
    df = df[df['period_end'] > '2021-06-01']
    #df['ghi'] = df['ghi']*(5/60)


    # Create a month-year column for grouping
    df['month_year'] = df['period_end'].dt.to_period('M')
   

    # Sort the data by month_year (chronologically)
    df.sort_values(['location'], inplace=True)

    # Calculate generation for each month-year for each location
    monthly_data = df.groupby(['location', 'month_year']).agg({panel_type: 'sum', 'ghi': 'sum', 'air_temp': 'mean'}).reset_index()

    # Calculate PR for each location
    # Calculate PR for each location
    monthly_data['PR'] = monthly_data[panel_type] / (monthly_data['ghi'] / 1000) * 100
    # Plot the monthly PR for each location in all years as a time series
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Convert month_year to string format for plotting
    monthly_data['month_year_str'] = monthly_data['month_year'].dt.strftime('%b-%Y')

    # Plot the solar panel output on ax1 using the string formatted month_year
    # increase the size of the dots
    sns.scatterplot(data=monthly_data, x='air_temp', y='PR', hue="location", ax=ax1, palette=custom_palette, s=100)

    ax1.set_ylabel("Performance Ratio (%)", fontsize=20)  # Set the label for the y-axis
    ax1.set_xlabel("Temperature (C^0)", fontsize=20)  # Set the label for the x-axis
    ax1.legend(title='Location', fontsize=20)  # Add a legend
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to fit everything nicely
    ax1.tick_params(axis='both', which='major', labelsize=13)
    save_name = f'plots/PR_vs_temp_{panel_type}.png'
    plt.savefig(save_name)  # Save the figure
    # incre
    


if __name__ == '__main__':
    print(data_cleaned.describe())
    #df = analyse_missing_data(data_cleaned)
    df = data_cleaned
    #df = data_cleaned.set_index('period_end')
    #print(df.head())
    # remove the rows with 0 generation
    #types = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2_HFB', 'soliteck_inn2_HFL' 'apolon_inn1', 'apolon_inn2', 'apolon_ref']
    #for t in types:
    #    df = df[df[t] > 0]
    print(df.columns)
    print(df[['manufacturer1_mono-facial', 'manufacturer1_bi-facial', 'manufacturer1_inn2_HFL', 'manufacturer1_inn2_HFB','manufacturer2_inn2_HFL','manufacturer2_inn2_HFL','manufacturer2_ref']].describe())
            #  'manufacturer1_mono_pr',
            #  'manufacturer1_bi_pr', 
            #  'manufacturer1_inn2_FB_pr', 'manufacturer1_inn2_FL_pr',
            #  'manufacturer2_inn2_HFB_pr',
            #  'manufacturer2_inn2_HFL_pr', 'manufacturer2_ref_pr',
            #  'manufacturer1_mono_eff', 'manufacturer1_bi_eff',
            #  'manufacturer1_inn2__HFB_eff',
            #  'manufacturer1_inn2__HFL_eff', 
            #  'manufacturer2_inn2_FL_eff', 'manufacturer2_inn2_FB_eff',
            #  'manufacturer2_ref_eff', 'manufacturer1_inn2_HFB_pr',
            #  'manufacturer1_inn2_HFL_pr',  'manufacturer1_inn2_HFB_eff',
            #  'manufacturer1_inn2_HFL_eff', 'manufacturer2_inn2_HFB_eff',
            #  'manufacturer2_inn2_HFL_eff'].describe())
    

    #plot_cpr(df)
    #plot_power_ghi(df)
    #plot_power_air_temp(df)
    #plot_pr_power_ghi(df)
    #plot_pr_power_air_temp(df)
    #plot_panel_generation(df)
    #plot_panel_pr(df)
    #plot_yearly_ghi(df)
    #plot_yearly_temp(df)
    plot_PR(df, 'manufacturer1_mono-facial')
    plot_PR(df, 'manufacturer1_bi-facial')
    plot_PR(df, 'manufacturer1_inn2_HFL')
    plot_PR(df, 'manufacturer1_inn2_HFB')
    plot_PR(df, 'manufacturer2_inn2_HFL')
    plot_PR(df, 'manufacturer2_inn2_HFB')
    plot_PR(df, 'manufacturer2_ref')

    # set index to period_end
    #df = df.set_index('period_end')
    plot_generation(df, 'manufacturer1_bi-facial')
    plot_generation(df, 'manufacturer1_mono-facial')
    plot_generation(df, 'manufacturer1_inn2_HFL')
    plot_generation(df, 'manufacturer1_inn2_HFB')
    plot_generation(df, 'manufacturer2_inn2_HFL')
    plot_generation(df, 'manufacturer2_inn2_HFB')
    plot_generation(df, 'manufacturer2_ref')

    # find the correlation between the temperature and the panel generation pr
    find_correlation(df, 'manufacturer1_mono-facial')
    find_correlation(df, 'manufacturer1_bi-facial')
    find_correlation(df, 'manufacturer1_inn2_HFL')
    find_correlation(df, 'manufacturer1_inn2_HFB')
    find_correlation(df, 'manufacturer2_inn2_HFL')
    find_correlation(df, 'manufacturer2_inn2_HFB')
    find_correlation(df, 'manufacturer2_ref')

    # Define your panel types and locations
    #panel_types = ['soliteck_mono', 'soliteck_bi', 'soliteck_inn1', 'soliteck_inn2', 'apolon_inn1', 'apolon_inn2', 'apolon_ref']
    #panel_types = ['soliteck_mono_pr', 'soliteck_bi_pr', 'soliteck_inn1_pr', 'soliteck_inn2_pr', 'apolon_inn1_pr', 'apolon_inn2_pr', 'apolon_ref_pr']
    
    panel_types = ['manufacturer1_bi-facial', 'manufacturer1_inn2_HFL', 'manufacturer1_inn2_HFB']
    locations = ['Oslo', 'Touzer', 'Sevilla']  # Replace with your actual location names

    df['period_end'] = pd.to_datetime(df['period_end'])

    # Call the function with appropriate parameters
    plot_trend_seasonality(df, panel_types, locations, resample_freq='M', decompose_freq=1)

    # Call the function with appropriate parameters
    #plot_trend_seasonality_sperate(df, panel_types, locations, resample_freq='M', decompose_freq=12)


