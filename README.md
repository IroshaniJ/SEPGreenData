# SEPGreenData Repository

Welcome to the SEPGreenData repository! This repository contains Python scripts designed for analyzing the dataset titled "PV and Weather Dataset in Oslo, Touzer, and Sevilla," which is available at [Zenodo](https://zenodo.org/records/10420786).

## Getting Started

To begin, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

## Data Availability

You can find the related PV data, metadata on PV modules, and weather data shared at this Zenodo link.

## Repository Contents

1. read_csv_oslo.py: Script to read data from the Oslo demo site.

2. read_csv_tunisia.py: Script to read data from the Touzer demo site.

3. read_csv_spain.py: Script to read data from the Sevilla demo site.

4. combine_weather_oslo.py: Script to merge Oslo PV data with corresponding weather data.

5. combine_weather_tunisia.py: Script to merge Touzer PV data with corresponding weather data.

6. combine_weather_spain.py: Script to merge Sevilla PV data with corresponding weather data.

7. combine_locations.py: Script to amalgamate data from all three demo sites.

8. eda_2.py: Script for exploratory data analysis of the combined dataset.

9. plot_all_hours.py: Script to plot operating hours across all demo sites.

10. plots/: Directory where all generated figures are stored.


