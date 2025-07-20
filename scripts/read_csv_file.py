import pandas as pd
import os
import matplotlib.pyplot as plt

# Define the path to the CSV file
file_path = os.path.join(os.path.dirname(__file__), '../outputs/cities_MPI-ESM1-2-HR-ssp585/grd_182-263/metacomunity_biomass_182-263.csv')

# Read the CSV file and compute weighted mean

def read_and_compute_weighted_mean():# -> Series | None:
    try:
        data = pd.read_csv(file_path)
        if 'cveg' in data.columns and 'ocp' in data.columns and 'year' in data.columns:
            # Check if the sum of the ocp vector is near 1 for each year
            ocp_sums = data.groupby('year')['ocp'].sum()
            if not all(abs(ocp_sums - 1) < 1e-6):
                print("Warning: The sum of the 'ocp' vector is not close to 1 for all years.")

            # Compute weighted mean assuming ocp is normalized
            weighted_means = data.groupby('year').apply(
                lambda x: (x['cveg'] * x['ocp']).sum()
            )
            return weighted_means
        else:
            print("Required columns 'cveg', 'ocp', or 'year' are missing in the dataset.")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def read_co2_data():
    co2_file_path = os.path.join(os.path.dirname(__file__), '../input/co2/ssp585_CO2_annual_2015-2100.csv')
    try:
        co2_data = pd.read_csv(co2_file_path)
        return co2_data
    except FileNotFoundError:
        print(f"File not found: {co2_file_path}")
        return None

def plot_weighted_means_and_co2(weighted_means, co2_data):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Restrict the time span to the range of weighted_means index
    start_year = weighted_means.index.min()
    end_year = weighted_means.index.max()

    # Plot weighted means on the first y-axis
    ax1.plot(weighted_means.index, weighted_means.values, marker='o', linestyle='-', color='b', label='Weighted Mean of cveg')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Weighted Mean of cveg', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(start_year, end_year)  # Set x-axis limits to the time span of weighted_means

    # Create a second y-axis for CO2 data
    if co2_data is not None:
        co2_data_filtered = co2_data[(co2_data['year'] >= start_year) & (co2_data['year'] <= end_year)]
        ax2 = ax1.twinx()
        ax2.plot(co2_data_filtered['year'], co2_data_filtered['atm_co2'], linestyle='--', color='r', label='Atmospheric CO2')
        ax2.set_ylabel('Atmospheric CO2', fontsize=14, color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    # Add a title and adjust layout
    plt.title('Weighted Mean of cveg and Atmospheric CO2 by Year', fontsize=16)
    fig.tight_layout()
    plt.show()

def print_unique_ids_per_year(data):
    if 'year' in data.columns and 'pls_id' in data.columns:
        unique_ids_per_year = data.groupby('year')['pls_id'].nunique()
        print("Number of unique IDs per year:")
        print(unique_ids_per_year)
    else:
        print("Required columns 'year' or 'pls_id' are missing in the dataset.")

def plot_all_cveg_timeseries(data):
    if 'year' in data.columns and 'pls_id' in data.columns and 'cveg' in data.columns:
        plt.figure(figsize=(12, 8))
        for pls_id, subset in data.groupby('pls_id'):
            plt.plot(subset['year'], subset['cveg'], alpha=0.5, label=f'PLS ID: {pls_id}')
        plt.title('Time Series of cveg for All PLS IDs', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('cveg', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
    else:
        print("Required columns 'year', 'pls_id', or 'cveg' are missing in the dataset.")

def calculate_root_shoot_ratio(data):
    if 'year' in data.columns and 'pls_id' in data.columns and 'vp_croot' in data.columns and 'vp_cleaf' in data.columns and 'vp_cwood' in data.columns:
        # Calculate root:shoot ratio for each pls_id and year
        data['root_shoot_ratio'] = data['vp_croot'] / (data['vp_cleaf'] + data['vp_cwood'])
        root_shoot_ratios = data.groupby(['year', 'pls_id'])['root_shoot_ratio'].mean()
        print("Root:Shoot Ratio by Year and PLS ID:")
        print(root_shoot_ratios)
        return root_shoot_ratios
    else:
        print("Required columns 'year', 'pls_id', 'vp_croot', 'vp_cleaf', or 'vp_cwood' are missing in the dataset.")
        return None

if __name__ == "__main__":
    weighted_means = read_and_compute_weighted_mean()
    co2_data = read_co2_data()
    if weighted_means is not None:
        print("Weighted means by year:")
        print(weighted_means)
        plot_weighted_means_and_co2(weighted_means, co2_data)

    # Read the main dataset to print unique IDs per year
    try:
        data = pd.read_csv(file_path)
        print_unique_ids_per_year(data)
        plot_all_cveg_timeseries(data)
        r_s = calculate_root_shoot_ratio(data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")