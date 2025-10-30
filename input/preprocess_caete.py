#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAETE Data Preprocessing Script

Converts ISIMIP climate forcing data to optimized NetCDF format for CAETE model.
Creates station-based timeseries format similar to LPJ-GUESS preprocessing.

"""

from pathlib import Path
import argparse
import json
import os
import sys
import tomllib

from netCDF4 import Dataset, MFDataset, MFTime, stringtochar
import numpy as np
from numpy import exp, array, arange, hstack, concatenate, meshgrid, column_stack
from numpy import float32 as flt


sys.path.append("../src")
from _geos import pan_amazon_region

__what__ = "Pre-processing of input data for CAETE model"
__author__ = "jpdarela"
__date__ = "2025-07-12"
__description__ = """
This script converts ISIMIP climate forcing data to an optimized NetCDF format
for the CAETE model. It creates station-based timeseries similar to LPJ-GUESS
format but with CAETE-specific unit conversions and variable requirements.

Output format:
- Single NetCDF file with station-based timeseries format.
- Variables: tas, pr, ps, hurs, rsds, sfcwind, vpd + soil variables
- Dimensions: (station, time) for climate [Optimal chunking for extration of timeseries]; (station) for soil
- Includes gridlist for station coordinate mapping
"""

# Argument parser
parser = argparse.ArgumentParser(
    description=__description__,
    usage="python preprocess_caete.py [-h] [--dataset DATASET] [--mode MODE]"
)

parser.add_argument('--dataset', type=str, default="20CRv3-ERA5",
                    help='Main dataset folder (e.g., 20CRv3-ERA5)')
parser.add_argument('--mode', type=str, default="spinclim",
                    help='Mode of the dataset, e.g., spinclim, transclim, etc.')
parser.add_argument('--mask-file', type=str, default=None,
                    help="Path to the mask file (default from config)")

# CAETE variable metadata
caete_var_metadata = {
    'tas': ["degC", 'air_temperature'],
    'pr': ["mm day-1", "precipitation_flux"],
    'ps': ["hPa", "surface_air_pressure"],
    'hurs': ["1", 'relative_humidity'],
    'rsds': ['mol m-2 day-1', "surface_downwelling_shortwave_flux_in_air"],  # Updated units after conversion
    'sfcwind': ["m s-1", "wind_speed"],
    'vpd': ["kPa", "vapor_pressure_deficit"],
    'tn': ["g m-2", "soil_mass_content_of_nitrogen"],
    'tp': ["g m-2", "soil_mass_content_of_phosphorus"],
    'ap': ["g m-2", "soil_mass_content_of_available_phosphorus"],
    'ip': ["g m-2", "soil_mass_content_of_inorganic_phosphorus"],
    'op': ["g m-2", "soil_mass_content_of_organic_phosphorus"]
}

# Conversion factors for CAETE (from ISIMIP units to CAETE units)
# Based on CAETE configuration file (caete.toml)
conversion_factors = {
    'tas': -273.15,  # K to °C (subtract)
    'pr': 86400.0,   # kg m-2 s-1 to mm day-1 (multiply)
    'ps': 0.01,      # Pa to hPa (multiply)
    'hurs': 0.01,    # % to kg kg-1 (multiply)
    'rsds': 0.198,   # W m-2 to mol(photons) m-2 day-1 (multiply)
                     # Calculation: W m-2 × 86400 s/day × 0.5 (PAR fraction) ÷ 218000 J/mol = × 0.198
    # Note: 'sfcwind' has no conversion factor (already in m s-1)
}

# Global configuration
cc = type("PanAmazon", (), pan_amazon_region)()

# Read configuration
with open("./pre_processing.toml", 'rb') as f:
    config_data = tomllib.load(f)

# Parse arguments
args = parser.parse_args()
dataset = args.dataset
mode = args.mode

# Setup paths
climate_data = Path(config_data["climate_data"])
raw_data = climate_data / dataset / f"{mode}_raw"
soil_data = Path(config_data["soil_data"])

if args.mask_file is None:
    mask_file = Path(config_data["mask_file"])
else:
    mask_file = Path(args.mask_file)

# Output path
output_data = Path(f"{dataset}/{mode}")
os.makedirs(output_data, exist_ok=True)

# Load mask
mask = np.load(mask_file)

def timer(func):
    """Timer decorator"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        hours = int((end - start) // 3600)
        minutes = round(((end - start) % 3600) // 60)
        seconds = round((end - start) % 60)
        if hours == 0:
            print(f"Elapsed time: {minutes}:{seconds}")
        elif minutes == 0:
            print(f"Elapsed time: {seconds} seconds")
        else:
            print(f"Elapsed time: {hours}:{minutes}:{seconds}")
        return result
    return wrapper

def Vsat_slope(Tair: array, method=3) -> array:
    """
    Calculate saturation vapor pressure and slope of the curve.
    Uses Allen_1998 method by default for CAETE.
    """
    methods = ("Sonntag_1990", "Alduchov_1996", "Allen_1998")
    formula = methods[method - 1]

    if formula == "Allen_1998":
        a = 610.8
        b = 17.27
        c = 237.3

    tair_degc = Tair - 273.15
    Pa2kPa = 1e-3

    # saturation vapor pressure
    Esat = a * exp((b * tair_degc) / (c + tair_degc))
    Esat = Esat * Pa2kPa

    # slope of the saturation vapor pressure curve
    Delta = a * (exp((b * tair_degc)/(c + tair_degc)) * (b/(c + tair_degc) - (b * tair_degc)/(c + tair_degc)**2))
    Delta = Delta * Pa2kPa

    return Esat, Delta

def VPD(Tair: array, RH: array) -> array:
    """
    Estimate VPD from Tair (K) and RH (%).
    For CAETE: returns positive VPD (svp - avp)
    """
    svp = Vsat_slope(Tair)[0]  # (kPa)
    avp = svp * (RH / 100.0)

    return svp - avp  # Positive VPD for CAETE

def read_clim_data(var: str) -> MFDataset | Dataset:
    """Read climate data from NetCDF files"""
    try:
        files = raw_data.glob(f"*_{var}_*")
    except:
        raise FileNotFoundError(f"No netCDF file for variable {var} in {raw_data}")

    files_list = list(files)

    if len(files_list) == 0:
        raise FileNotFoundError(f"No netCDF file for variable {var} in {raw_data}")
    elif len(files_list) == 1:
        reader = Dataset
        to_read = files_list[0]
    else:
        reader = MFDataset
        to_read = files_list

    try:
        dataset = reader(to_read)
    except:
        raise FileNotFoundError(f"No netCDF file for variable {var} in {raw_data}")

    return dataset

def get_dataset_size(dataset: MFDataset | Dataset) -> int:
    """Get time dimension size"""
    return dataset.variables["time"][...].size

def read_soil_data(var: str):
    """Read soil data from numpy files"""
    soil_files = {
        'tn': config_data["tn_file"],
        'tp': config_data["tp_file"],
        'ap': config_data["ap_file"],
        'ip': config_data["ip_file"],
        'op': config_data["op_file"]
    }

    if var in soil_files:
        return np.load(os.path.join(soil_data, Path(soil_files[var])))
    else:
        raise ValueError(f"Unknown soil variable: {var}")

def get_metadata_from_dataset(variables: str):
    """Extract metadata from NetCDF dataset"""

    dss = [read_clim_data(var) for var in variables]
    times = []

    # Use MFTime to standardize the time dimension across datasets
    for ds in dss:
        if isinstance(ds, MFDataset):
            times.append(MFTime(ds.variables['time']))
        elif isinstance(ds, Dataset):
            times.append(ds.variables['time'])
        else:
            raise TypeError("Dataset must be either MFDataset or Dataset")

    # Check time metadata consistency
    for x in range(1, len(times)):
        assert np.all(times[0][:] == times[x][:]), "Time values are not equal across datasets"
        assert times[0].calendar == times[x].calendar, "Calendars are not equal across datasets"
        assert times[0].units == times[x].units, "Units are not equal across datasets"
        # assert times[0].standard_name == times[x].standard_name, "Standard names are not equal across datasets"

    var = variables[0]
    with read_clim_data(var) as dataset:
        var_obj = dataset.variables[var]
        time_obj = times[0]  # Use the first dataset's time object
        lat_obj = dataset.variables["lat"]
        lon_obj = dataset.variables["lon"]

        metadata = {
            'variable': {
                'standard_name': getattr(var_obj, 'standard_name', ''),
                'long_name': getattr(var_obj, 'long_name', ''),
                'units': getattr(var_obj, 'units', ''),
                'fill_value': getattr(var_obj, '_FillValue', 1e+20),
                'missing_value': getattr(var_obj, 'missing_value', 1e+20)
            },
            'time': {
                'units': time_obj.units,
                'calendar': getattr(time_obj, 'calendar', 'standard'),
                'axis': 'T',
                'time_data': time_obj[:]
            },
            'coordinates': {
                'lat_units': lat_obj.units,
                'lon_units': lon_obj.units,
                'lat_long_name': getattr(lat_obj, 'long_name', 'latitude'),
                'lon_long_name': getattr(lon_obj, 'long_name', 'longitude'),
                'lat_std_name': getattr(lat_obj, 'standard_name', 'latitude'),
                'lon_std_name': getattr(lon_obj, 'standard_name', 'longitude')
            }
        }

    for ds in dss:
        ds.close()
    del dss

    return metadata

def process_climate_variable(var: str):
    """
    VECTORIZED IMPLEMENTATION: Process a single climate variable and convert to station format.
    Returns (station_data, metadata)
    """
    print(f"Processing climate variable: {var}")

    # Get regional mask
    temp_mask = mask[cc.ymin:cc.ymax+1, cc.xmin:cc.xmax+1]
    ny, nx = temp_mask.shape

    # VECTORIZED STATION IDENTIFICATION
    # Create index arrays using meshgrid
    i_indices, j_indices = meshgrid(arange(ny), arange(nx), indexing='ij')
    valid_mask = ~temp_mask
    valid_i = i_indices[valid_mask]
    valid_j = j_indices[valid_mask]
    station_count = len(valid_i)

    # Read data
    with read_clim_data(var) as dataset:
        tsize = get_dataset_size(dataset)
        variable_obj = dataset.variables[var]

        # Check the actual lat/lon values in the file
        if 'lat' in dataset.variables and 'lon' in dataset.variables:
            file_lats = dataset.variables['lat'][:]
            file_lons = dataset.variables['lon'][:]
            print(f"  File lat range: [{file_lats.min():.2f}, {file_lats.max():.2f}]")
            print(f"  File lon range: [{file_lons.min():.2f}, {file_lons.max():.2f}]")
            print(f"  Extracting region indices y[{cc.ymin}:{cc.ymax+1}], x[{cc.xmin}:{cc.xmax+1}]")
            print(f"  Which corresponds to:")
            print(f"    Lat: [{file_lats[cc.ymin]:.2f}, {file_lats[cc.ymax]:.2f}]")
            print(f"    Lon: [{file_lons[cc.xmin]:.2f}, {file_lons[cc.xmax]:.2f}]")

        # VECTORIZED DATA EXTRACTION
        # Read all regional data at once
        print(f"  Reading all {tsize} time steps for region...")
        regional_data = variable_obj[:, cc.ymin:cc.ymax+1, cc.xmin:cc.xmax+1]  # (time, y, x)

        # Extract all station data at once: (time, station)
        station_data = regional_data[:, valid_i, valid_j]  # Transpose to (station, time)

        print(f"  Extracted {station_count} stations x {tsize} time steps")

    # Apply unit conversion (vectorized)
    if var in conversion_factors:
        factor = conversion_factors[var]
        if var == 'tas':  # Subtraction for temperature
            station_data = station_data + factor
            print(f"  Applied conversion: {var} (K to °C, factor: {factor})")
        else:  # Multiplication for others
            station_data = station_data * factor
            print(f"  Applied conversion: {var} (factor: {factor})")
    else:
        print(f"  No conversion applied for {var} (using original units)")

    print(f"  Completed {var}: {station_data.shape} -> {station_data.nbytes/1024**2:.1f} MB")
    return station_data

def process_soil_variable(var: str):
    """
    Process a single soil variable and convert to station format.
    Returns 1D array with one value per station.
    """
    print(f"Processing soil variable: {var}")

    # Get regional mask
    temp_mask = mask[cc.ymin:cc.ymax+1, cc.xmin:cc.xmax+1]
    ny, nx = temp_mask.shape

    # VECTORIZED STATION IDENTIFICATION (like climate processing)
    i_indices, j_indices = meshgrid(arange(ny), arange(nx), indexing='ij')
    valid_mask = ~temp_mask
    valid_i = i_indices[valid_mask]
    valid_j = j_indices[valid_mask]
    station_count = len(valid_i)

    # Read global soil data and slice to region
    soil_global = read_soil_data(var)
    soil_regional = soil_global[cc.ymin:cc.ymax+1, cc.xmin:cc.xmax+1]

    # VECTORIZED EXTRACTION
    station_data = soil_regional[valid_i, valid_j].astype(flt)

    # Add value range information with PROPER SIZE CALCULATION
    data_size_kb = station_data.nbytes / 1024  # Calculate KB correctly

    print(f"  Extracted {station_count} stations")
    print(f"  Value range: [{station_data.min():.3f}, {station_data.max():.3f}] {caete_var_metadata[var][0]}")
    print(f"  Completed {var}: {station_data.shape} -> {data_size_kb:.1f} KB")

    return station_data

def create_station_coordinates():
    """Create coordinate arrays for stations using vectorized operations"""
    # Get regional mask
    temp_mask = mask[cc.ymin:cc.ymax+1, cc.xmin:cc.xmax+1]
    ny, nx = temp_mask.shape

    # VECTORIZED APPROACH using meshgrid
    i_indices, j_indices = meshgrid(arange(ny), arange(nx), indexing='ij')

    # Find valid stations (vectorized)
    valid_mask = ~temp_mask
    valid_i = i_indices[valid_mask]
    valid_j = j_indices[valid_mask]

    # Calculate global coordinates (vectorized)
    global_y = valid_i + cc.ymin
    global_x = valid_j + cc.xmin

    # Convert to lat/lon (vectorized) !! Assuming 0.5 degree resolution
    lats = 90.0 - (global_y + 0.5) * 0.5
    lons = (global_x + 0.5) * 0.5 - 180.0

    # Create station names (vectorized)
    station_names = np.array([f"station_{y}-{x}" for y, x in zip(global_y, global_x)], dtype='<U15')
    station_indices = column_stack([global_y, global_x])

    return lats, lons, station_names, station_indices

def calculate_vpd(tas_data, hurs_data):
    """Calculate VPD from temperature and humidity data"""
    print("Calculating VPD from tas and hurs...")

    # Convert CAETE units back to K and % for calculation
    tas_k = tas_data + 273.15  # °C to K
    hurs_pct = hurs_data / 0.01  # kg kg-1 to %

    # Calculate VPD
    vpd_data = VPD(tas_k, hurs_pct)

    print(f"  Completed VPD: {vpd_data.shape} -> {vpd_data.nbytes/1024**2:.1f} MB")
    return vpd_data

def write_gridlist(lats, lons, station_names, station_indices, output_path):
    """Write gridlist file for CAETE"""
    gridlist_file = output_path / "gridlist_caete.csv"

    with open(gridlist_file, 'w') as f:
        f.write("station_id,lat,lon,station_name,global_y,global_x\n")
        for i in range(len(lats)):
            f.write(f"{i},{lats[i]:.4f},{lons[i]:.4f},{station_names[i]},"
                   f"{station_indices[i,0]},{station_indices[i,1]}\n")

    print(f"Written gridlist: {gridlist_file}")
    return gridlist_file

def write_caete_netcdf(climate_data, soil_data, vpd_data, lats, lons, station_names,
                       metadata, output_path):
    """Write CAETE NetCDF file in station-timeseries format"""

    output_file = output_path / f"caete_input_{dataset}_{mode}.nc"

    print(f"Writing CAETE NetCDF file: {output_file}")

    # Get dimensions
    n_stations = len(lats)
    n_times = climate_data['tas'].shape[0]  # Assuming all climate variables have same time dimension

    # OPTIMAL CHUNKING for access pattern (reading all time for selected stations)
    time_chunk = n_times
    station_chunk = min(128, n_stations)  # Chunk 128 stations
    chunks = (time_chunk, station_chunk)

    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        # Global attributes
        nc.description = f"CAETE input data - {dataset} {mode}"
        nc.reference = "ISIMIP climate forcing data converted for CAETE model"
        nc.featureType = "timeSeries"
        nc.created = f"Created by preprocess_caete.py on {np.datetime64('now')}"

        # Create dimensions
        nc.createDimension("time", None)
        nc.createDimension("station", n_stations)
        nc.createDimension("string_length", 15)

        # Create coordinate variables
        time_var = nc.createVariable("time", 'f4', ("time",))
        station_var = nc.createVariable("station", 'i4', ("station",))
        lat_var = nc.createVariable("lat", 'f4', ("station",))
        lon_var = nc.createVariable("lon", 'f4', ("station",))

        station_name_var = nc.createVariable("station_name", 'S1', ("station", "string_length"))
        station_name_var._Encoding = 'ascii'
        # Fill coordinate variables
        time_var[:] = metadata['time']['time_data']
        station_var[:] = arange(n_stations, dtype='i4')
        lat_var[:] = lats
        lon_var[:] = lons

        station_names_padded = np.array([name.ljust(15)[:15] for name in station_names], dtype='S15')
        station_name_var[:] = stringtochar(station_names_padded, encoding='ascii')

        # COORDINATE ATTRIBUTES
        time_var.units = metadata['time']['units']
        time_var.calendar = metadata['time']['calendar']
        time_var.standard_name = "time"
        time_var.axis = "T"

        station_var.long_name = "Station index"
        station_var.cf_role = "timeseries_id"

        lat_var.units = "degrees_north"
        lat_var.long_name = "Latitude"
        lat_var.standard_name = "latitude"
        lat_var.axis = "Y"

        lon_var.units = "degrees_east"
        lon_var.long_name = "Longitude"
        lon_var.standard_name = "longitude"
        lat_var.axis = "X"

        station_name_var.long_name = "Station identifier"
        station_name_var.cf_role = "timeseries_id"

        # Create climate variables
        for var_name, data in climate_data.items():
            var_obj = nc.createVariable(var_name, 'f4', ("time", "station"),
                                      fill_value=1e+20, compression='zlib', complevel=1,
                                      chunksizes=chunks)
            var_obj[:] = data

            var_obj.units = caete_var_metadata[var_name][0]
            if caete_var_metadata[var_name][1] is not None:
                var_obj.standard_name = caete_var_metadata[var_name][1]
            var_obj.long_name = f"{var_name.upper()} for CAETE"
            var_obj.coordinates = "lat lon"

        # Add VPD with CDO-compatible dimensions
        vpd_var = nc.createVariable("vpd", 'f4', ("time", "station"),
                                   fill_value=1e+20, compression='zlib', complevel=1,
                                   chunksizes=chunks)
        vpd_var[:] = vpd_data
        vpd_var.units = caete_var_metadata['vpd'][0]
        vpd_var.standard_name = caete_var_metadata['vpd'][1]
        vpd_var.long_name = "Vapor Pressure Deficit for CAETE"
        vpd_var.coordinates = "lat lon"

        # Create soil variables (remain as station-only dimensions)
        for var_name, data in soil_data.items():
            var_obj = nc.createVariable(var_name, 'f4', ("station",),
                                      fill_value=1e+20, contiguous=True)
            var_obj[:] = data
            var_obj.units = caete_var_metadata[var_name][0]
            var_obj.long_name = f"{var_name.upper()} for CAETE"
            var_obj.coordinates = "lat lon"

    print(f"Successfully written CDO-compatible file: {output_file}")
    return output_file

def assemble_climate_data_vectorized(climate_vars, vpd_data):
    """
    VECTORIZED IMPLEMENTATION: Use hstack to efficiently combine all climate data
    """
    print("Assembling climate data using vectorized operations...")

    # Get all climate arrays in a list for concatenation
    data_arrays = []
    var_names = []

    for var in climate_vars:
        if var in climate_data:
            data_arrays.append(climate_data[var])
            var_names.append(var)

    # Add VPD data
    data_arrays.append(vpd_data)
    var_names.append('vpd')

    # VECTORIZED COMBINATION using hstack for efficient memory layout
    # This creates a single contiguous array (station, total_time_vars)
    print(f"  Combining {len(data_arrays)} variables using hstack...")
    combined_data = hstack(data_arrays)  # Horizontal stacking along time dimension

    print(f"  Combined shape: {combined_data.shape}")
    print(f"  Combined size: {combined_data.nbytes/1024**2:.1f} MB")

    return combined_data, var_names

def process_multiple_climate_files_vectorized(climate_vars):
    """
    VECTORIZED IMPLEMENTATION: Process multiple climate files efficiently using concatenate
    """
    print("Processing multiple climate files with vectorized operations...")

    # Process variables in batches for memory efficiency
    batch_size = 3  # Process 3 variables at a time
    all_data = {}

    for i in range(0, len(climate_vars), batch_size):
        batch_vars = climate_vars[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}: {batch_vars}")

        # Process batch variables
        batch_data = []
        for var in batch_vars:
            data = process_climate_variable(var)
            batch_data.append(data)
            all_data[var] = data

        # VECTORIZED BATCH PROCESSING using concatenate for validation
        if len(batch_data) > 1:
            # Concatenate along station axis for validation (optional)
            batch_combined = concatenate(batch_data, axis=0)  # Stack stations
            print(f"    Batch shape validation: {batch_combined.shape}")
            del batch_combined  # Free memory

        # Clear batch data to free memory
        del batch_data

    return all_data

def validate_data_quality_vectorized(climate_data, vpd_data, station_coords):
    """
    VECTORIZED IMPLEMENTATION: Comprehensive data quality validation using numpy operations
    """
    print("Performing vectorized data quality validation...")

    # VECTORIZED QUALITY CHECKS
    total_stations = len(station_coords)

    # Check for missing data across all variables (vectorized)
    quality_report = {}

    for var_name, data in climate_data.items():
        # Vectorized NaN detection
        nan_count = np.isnan(data).sum()
        nan_percentage = (nan_count / data.size) * 100

        # Vectorized range validation
        if var_name == 'tas':
            # Temperature range check (vectorized)
            valid_range = (-50 <= data) & (data <= 60)  # °C
        elif var_name == 'pr':
            # Precipitation range check (vectorized)
            valid_range = (0 <= data) & (data <= 1000)  # mm/day
        elif var_name == 'hurs':
            # Humidity range check (vectorized)
            valid_range = (0 <= data) & (data <= 1)  # kg kg-1
        elif var_name == 'rsds':
            # Solar radiation range check (vectorized) - after conversion to mol m-2 day-1
            valid_range = (0 <= data) & (data <= 60)  # mol(photons) m-2 day-1
        elif var_name == 'ps':
            # Pressure range check (vectorized)
            valid_range = (800 <= data) & (data <= 1100)  # hPa
        elif var_name == 'sfcwind':
            # Wind speed range check (vectorized)
            valid_range = (0 <= data) & (data <= 50)  # m s-1
        else:
            valid_range = np.ones_like(data, dtype=bool)  # Default: all valid

        invalid_count = (~valid_range).sum()
        invalid_percentage = (invalid_count / data.size) * 100

        quality_report[var_name] = {
            'shape': data.shape,
            'nan_count': int(nan_count),
            'nan_percentage': float(nan_percentage),
            'invalid_count': int(invalid_count),
            'invalid_percentage': float(invalid_percentage),
            'memory_mb': float(data.nbytes / 1024**2)
        }

    # VPD quality check (vectorized)
    vpd_nan_count = np.isnan(vpd_data).sum()
    vpd_invalid = (vpd_data < 0).sum()  # VPD should be positive

    quality_report['vpd'] = {
        'shape': vpd_data.shape,
        'nan_count': int(vpd_nan_count),
        'nan_percentage': float((vpd_nan_count / vpd_data.size) * 100),
        'invalid_count': int(vpd_invalid),
        'invalid_percentage': float((vpd_invalid / vpd_data.size) * 100),
        'memory_mb': float(vpd_data.nbytes / 1024**2)
    }

    # VECTORIZED SUMMARY STATISTICS
    print("  Data quality summary:")
    total_memory = sum(report['memory_mb'] for report in quality_report.values())
    print(f"    Total memory usage: {total_memory:.1f} MB")
    print(f"    Total stations: {total_stations}")

    for var_name, report in quality_report.items():
        print(f"    {var_name}: {report['nan_percentage']:.2f}% NaN, "
              f"{report['invalid_percentage']:.2f}% invalid, "
              f"{report['memory_mb']:.1f} MB")

    return quality_report

@timer
def main():
    """Main processing function"""
    print("CAETE Data Preprocessing")
    print("=" * 50)
    print(f"Dataset: {dataset}")
    print(f"Mode: {mode}")
    print(f"Region: Pan Amazon ({cc.ymin}:{cc.ymax}, {cc.xmin}:{cc.xmax})")
    print()

    # Climate variables to process
    climate_vars = ['tas', 'pr', 'ps', 'hurs', 'rsds']
    soil_vars = ['tn', 'tp', 'ap', 'ip', 'op']

    # Create station coordinates
    print("Creating station coordinate system...")
    lats, lons, station_names, station_indices = create_station_coordinates()
    n_stations = len(lats)
    print(f"Found {n_stations} valid stations in Pan Amazon region")
    print()

    # Get metadata from first climate variable
    metadata = get_metadata_from_dataset(climate_vars)

    # Process climate variables using VECTORIZED APPROACH
    print("Processing climate variables with vectorized operations...")
    climate_data = process_multiple_climate_files_vectorized(climate_vars)
    print()

    # Calculate VPD
    vpd_data = calculate_vpd(climate_data['tas'], climate_data['hurs'])
    print()

    # VECTORIZED DATA QUALITY VALIDATION
    quality_report = validate_data_quality_vectorized(climate_data, vpd_data, station_indices)
    print()

    # Process soil variables
    print("Processing soil variables...")
    soil_data = {}
    for var in soil_vars:
        soil_data[var] = process_soil_variable(var)
    print()

    # Write gridlist
    gridlist_file = write_gridlist(lats, lons, station_names, station_indices, output_data)

    # Write NetCDF
    netcdf_file = write_caete_netcdf(climate_data, soil_data, vpd_data, lats, lons,
                                    station_names, metadata, output_data)

    # Validate data quality
    print("Validating data quality...")
    quality_report = validate_data_quality_vectorized(climate_data, vpd_data, station_indices)
    print()

    # Write summary metadata
    summary = {
        'dataset': dataset,
        'mode': mode,
        'region': 'pan_amazon',
        'n_stations': int(n_stations),
        'n_timesteps': int(climate_data['tas'].shape[0]),
        'climate_variables': climate_vars + ['vpd'],
        'soil_variables': soil_vars,
        'output_files': {
            'netcdf': str(netcdf_file),
            'gridlist': str(gridlist_file)
        },
        'unit_conversions': conversion_factors
    }

    summary_file = output_data / "caete_preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("Processing completed successfully!")
    print(f"Output files:")
    print(f"  NetCDF: {netcdf_file}")
    print(f"  Gridlist: {gridlist_file}")
    print(f"  Summary: {summary_file}")

if __name__ == "__main__":
    main()
