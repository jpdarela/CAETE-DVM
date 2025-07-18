from pathlib import Path

import argparse
import bz2
import concurrent.futures
import gc
import os
import sys
import pickle as pkl
import tomllib
from copy import deepcopy

from netCDF4 import Dataset, MFDataset, MFTime
from numba import njit
# from numpy.typing import NDArray
# from typing import Generator
import numpy as np


sys.path.append("../src")
from _geos import pan_amazon_region


__what__ = "Pre-processing of input data to feed CAETÊ"
__author__ = "jpdarela"
__date__ = "Mon Dec 28 18:08:27 -03 2020"
__descr__ = """Preprocesses climate and soil data for the CAETÊ Dynamic Vegetation Model.

OVERVIEW:
    Transforms raw ISIMIP climate datasets and soil nutrient data into gridcell-specific
    compressed files for efficient CAETÊ model input. Processes only valid land pixels
    within the Pan-Amazon region as defined by the geographic coordinate bounds.
    Variable units are not transformed, but the data is cropped to the Pan-Amazon region.

INPUT DATA STRUCTURE & PATH RESOLUTION:
    Raw Climate Data Search Path:
    The script constructs the climate data path using a hierarchical approach:

    1. Base Path: Defined in pre_processing.toml as "climate_data"
       Example: "/path/to/climate/root/"

    2. Dataset Directory: Specified via --dataset argument
       Example: "20CRv3-ERA5"

    3. Mode-Specific Raw Data Folder: "{mode}_raw"
       Example: "spinclim_raw" or "transclim_raw"

    Final Search Path: {climate_data}/{dataset}/{mode}_raw/
    Complete Example: "/path/to/climate/root/20CRv3-ERA5/spinclim_raw/"

    Expected NetCDF Files in Raw Data Directory:
    ├── *_hurs_*.nc     # Relative humidity (any files matching pattern)
    ├── *_tas_*.nc      # Air temperature
    ├── *_pr_*.nc       # Precipitation
    ├── *_ps_*.nc       # Surface pressure
    ├── *_rsds_*.nc     # Solar radiation
    └── *_sfcwind_*.nc  # Wind speed

    File Discovery: Uses glob patterns (*_{variable}_*) to automatically detect
    single files or multiple files per variable (handled via MFDataset).

    Soil Data Path Resolution:
    Base path defined in pre_processing.toml as "soil_data", with individual
    filenames specified in [soil_files] section:
    ├── {tn_file}       # Total nitrogen (.npy)
    ├── {tp_file}       # Total phosphorus (.npy)
    ├── {ap_file}       # Available phosphorus (.npy)
    ├── {ip_file}       # Inorganic phosphorus (.npy)
    └── {op_file}       # Organic phosphorus (.npy)

OUTPUT STRUCTURE & DIRECTORY MIRRORING:
    The output directory structure mirrors the input climate data organization:

    Input:  {climate_data}/{dataset}/{mode}_raw/
    Output: ./{dataset}/{mode}/

    Example:
    Input:  "/data/climate/20CRv3-ERA5/spinclim_raw/"
    Output: "./20CRv3-ERA5/spinclim/"

    Generated Files:
    ├── input_data_{global_y}-{global_x}.pbz2    # Per-gridcell data packages
    └── METADATA.pbz2                            # NetCDF coordinate/time metadata

    Where {global_y} and {global_x} are the original global grid coordinates
    (y ∈ [0,360), x ∈ [0,720) for 0.5° resolution).

PROCESSING WORKFLOW:
    1. Resolves climate data path: {climate_data}/{dataset}/{mode}_raw/
    2. Discovers NetCDF files using glob patterns (*_{var}_*)
    3. Extracts metadata (time, lat, lon) from first climate dataset
    4. Applies mask filtering to identify valid land pixels
    5. Crops all data to Pan-Amazon region bounds (ymin:ymax, xmin:xmax)
    6. Loads soil nutrient arrays and crops to same region
    7. Creates time-series matrices for all climate variables simultaneously
    8. Packages each gridcell's complete dataset (climate + soil) into .pbz2 files

GRIDCELL DATA FORMAT:
    Each .pbz2 file contains a dictionary with:
    • Climate time series: hurs[time], tas[time], pr[time], ps[time], rsds[time], sfcwind[time]
    • Soil scalars: tn, tp, ap, ip, op (single values per gridcell)

COORDINATE SYSTEM:
    • Global indices: y ∈ [0,360), x ∈ [0,720) for 0.5° resolution
    • Regional processing: Constrained to Pan-Amazon bounds from src/_geos
    • Mask-based filtering: Only processes pixels where mask[y,x] = False
    • Output filenames use global coordinates: input_data_{global_y}-{global_x}.pbz2

CONFIGURATION (pre_processing.toml):
    [climate_data] = "/path/to/climate/root"     # Base path for dataset directories
    [soil_data] = "/path/to/soil/arrays"         # Directory containing .npy files
    [mask_file] = "/path/to/land_mask.npy"       # Boolean mask (360×720)

    [tn_file] = "filename.npy"                   # Total nitrogen
    [tp_file] = "filename.npy"                   # Total phosphorus
    [ap_file] = "filename.npy"                   # Available phosphorus
    [ip_file] = "filename.npy"                   # Inorganic phosphorus
    [op_file] = "filename.npy"                   # Organic phosphorus

COMMAND LINE USAGE:
    python preprocess_caete_pbz2.py --dataset "20CRv3-ERA5" --mode "spinclim"
    python preprocess_caete_pbz2.py --test                    # Validates output integrity
    python preprocess_caete_pbz2.py --mask-file custom.npy    # Override default mask

PERFORMANCE FEATURES:
    • Memory-efficient time-series processing using generators
    • Parallel file I/O (up to 256 threads) for gridcell data writing
    • Numba-accelerated mask-based data extraction
    • Automatic cleanup of previous runs (unless --test mode)

VALIDATION:
    Test mode randomly samples 3 gridcells and compares processed data against
    raw NetCDF files to ensure numerical accuracy (tests: hurs, tas, pr, ps, rsds, sfcwind).
"""
# ===============================

parser = argparse.ArgumentParser(
    description= __descr__,
    usage="python preprocess_caete_pbz2.py [-h] [--dataset DATASET] [--mode MODE]"
)

parser.add_argument('--dataset', type=str, default="20CRv3-ERA5", help='Main dataset folder (e.g., 20CRv3-ERA5)')
parser.add_argument('--mode', type=str, default="spinclim", help='Mode of the dataset, e.g., spinclim, transclim, etc.')
parser.add_argument('--mask-file', type=str, default=None, help="Path to the mask file. Default from configuration file [pre_process.toml]")
parser.add_argument('--test', action='store_true', help="Run a test to check if the data was correctly processed")


header = """CAETE-Copyright 2017- LabTerra
            This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU General Public License as published by
            the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.

            Pre-processing tool for input data preparation"""


#===============================
# ENVIRONMENT SETUP
#===============================
cc = type("PanAmazon", (), pan_amazon_region)() # Create a PanAmazon region object (contains ymin, ymax, xmin, xmax)
assert hasattr(cc, 'ymin') and hasattr(cc, 'ymax'), "Region must have ymin and ymax"
assert hasattr(cc, 'xmin') and hasattr(cc, 'xmax'), "Region must have xmin and xmax"
assert 0 <= cc.ymin < cc.ymax < 360, f"Invalid y coordinates: {cc.ymin}, {cc.ymax}"
assert 0 <= cc.xmin < cc.xmax < 720, f"Invalid x coordinates: {cc.xmin}, {cc.xmax}"

# Load configuration file
with open("./pre_processing.toml", 'rb') as f:
    # Works only with python 3.11 and above
    config_data = tomllib.load(f)

# St the climate data path and check if it exists
climate_data = Path(config_data["climate_data"])
assert climate_data.exists(), "Climate data folder does not exists"

# Parse the arguments
args = parser.parse_args()

# Set the global variables
dataset = args.dataset
mode = args.mode

# OUTPUT FILE WITH METADATA
metadata_filename_str = "METADATA.pbz2"

# NetCDF files with raw data to be processed
raw_data = climate_data / dataset / f"{mode}_raw"
if not raw_data.exists():
    raise FileNotFoundError(f"Raw data folder {raw_data} does not exists")

# INPUT FILES WITH SOIL DATA (NUTRIENTS)
soil_data = Path(config_data["soil_data"])
if not soil_data.exists():
    raise FileNotFoundError(f"Soil data folder {soil_data} does not exists")

# Load mask
if args.mask_file is None:
    mask_file = Path(config_data["mask_file"])
else:
    mask_file = Path(args.mask_file)

if not mask_file.exists():
    raise FileNotFoundError(f"Mask file {mask_file} does not exists")

mask = np.load(mask_file)
if mask.shape != (360, 720):
    raise ValueError(f"Mask file {mask_file} must have shape (360, 720), but has shape {mask.shape}")

# dump folder. CAETE input files are stored here
shared_data = Path(f"{dataset}/{mode}")
os.makedirs(shared_data, exist_ok=True)

# if there are files in the shared_data folder, remove them
if args.test:
    # We dont want to remove the files if we are testing
    pass
else:
    try:
        for file in shared_data.glob("*.pbz2"):
            os.remove(file)
        print("Removing old files from shared_data folder")
    except:
        # Skip if something goes wrong, with some info
        print("info: Could not remove old files from shared_data folder")
        pass

# ===============================
## ENVIRONMENT SETUP END
# ===============================

# ===============================
# FUNCTIONS & CLASSES
# ===============================

@njit
def get_values_at(array, mask):
    size = np.logical_not(mask).sum()
    out = np.zeros(size, dtype=np.float32)
    n = 0
    ny, nx = mask.shape
    for Y in range(ny):
        for X in range(nx):
            if not mask[Y][X]:
                out[n] = array[Y, X]
                n += 1
    return out

def read_clim_data(var:str) -> MFDataset | Dataset:
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

    # Correct the time index if it is not in the correct format

    return dataset

def get_dataset_size(dataset:MFDataset | Dataset) -> int:
    out = dataset.variables["time"][:].size
    return out

def process_climate_variable_vectorized(var: str, temp_mask):
    """
    VECTORIZED IMPLEMENTATION: Read all timesteps at once like preprocess_caete.py
    """
    print(f"Processing {var} (vectorized)...")

    with read_clim_data(var) as dataset:
        tsize = get_dataset_size(dataset)

        # Read ALL timesteps at once (vectorized)
        print(f"  Reading all {tsize} timesteps...")
        regional_data = dataset.variables[var][:, cc.ymin:cc.ymax, cc.xmin:cc.xmax]  # (time, y, x)

        # Extract all station data at once using vectorized indexing
        ny, nx = temp_mask.shape
        i_indices, j_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        valid_mask = ~temp_mask
        valid_i = i_indices[valid_mask]
        valid_j = j_indices[valid_mask]

        # Vectorized extraction: (time, station) -> (station, time)
        station_data = regional_data[:, valid_i, valid_j].T  # Transpose for pbz2 format

        print(f"  Extracted {len(valid_i)} stations x {tsize} timesteps")

    return station_data.filled(fill_value=station_data.mean())  # Fill NaNs for missing data

def read_soil_data(var):
    if var == 'tn':
        return np.load(os.path.join(soil_data, Path(config_data["tn_file"])))
    elif var == 'tp':
        return np.load(os.path.join(soil_data, Path(config_data["tp_file"])))
    elif var == 'ap':
        return np.load(os.path.join(soil_data, Path(config_data["ap_file"])))
    elif var == 'ip':
        return np.load(os.path.join(soil_data, Path(config_data["ip_file"])))
    elif var == 'op':
        return np.load(os.path.join(soil_data, Path(config_data["op_file"])))
    else:
        raise ValueError("Variable not found")

def process_gridcell(grd , var, data):
    grd.load()
    grd._load_dict(var, data)
    grd.write()


class ds_metadata:
    """ Helper to collect and save the netCDF files ancillary data"""


    def __init__(self):
        self.data = None
        self.ok = False
        self.fpath = None
        self.time = {"standard_name": None,
                     "units": None,
                     "calendar": None,
                     "time_index": None}

        self.lat = {"standard_name": None,
                    "units": None,
                    "axis": None,
                    "lat_index": None}

        self.lon = {"name": None,
                    "units": None,
                    "axis": None,
                    "lon_index": None}
        return None


    def fill_metadata(self, ds, ts=None):
        if ts is None:
            self.time["standard_name"] = ds.variables['time'].standard_name
            self.time["units"] = ds.variables['time'].units
            self.time["calendar"] = ds.variables['time'].calendar
            self.time["time_index"] = ds.variables['time'][:]
        else:
            self.time["standard_name"] = ts.standard_name
            self.time["units"] = ts.units
            self.time["calendar"] = ts.calendar
            self.time["time_index"] = ts[:]

        self.lat["standard_name"] = ds.variables['lat'].standard_name
        self.lat["units"] = ds.variables['lat'].units
        self.lat["axis"] = ds.variables['lat'].axis
        self.lat["lat_index"] = ds.variables['lat'][:]

        self.lon["standard_name"] = ds.variables['lon'].standard_name
        self.lon["units"] = ds.variables['lon'].units
        self.lon["axis"] = ds.variables['lon'].axis
        self.lon["lon_index"] = ds.variables['lon'][:]
        self.ok = True

        self.data = (self.time, self.lat, self.lon)


    def write(self, fpath):
        assert self.ok, 'Incomplte data, apply fill_metadata first'
        self.fpath = fpath
        with bz2.BZ2File(self.fpath, mode='w') as fh:
            pkl.dump(self.data, fh)


class input_data:
    """ Helper to transform and write data for CAETÊ input"""


    def __init__(self, y, x, dpath):
        self.y = y + cc.ymin
        self.x = x + cc.xmin
        self.filename = f"input_data_{self.y}-{self.x}.pbz2"
        self.dpath = Path(dpath)
        self.fpath = Path(os.path.join(self.dpath, self.filename))
        self.vars = ["hurs", "tas", "ps", "pr",
                     "rsds", "sfcwind", "tn", "tp", "ap", "ip", "op"]

        self.data = {"hurs": None,
                     "tas": None,
                     "ps": None,
                     "pr": None,
                     "rsds": None,
                     "sfcwind": None,
                     "tn": None,
                     "tp": None,
                     "ap": None,
                     "ip": None,
                     "op": None}

        self.cache = True if self.fpath.exists() else False
        self.loaded = True


    def _clean_memory(self):
        self.loaded = False
        self.data = {"hurs": None,
                     "tas": None,
                     "ps": None,
                     "pr": None,
                     "rsds": None,
                     "sfcwind": None,
                     "tn": None,
                     "tp": None,
                     "ap": None,
                     "ip": None,
                     "op": None}


    def _load_dict(self, var, DATA):
        assert var in self.vars, "Variable does not exists"
        # assert self.data[var] is None, "Variable already has data"
        self.data[var] = deepcopy(DATA)


    def load(self):
        # assert self.cache == True, "There is no cache data"
        assert self.dpath.exists()
        with bz2.BZ2File(self.fpath, mode='r') as fh:
            self.data = pkl.load(fh)
            self.loaded = True


    def write(self):
        assert self.loaded, "Data struct not loaded in memory for file write"
        assert self.dpath.exists()
        with bz2.BZ2File(self.fpath, mode='w') as fh:
            pkl.dump(self.data, fh)
            self.cache = True
            self.loaded = False
            self._clean_memory()

# ===============================
# END OF FUNCTIONS & CLASSES
# ===============================

def main():
    # SAVE METADATA
    variables = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']
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

    for x in range(1, len(times)):
        assert np.all(times[0][:] == times[x][:]), "Time values are not equal across datasets"
        assert times[0].calendar == times[x].calendar, "Calendars are not equal across datasets"
        assert times[0].units == times[x].units, "Units are not equal across datasets"
        # assert times[0].standard_name == times[x].standard_name, "Standard names are not equal across datasets"

    ancillary_data = ds_metadata()
    ancillary_data.fill_metadata(dss[0], ts=times[0])
    ancillary_data.write(shared_data / metadata_filename_str)
    # Close the datasets
    for ds in dss:
        ds.close()
    del dss

    # Prepare input templates
    input_templates = []
    ngrid = 0
    temp_mask = mask[cc.ymin: cc.ymax, cc.xmin:cc.xmax]
    ny, nx = temp_mask.shape
    for Y in range(ny):
        for X in range(nx):
            if not temp_mask[Y][X]:
                ngrid += 1
                input_templates.append(input_data(Y, X, shared_data))
    input_templates = np.array(input_templates, dtype=object)


    # Read Soil data
    tn_global = read_soil_data('tn') # Total Nitrogen
    tp_global = read_soil_data('tp') # Total Phosphorus
    ap_global = read_soil_data('ap') # Available Phosphorus
    ip_global = read_soil_data('ip') # Inorganic Phosphorus
    op_global = read_soil_data('op') # Organic Phosphorus
    # Slice soil data to regional extent
    tn = tn_global[cc.ymin: cc.ymax, cc.xmin:cc.xmax]
    tp = tp_global[cc.ymin: cc.ymax, cc.xmin:cc.xmax]
    ap = ap_global[cc.ymin: cc.ymax, cc.xmin:cc.xmax]
    ip = ip_global[cc.ymin: cc.ymax, cc.xmin:cc.xmax]
    op = op_global[cc.ymin: cc.ymax, cc.xmin:cc.xmax]

    # Load soil data
    for grd in input_templates:
        print(f"Processing soil data for gridcell {grd.y}-{grd.x}{' ' * 20}")
        local_y = grd.y - cc.ymin
        local_x = grd.x - cc.xmin
        total_nitrogen = tn[local_y, local_x]
        total_phosphorus = tp[local_y, local_x]
        available_phosphorus = ap[local_y, local_x]
        inorganic_phosphorus = ip[local_y, local_x]
        organic_phosphorus = op[local_y, local_x]

        assert total_nitrogen >= 0, "Total Nitrogen must be positive"
        assert total_phosphorus >= 0, "Total Phosphorus must be positive"
        assert available_phosphorus >= 0, "Available Phosphorus must be positive"
        assert inorganic_phosphorus >= 0, "Inorganic Phosphorus must be positive"
        assert organic_phosphorus >= 0, "Organic Phosphorus must be positive"

        grd._load_dict('tn', total_nitrogen)
        grd._load_dict('tp', total_phosphorus)
        grd._load_dict('ap', available_phosphorus)
        grd._load_dict('ip', inorganic_phosphorus)
        grd._load_dict('op', organic_phosphorus)
        grd.write()


    # VECTORIZED CLIMATE DATA PROCESSING
    print("Processing climate data with vectorized approach...")

    # Process all variables using vectorized approach
    climate_data = {}
    for var in variables:
        climate_data[var] = process_climate_variable_vectorized(var, temp_mask)

    # Write climate data to files in parallel
    print("\033[94m Writing climate data to files \033[0m")
    tasks = []

    # Collect all tasks
    for var in variables:
        for i, grd in enumerate(input_templates):
            tasks.append((grd, var, climate_data[var][i]))

    # Write data to files
    print(f"\033[94m  Writing... \033[0m{' ' * 20}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_gridcell, grd, var, data) for grd, var, data in tasks]
        concurrent.futures.wait(futures)

    print("Completed processing all climate variables")


def test(var, y=160, x=236, sample=500):
    # GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    CYAN = "\033[96m"

    # comapre raw data and processed data
    assert var in {'hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind'}, "Variable not found"
    assert shared_data.exists(), "Shared data folder does not exists"
    file_name = shared_data/Path(f"input_data_{y}-{x}.pbz2")
    assert file_name.exists(), "File not found"

    with bz2.BZ2File(file_name, mode='r') as fh:
        pbz2_data = pkl.load(fh)

    dss = read_clim_data(var)
    # get a slice of the data
    arr = dss.variables[var][:sample, y, x]
    saved_arr = pbz2_data[var][:sample]
    all_close=np.allclose(arr, saved_arr)
    mean_error = np.mean(np.abs(arr - saved_arr))
    dss.close()

    if all_close:
        print(f"{CYAN}Test for {var} PASSED with mean error {mean_error}{RESET}")
    else:
        print(f"{RED}Test for {var} FAILED with mean error {mean_error}{RESET}")


if __name__ == "__main__":
    print("\033[94m")
    print("\n", header, "\n")
    print("\033[0m")

    if args.test:
        print(f"Testing dataset: {dataset}, Mode: {mode}")
        # Collect the indices of the input_data files to perform the test
        files = list(shared_data.glob("input_data_*-*.pbz2"))
        assert len(files) > 0, "No input_data files found"
        size_files = len(files)
        indices = []
        for file in files:
            indices.append(file.stem.split("_")[-1].split("-"))
        indices = list(map(lambda x: (int(x[0]), int(x[1])), indices))

        # Select a random sample of indices to test
        idx = np.random.randint(0, size_files, min(5, size_files - 1))
        indices = [indices[i] for i in idx]
        tested = [files[i] for i in idx]
        print(f"Testing the following indices: {indices}")
        for var in ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']:
            for y, x in indices:
                print(f"Testing {var} for gridcell {y}-{x}")
                test(var, y=y, x=x)
            print(f"Tested files {tested}\n\n")
    else:
        print("\nProcessing details:")
        print("\033[91m")
        print(f"Dataset: {dataset}, Mode: {mode}")
        print(f"Processing soil data from {soil_data}")
        print(f"We are processing data from {raw_data}")
        print(f"Mask file loaded from {mask_file}")
        print(f"The processed data will be stored in {shared_data}\n")
        print("\033[0m")
        main()
