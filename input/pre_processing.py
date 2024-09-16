from pathlib import Path

import argparse
import bz2
import concurrent.futures
import os
import sys
import pickle as pkl
import tomllib

from netCDF4 import Dataset, MFDataset # type: ignore
from numba import njit
from numpy.typing import NDArray
from typing import Generator

import numpy as np

__what__ = "Pre-processing of input data to feed CAETÊ"
__author__ = "jpdarela"
__date__ = "Mon Dec 28 18:08:27 -03 2020"
__descr__ = """ This script reads the raw data from the 20CRv3-ERA5 or any other
                ISIMIP climate forcing dataset (0.5 degrees resolution) and reorganize the
                data to feed the CAETÊ model. This script also reads the soil data and save
                it along with the climate data. The script assumes that the raw data is
                already downloaded and the soil data is available in the soil folder.
                It also assumes that the mask file is available in the mask folder.
                The processed data is stored in the shared_data folder.
                This program also saves the metadata of the processed data in a compressed file named
                as ISIMIP_HISTORICAL_METADATA.pbz2 in the shared_data folder.

                IT can be run from the command line or from an ipython environment.

                The dataset is given by the --dataset argument

                The mode (spinclim, transclim, etc.) is given by the --mode argument
                It sets the name of the folder where the data will be stored. The raw data downloaded
                from the ISIMIP data repository must be found in the folder named as dataset/<mode>_raw.
                The soil data must be found in the soil folder.

                The mask file is given by the --mask-file argument

                Both for the mask file and the soil files, the data must be in numpy format.
                Examples of the mask file and the soil data are available in the mask and soil folders.
                Currently, all files represents the data for the entire globe in a 360x720 grid.
                The mask file is a numpy array with boolean values. The soil data are numpy arrays with
                float32 values.

                You can change the mask file and the soil data folders (also the filenames of the soil data)
                by editing the pre_processing.toml file.

                You can have a look at the parameters.py file in the source folder.
                There are extra numpy arrays defining soil hydrological parameters that are read
                directly from the files during model execution. You can check the file formats
                necessary for the mask and soil input data by looking at it.

                The variables from the ISIMIP dataset to be processed are: hurs, tas, pr, ps, rsds, sfcwind
                The soil data variables are: tn, tp, ap, ip, op (Total Nitrogen, Total Phosphorus, Available Phosphorus,
                Inorganic Phosphorus, Organic Phosphorus)

                Look at the README.md file for more information.

                """

parser = argparse.ArgumentParser(
    description= __descr__,
    usage="python pre_processing.py [-h] [--dataset DATASET] [--mode MODE] \n\t from ipython: run pre_processing.py --dataset DATASET --mode MODE"
)

parser.add_argument('--dataset', type=str, default="20CRv3-ERA5", help='Main dataset folder (e.g., 20CRv3-ERA5)')
parser.add_argument('--mode', type=str, default="spinclim", help='Mode of the dataset, e.g., spinclim, transclim, etc.')
parser.add_argument('--mask-file', type=str, default="./mask/mask_raisg-360-720.npy",
                        help="Path to the mask file (default: ./mask/mask_raisg-360-720.npy)")
parser.add_argument('--test', action='store_true', help="Run a test to check if the data was correctly processed")


header = """CAETE-Copyright 2017- LabTerra
            This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU General Public License as published by
            the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.

            Pre-processing tool for input data preparation"""

with open("./pre_processing.toml", 'rb') as f:
    # Works only with python 3.11 and above
    config_data = tomllib.load(f)

# Parse the arguments
args = parser.parse_args()

# Set the global variables
dataset = args.dataset
mode = args.mode

# OUTPUT FILE WITH METADATA
metadata_filename_str = "ISIMIP_HISTORICAL_METADATA.pbz2"

# NerCDF files with raw data to be processed
raw_data = Path(f"{dataset}/{mode}_raw")
assert raw_data.exists(), "Raw data folder does not exists"

# INPUT FILES WITH SOIL DATA (NUTRIENTS)
soil_data = Path(config_data["soil_nutrients_data"])
assert soil_data.exists(), "Soil data folder does not exists"

# Load mask
mask_file = Path(args.mask_file)
assert mask_file.exists(), "Mask file does not exists"
mask = np.load(mask_file)

# dump folder. CAETE input files are stored here
shared_data = Path(f"{dataset}/{mode}")
os.makedirs(shared_data, exist_ok=True)
assert shared_data.exists(), "Shared data folder does not exists"

# if there are files in the shared_data folder, remove them
if args.test:
    # We dont want to remove the files if we are testing
    pass
else:
    try:
        for file in shared_data.glob("*"):
            os.remove(file)
        print("Removing old files from shared_data folder")
    except:
        # Skip if something goes wrong, with some info
        print("info: Could not remove old files from shared_data folder")
        pass

# Timer wrapper
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        hours = (end - start) // 3600
        minutes = round(((end - start) % 3600) // 60)
        seconds = round((end - start) % 60)
        if hours == 0:
            print(f"Elapsed time: {minutes}:{seconds}")
        elif minutes == 0:
            print(f"Elapsed time: {seconds} seconds")
        else:
            print(f"Elapsed time: {hours}:{minutes}:{seconds}")
    return wrapper


def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=30):
    """FROM Stack Overflow/GIST, THANKS
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    RESET = "\033[0m"
    CYAN = "\033[96m"
    blue_bar = f"{CYAN}{'█'}{RESET}"
    bar_utf = b'\xe2\x96\x88'  # bar -> unicode symbol = u'\u2588'
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = blue_bar * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r\033[95m%s\033[0m \033[91m|%s|\033[0m \033[94m%s%s\033[0m %s' %
                     (prefix, bar, percents, '%', suffix)), # type: ignore

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


@njit
def get_values_at(array, mask=mask):
    size = np.logical_not(mask).sum()
    out = np.zeros(size, dtype=np.float32)
    n = 0
    for Y in range(360):
        for X in range(720):
            if not mask[Y][X]:
                out[n] = array[Y, X]
                n += 1
    return out


class ds_metadata:
    """ Helper to collect and save the netCDF files ancillary data"""

    def __init__(self, dsets):
        self.ds_dict = [ds.__dict__ for ds in dsets]
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

    def fill_metadata(self, ds):
        self.time["standard_name"] = ds.variables['time'].standard_name
        self.time["units"] = ds.variables['time'].units
        self.time["calendar"] = ds.variables['time'].calendar
        self.time["time_index"] = ds.variables['time'][:]

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
        assert self.ok, 'INcomplte data, apply fill_metadata first'
        self.fpath = fpath
        with bz2.BZ2File(self.fpath, mode='w') as fh:
            pkl.dump(self.data, fh)


class input_data:
    """ Helper to transform and write data for CAETÊ input"""

    def __init__(self, y, x, dpath):
        self.y = y
        self.x = x
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
        self.data[var] = DATA

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

    return dataset


def get_dataset_size(dataset:MFDataset | Dataset) -> int:
    out = dataset.variables["time"][:].size
    return out


def _read_clim_data_(var:str) -> Generator[NDArray, None, None]:
    with read_clim_data(var) as dataset:
        try:
            zero_dim = get_dataset_size(dataset)
        except:
            raise ValueError("Cannot get dataset size")

        for i in range(zero_dim):
            yield dataset.variables[var][i, :, :].data


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


def process_gridcell(grd:input_data , var, data):
    grd.load()
    grd._load_dict(var, data)
    grd.write()


# def process_data(j, hurs, tas, pr, ps, rsds, sfcwind):
#     return {
#         "j": j,
#         "hurs": get_values_at(hurs),
#         "tas": get_values_at(tas),
#         "pr": get_values_at(pr),
#         "ps": get_values_at(ps),
#         "rsds": get_values_at(rsds),
#         "sfcwind": get_values_at(sfcwind)
#     }


@timer
def main():
    # SAVE METADATA
    variables = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']
    dss = [read_clim_data(var) for var in variables]
    ancillary_data = ds_metadata(dss)
    ancillary_data.fill_metadata(dss[0])
    ancillary_data.write(shared_data / metadata_filename_str)
    # Close the datasets
    for ds in dss:
        ds.close()
    del dss

    # Prepare input templates
    input_templates = []
    ngrid = 0
    for Y in range(360):
        for X in range(720):
            if not mask[Y][X]:
                ngrid += 1
                input_templates.append(input_data(Y, X, shared_data))
    input_templates = np.array(input_templates, dtype=object)

    # Read Soil data
    tn = read_soil_data('tn') # Total Nitrogen
    tp = read_soil_data('tp') # Total Phosphorus
    ap = read_soil_data('ap') # Available Phosphorus
    ip = read_soil_data('ip') # Inorganic Phosphorus
    op = read_soil_data('op') # Organic Phosphorus

    # Load soil data
    for grd in input_templates:
        print(f"Processing soil data for gridcell {grd.y}-{grd.x}{' ' * 20}", end="\r")
        total_nitrogen = tn[grd.y, grd.x].copy(order="F")
        total_phosphorus = tp[grd.y, grd.x].copy(order="F")
        available_phosphorus = ap[grd.y, grd.x].copy(order="F")
        inorganic_phosphorus = ip[grd.y, grd.x].copy(order="F")
        organic_phosphorus = op[grd.y, grd.x].copy(order="F")

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

    # Load clim_data and write to input templates
    array_data = []
    variables = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']

    tsize = get_dataset_size(read_clim_data(variables[0])) # all datasets have the same size

    _data = dict(zip(variables, [np.zeros((ngrid, tsize), dtype=np.float32) for _ in range(len(variables))]))

    hurs_gen = _read_clim_data_('hurs')
    tas_gen = _read_clim_data_('tas')
    pr_gen = _read_clim_data_('pr')
    ps_gen = _read_clim_data_('ps')
    rsds_gen = _read_clim_data_('rsds')
    sfcwind_gen = _read_clim_data_('sfcwind')

    i = 0
    print(f"Reading data: {variables}{'' * 20}")
    print_progress(i, tsize, prefix='Reading data:', suffix='Complete')
    for hurs, tas, pr, ps, rsds, sfcwind, j in zip(hurs_gen, tas_gen, pr_gen, ps_gen, rsds_gen, sfcwind_gen, range(tsize)):
        _data["hurs"][:, j] = get_values_at(hurs)
        _data["tas"][:, j] = get_values_at(tas)
        _data["pr"][:, j] = get_values_at(pr)
        _data["ps"][:, j] = get_values_at(ps)
        _data["rsds"][:, j] = get_values_at(rsds)
        _data["sfcwind"][:, j] = get_values_at(sfcwind)
        print_progress(i+1, tsize, prefix='Reading data:', suffix='Complete')
        i += 1
    # i = 0
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(process_data, j, hurs, tas, pr, ps, rsds, sfcwind)
    #         for hurs, tas, pr, ps, rsds, sfcwind, j in zip(hurs_gen, tas_gen, pr_gen, ps_gen, rsds_gen, sfcwind_gen, range(tsize))
    #         ]

    #     print(f"Reading data: {variables}{'' * 20}")
    #     print_progress(i, tsize, prefix='Reading data:', suffix='Complete')
    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #         j = result["j"]
    #         _data["hurs"][:, j] = result["hurs"]
    #         _data["tas"][:, j] = result["tas"]
    #         _data["pr"][:, j] = result["pr"]
    #         _data["ps"][:, j] = result["ps"]
    #         _data["rsds"][:, j] = result["rsds"]
    #         _data["sfcwind"][:, j] = result["sfcwind"]
    #         i += 1
    #         print_progress(i, tsize, prefix='Reading data:', suffix='Complete')

    array_data = _data["hurs"], _data["tas"], _data["pr"], _data["ps"], _data["rsds"], _data["sfcwind"]

    print("\033[94m Writing data to files \033[0m")
    # Write data to files in parallel
    tasks = []
    # Collect all tasks
    for var, data in zip(variables, array_data):
        for i, grd in enumerate(input_templates):
            tasks.append((grd, var, data[i]))

    # Write data to files
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(process_gridcell, grd, var, data) for grd, var, data in tasks]
        concurrent.futures.wait(futures)

    # for var, data in zip(variables, array_data):
    #     print(f"Writing \033[94m{var}\033[0m{' ' * 20}", end="\r")
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #         futures = [executor.submit(process_gridcell, grd, var, data[i]) for i, grd in enumerate(input_templates)]
    #         concurrent.futures.wait(futures)


def test(var, y=160, x=236, sample=500):
    GREEN = "\033[92m"
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
        idx = np.random.randint(0, size_files, 3)
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
