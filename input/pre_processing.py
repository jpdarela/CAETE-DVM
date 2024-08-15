from pathlib import Path

import argparse
import bz2
import concurrent.futures
import os
import pickle as pkl
# import queue
# import threading

from netCDF4 import MFDataset
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

                The script can be run from the command line or from an ipython environment.

                The dataset is given by the --dataset argument

                The mode (spinclim, transclim, etc.) is given by the --mode argument
                It sets the name of the folder where the data will be stored. The raw data downloaded
                from the ISIMIP data repository must be found in the folder named as dataset/<mode>_raw.
                The soil data must be found in the soil folder.

                The mask file is given by the --mask-file argument

                Both for the mask file and the soil data, the data must be in numpy format.
                Examples of the mask file and the soil data are available in the mask and soil folders.
                They are used as default values for the respective arguments. You can check the file formats
                necessary for the mask and soil data by looking at it.

                The variables from the ISIMIP dataset to be processed are: hurs, tas, pr, ps, rsds, sfcwind
                The soil data variables are: tn, tp, ap, ip, op (Total Nitrogen, Total Phosphorus, Available Phosphorus,
                Inorganic Phosphorus, Organic Phosphorus)

                Look at the README.md file for more information.

                """

parser = argparse.ArgumentParser(
    description= __descr__,
    usage="python pre_processing.py [-h] [--dataset DATASET] [--mode MODE] \n from ipython: run pre_processing.py --dataset DATASET --mode MODE"
)

parser.add_argument('--dataset', type=str, default="20CRv3-ERA5", help='Main dataset folder (e.g., 20CRv3-ERA5)')
parser.add_argument('--mode', type=str, default="spinclim", help='Mode of the dataset, e.g., spinclim, transclim, etc.')
parser.add_argument('--mask-file', type=str, default="./mask/mask_raisg-360-720.npy",
                        help="Path to the mask file (default: ./mask/mask_raisg-360-720.npy)")

# Parse the arguments
args = parser.parse_args()

# Set the global variables
dataset = args.dataset
mode = args.mode

# OUTPUT FILE WITH METADATA
ANCILLARY_OUTPUT = "ISIMIP_HISTORICAL_METADATA.pbz2"

# dump folder. CAETE input files are stored here
shared_data = Path(f"{dataset}/{mode}")
assert shared_data.exists(), "Shared data folder does not exists"

# NerCDF files with raw data to be processed
raw_data = Path(f"{dataset}/{mode}_raw")
assert raw_data.exists(), "Raw data folder does not exists"

# INPUT FILES WITH SOIL DATA (NUTRIENTS)
soil_data = Path("./soil")
assert soil_data.exists(), "Soil data folder does not exists"

# Load Pan Amazon mask
mask = np.load(Path(args.mask_file))


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


def read_clim_data(var:str) -> MFDataset:
    try:
        files = raw_data.glob(f"*{var}*")
    except:
        raise FileNotFoundError(f"No netCDF file for variable {var} in {raw_data}")

    try:
        dataset = MFDataset(list(files))
    except:
        raise FileNotFoundError(f"No netCDF file for variable {var} in {raw_data}")

    return dataset


def get_dataset_size(dataset):
    out = dataset.variables["time"][:].size
    return out


def _read_clim_data_(var:str) -> Generator[NDArray, None, None]:
    with read_clim_data(var) as dataset:
        try:
            zero_dim = get_dataset_size(dataset)
        except:
            raise ValueError("Cannot get dataset size")

        for i in range(zero_dim):
            yield dataset.variables[var][i, :, :]


def read_soil_data(var):
    if var == 'tn':
        return np.load(os.path.join(soil_data, Path('total_n_PA.npy')))
    elif var == 'tp':
        return np.load(os.path.join(soil_data, Path('total_p.npy')))
    elif var == 'ap':
        return np.load(os.path.join(soil_data, Path('avail_p.npy')))
    elif var == 'ip':
        return np.load(os.path.join(soil_data, Path('inorg_p.npy')))
    elif var == 'op':
        return np.load(os.path.join(soil_data, Path('org_p.npy')))
    else:
        raise ValueError("Variable not found")


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


def process_gridcell(grd:input_data , var, data):
    print(f"Processing {var} for gridcell {grd.y}-{grd.x}      ", end="\r")
    grd.load()
    grd._load_dict(var, data)
    grd.write()


def main():
    # SAVE METADATA
    variables = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']

    dss = [read_clim_data(var) for var in variables]
    ancillary_data = ds_metadata(dss)
    ancillary_data.fill_metadata(dss[0])
    ancillary_data.write(shared_data / ANCILLARY_OUTPUT)

    for ds in dss:
        ds.close()
    del dss

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
        print(f"Processing soil data for gridcell {grd.y}-{grd.x}       ", end="\r")
        grd._load_dict('tn', tn[grd.y, grd.x].copy(order="F"))
        grd._load_dict('tp', tp[grd.y, grd.x].copy(order="F"))
        grd._load_dict('ap', ap[grd.y, grd.x].copy(order="F"))
        grd._load_dict('ip', ip[grd.y, grd.x].copy(order="F"))
        grd._load_dict('op', op[grd.y, grd.x].copy(order="F"))
        grd.write()

    # Load clim_data and write to input templates
    array_data = []
    variables = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']
    for var in variables:
        tsize = get_dataset_size(read_clim_data(var))
        data = np.zeros((ngrid, tsize), dtype=np.float32)

        j = 0

        for arr in _read_clim_data_(var):
            print(f"Processing {var} data for day {j}      ", end="\r")
            data[:, j] = get_values_at(arr.data)
            j += 1
        array_data.append(data)

    data_dict = dict(zip(variables, array_data))

    for var, data in data_dict.items():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_gridcell, grd, var, data[i]) for i, grd in enumerate(input_templates)]
            concurrent.futures.wait(futures)

## TODO: Improve the test function and add an argument to run it
def test(var, y=160, x=236):
    # comapre raw data and processed data
    with bz2.BZ2File(f"./20CRv3-ERA5/spinclim/input_data_{y}-{x}.pbz2", mode='r') as fh:
        pbz2_data = pkl.load(fh)

    dss = read_clim_data(var)
    # get a slice of the data
    arr = dss.variables[var][:500, y, x]
    saved_arr = pbz2_data[var][:500]
    print(np.allclose(arr, saved_arr))


if __name__ == "__main__":
    main()
    # # A small test to check if the data was correctly processed
    # for var in ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']:
    #     test(var)
