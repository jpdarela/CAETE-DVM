import os
from pathlib import Path
import _pickle as pkl
import bz2
import numpy as np
from netCDF4 import MFDataset

__wat__ = "Pre-processing of input data to feed CAETÊ"
__author__ = "jpdarela"
__date__ = "Mon Dec 28 18:08:27 -03 2020"
__descr__ = """ This script works in the folowing manner: Given a directory (raw_data) with
                input climatic data in the form of netCDF files the script opens these files as
                MFDataset objects. THen the metadata of the climatic data is compiled from the source
                files and writen to a file in the output folder (clim_data). This folder will store
                the files with input data for each gridcell for all climatic and soil variables. Each
                gridcell will have one file with all variables for the entire timespan covered by the
                netCDF files.
                """

# GLOBAL VARIABLES (paths)

CLIMATIC_DATA = "historical_ISIMIP-v3"
ANCILLARY_OUTPUT = "ISIMIP_HISTORICAL_METADATA.pbz2"


# FOLDER IN THE SERVER WHERE ALL data IS stored FOR ALL USERS
shared_data = Path("/home/amazonfaceme/shared_data/")

# INPUT NETCDF FILES WITH historical CLIMATIC DATA
raw_data = Path(os.path.join(
    shared_data, Path(CLIMATIC_DATA)))

# INPUT FILES WITH SOIL DATA (NUTRIENTS)
soil_data = Path(os.path.join(shared_data, "soil"))

# OUTPUT FOLDER - WILL STORE THE DATA THAT WILL RUN CAETÊ
clim_data = Path(os.path.join(shared_data, "HISTORICAL-RUN"))

# Load Pan Amazon mask
mask = np.load(os.path.join(
    shared_data, Path("mask/mask_raisg-360-720.npy")))


# Classes and functions to help data transformation


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
                     "rsds", "tn", "tp", "ap", "ip", "op"]
        self.data = {"hurs": None,
                     "tas": None,
                     "ps": None,
                     "pr": None,
                     "rsds": None,
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
                     "tn": None,
                     "tp": None,
                     "ap": None,
                     "ip": None,
                     "op": None}

    def _load_dict(self, var, DATA):
        assert var in self.vars, "Variable does not exists"
        assert self.data[var] is None, "Variable already sat"
        self.data[var] = DATA

    def load(self):
        assert self.cache == True, "There is no cache data"
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


# HElpers to open and load clmatic and soil datasets
def read_clim_data(var):

    if var == 'hurs':
        ds_hurs = MFDataset(os.path.join(raw_data, "hurs_*.nc4"))
        dt = ds_hurs.variables['hurs'][:]
        no_data = ds_hurs.variables['hurs'].missing_value
        ds_hurs.close()
        return dt, no_data

    elif var == 'tas':
        ds_tas = MFDataset(os.path.join(raw_data, "tas_*.nc4"))
        dt = ds_tas.variables['tas'][:]
        no_data = ds_tas.variables['tas'].missing_value
        ds_tas.close()
        return dt, no_data

    elif var == 'pr':
        ds_pr = MFDataset(os.path.join(raw_data, "pr_*.nc4"))
        dt = ds_pr.variables['pr'][:]
        no_data = ds_pr.variables['pr'].missing_value
        ds_pr.close()
        return dt, no_data

    elif var == 'ps':
        ds_ps = MFDataset(os.path.join(raw_data, "ps_*.nc4"))
        dt = ds_ps.variables['ps'][:]
        no_data = ds_ps.variables['ps'].missing_value
        ds_ps.close()
        return dt, no_data

    elif var == 'rsds':
        ds_rsds = MFDataset(os.path.join(raw_data, "rsds_*.nc4"))
        dt = ds_rsds.variables['rsds'][:]
        no_data = ds_rsds.variables['rsds'].missing_value
        ds_rsds.close()
        return dt, no_data

# Open soil Stuff


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


def main():
    # SAVE METADATA
    dss = (MFDataset(os.path.join(raw_data, "hurs_*.nc4")),
           MFDataset(os.path.join(raw_data, "tas_*.nc4")),
           MFDataset(os.path.join(raw_data, "pr_*.nc4")),
           MFDataset(os.path.join(raw_data, "ps_*.nc4")),
           MFDataset(os.path.join(raw_data, "rsds_*.nc4")))

    ancillary_data = ds_metadata(dss)
    ancillary_data.fill_metadata(dss[0])
    ancillary_data.write(os.path.join(
        clim_data, ANCILLARY_OUTPUT))

    for ds in dss:
        ds.close()
    del dss

    # Create input templates
    input_templates = []
    # Check outputs dir
    dir_check = True if clim_data.exists() else os.mkdir(clim_data)

    for Y in range(360):
        for X in range(720):
            if not mask[Y][X]:
                input_templates.append(input_data(Y, X, clim_data))
    input_templates = np.array(input_templates, dtype=object)

    # HURS & soil:
    hurs, no_data = read_clim_data('hurs')
    tn = read_soil_data('tn')
    tp = read_soil_data('tp')
    ap = read_soil_data('ap')
    ip = read_soil_data('ip')
    op = read_soil_data('op')

    for grd in input_templates:
        grd._load_dict('hurs', hurs[:, grd.y, grd.x].data.copy(order="F"))
        grd._load_dict('tn', tn[grd.y, grd.x].copy(order="F"))
        grd._load_dict('tp', tp[grd.y, grd.x].copy(order="F"))
        grd._load_dict('ap', ap[grd.y, grd.x].copy(order="F"))
        grd._load_dict('ip', ip[grd.y, grd.x].copy(order="F"))
        grd._load_dict('op', op[grd.y, grd.x].copy(order="F"))

        hurs[:, grd.y, grd.x] = no_data
        grd.write()

    del tn
    del tp
    del ap
    del ip
    del op
    del hurs

    tas, no_data = read_clim_data('tas')
    for grd in input_templates:
        grd.load()
        grd._load_dict('tas', tas[:, grd.y, grd.x].data.copy(order="F"))
        tas[:, grd.y, grd.x] = no_data
        grd.write()
    del tas

    pr, no_data = read_clim_data('pr')
    for grd in input_templates:
        grd.load()
        grd._load_dict('pr', pr[:, grd.y, grd.x].data.copy(order="F"))
        pr[:, grd.y, grd.x] = no_data
        grd.write()
    del pr

    ps, no_data = read_clim_data('ps')
    for grd in input_templates:
        grd.load()
        grd._load_dict('ps', ps[:, grd.y, grd.x].data.copy(order="F"))
        ps[:, grd.y, grd.x] = no_data
        grd.write()
    del ps

    rsds, no_data = read_clim_data('rsds')
    for grd in input_templates:
        grd.load()
        grd._load_dict('rsds', rsds[:, grd.y, grd.x].data.copy(order="F"))
        rsds[:, grd.y, grd.x] = no_data
        grd.write()
    del rsds


if __name__ == "__main__":
    main()
