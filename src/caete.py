# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

DESCRIPTION = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """

# """_summary_
# This module contains the classes that define the gridcell and the region objects.
# The gridcell object is the basic unit of the simulation. It contains the data and the methods to
# run the simulation for a single gridcell. The region object represents is a collection of gridcells.
# It contains the data for a simulation and also provides  the methods to run the simulation
# for a collection of gridcells in parallel (multiprocessing).

# THe architecture of the code is defined by the following classes:
# - state_zero: base class with input/output related data (paths, filenames, etc)
# - climate: class with climate data
# - time: class with time data
# - soil: class with soil data
# - gridcell_output: class to manage gridcell outputs

#  All these classes also have some particular methods to manage the data
#  They are isolated in these classes to make the code more readable and maintainable.
#  All the above classes are used as base for the class that regresents a gricell
#  in the simulation:

# - grd_mt: class to manage the gridcell simulation
#     It has the methods to set up, run the simulation and save the outputs in a
#     folder. Each gridcell has one plant metacommunity. The plant metacommunity is a collection of
#     plant communities. Each plant community is a collection of plant life strategies (PLS).
#     At creation time, a community receives a sample of PLS from a global table. A community is not
#     allowed to have duplicated PLS but the same PLS can be in more than one community in a
#     metacommunity. Both the number of communities and the number of PLS in each community
#     are defined by the user in the configuration file called caete.toml. The number of
#     communities can be changed freely between different simulations. However, If you want
#     to change the number of PLS in a community, you must not only change the configuration
#     in the caete.toml file but also recompile the shared library using gnu-make or nmake.
#     The shared library here is the fortran code that we use to run the "daily processes".

#     The global PLS table is a collection of PLS that is shared by all gridcells in the region. There is a
#     script called plsgen.py that can create a global PLS table. Run it in the src folder like this:

#     ```$ python plsgen.py -n 25000 -f ./MyPLSDataFolder```

#     This will save a file named pls_attrs<n>.csv (where n = 25000) in the local folder MyPLSDataFolder.
#     The script uses data from literature to generate a quasi-random sample of PLS based on 17 plant
#     functional traits. There is a plsgen.toml file that contains some of the parameters used
#     to generate the PLS table. There are comments in the script. Pls look.

#     During the simulation of a gridcell, it is possible to control the number of PLS in a community.
#     You can reset a community to a new set of PLS. This is useful when the set of PLS initially designated
#     to a community is not able to accumulate biomass in the initial stage of the simulation.
#     It is also possible to reset the entire metacommunity at once. This is useful to reset the
#     distributions of PLS after a initial spinup aiming to reach a stable state in the soil pools.
#     Right after this soil spinup, it is possible to seed new PLS in the communities while runing the
#     model. This is useful to simulate the filtering of the gridcell "environment" along time while adding new
#     PLS to the communities.
#     Given that our sample space is cursed 17 times (17 traits), these strategies are important do start
#     the model (spin it up) and to keep it running while simulating the filtering of the
#     gridcell "environment". The grd_mt.run_gridcell method has several parameters to control
#     the simulation. You can have a look at the method signature and docstring to understand the options.

# Finally, the region class is defined. It represents collection of gridcells. It has the methods to
# run the simulation for a collection of gridcells in parallel (multiprocessing).

# - region: class to manage the region simulation, the global PLS table, IO, multiprocessing etc.

# - worker (only @staticmethods): class grouping worker functions defining different phases of simulation.
#   These functions are called by the region object to run the simulation in parallel. The worker class
#   also have some utility functions to load and save data. You can save a region in a state file and
#   restart the simulation from this point. Note that in this case a entire region is saved. All relatred data
#   for each gridcell is saved and the final file can become huge. This tradeoff with the facitily to restart
#   the simulation from a specific point with a very low amount of code. The state file is compressed with zsdt

#   I am testing the model with a script called test.py. This script is a good example of how to use the model.
#   In the end of this source file (caete.py) there is some code used to test/profile the python code.

#   The old implementation of the model is in the classes grd and plot at the end of this source file.
# """

import bz2
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import gc
import multiprocessing as mp
import os
import pickle as pkl
import random as rd
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Collection, Set

import cftime
import numba
import numpy as np
from joblib import dump, load
from numpy.typing import NDArray
import zstandard as zstd

from _geos import calculate_area, find_coordinates_xy, find_indices, find_indices_xy
from config import Config, fetch_config, fortran_runtime
from hydro_caete import soil_water
import metacommunity as mc
from output import budget_output

# Tuples with hydrological parameters for the soil water calculations
from parameters import hsoil, ssoil, tsoil
from parameters import output_path

# This code is only relevant in Windows systems. It adds the fortran compiler dlls to the PATH
# so the shared library can find the fortran runtime libraries of the intel one API compiler (ifx)
# Note: This is only necessary in Windows systems
if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_runtime)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

# shared library
from caete_module import budget as model
from caete_module import global_par as gp
from caete_module import photo as m
from caete_module import soil_dec
from caete_module import water as st

from memory_profiler import profile

# Global lock. Used to lock the access to the main table of Plant Life Strategies
lock = mp.Lock()

# Define a type for nested configuration data
config_type: Union[Config, int, float, str]

warnings.simplefilter("default")

# Define some util functions #
def rwarn(txt:str='RuntimeWarning'):
    """Raise a RuntimeWarning"""
    warnings.warn(f"{txt}", RuntimeWarning)

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
    bar_utf = b'\xe2\x96\x88'  # bar -> unicode symbol = u'\u2588'
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)), # type: ignore

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def budget_daily_result(out: Tuple[Union[NDArray, str, List, int, float]]) -> budget_output:
    """_summary_

    Args:
        out (Tuple[Union[NDArray, str, List, int, float]]): _description_

    Returns:
        budget_output: _description_
    """
    return budget_output(*out)


def catch_out_budget(out: Tuple[Union[NDArray, str, List, int, float]]) -> Dict[str, Union[NDArray, str, List, int, float]]:
    """_summary_

    Args:
        out (Tuple[Union[NDArray, str, List, int, float]]): _description_

    Returns:
        Dict[str, Union[NDArray, str, List, int, float]]: _description_
    """
    # This is currently used in the old implementation (classes grd and plot)
    # WARNING keep the lists of budget/carbon3 outputs updated with fortran code

    lst = ["evavg", "epavg", "phavg", "aravg", "nppavg",
           "laiavg", "rcavg", "f5avg", "rmavg", "rgavg", "cleafavg_pft", "cawoodavg_pft",
           "cfrootavg_pft", "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg", "vcmax",
           "specific_la", "nupt", "pupt", "litter_l", "cwd", "litter_fr", "npp2pay", "lnc",
           "limitation_status", "uptk_strat", 'cp', 'c_cost_cwm', "rnpp_out"]

    return dict(zip(lst, out))

def catch_out_carbon3(out: Tuple[Union[NDArray, str, List, int, float]]) -> Dict:
    """_summary_

    Args:
        out (Tuple[Union[NDArray, str, List, int, float]]): _description_

    Returns:
        Dict: _description_
    """
    lst = ['cs', 'snc', 'hr', 'nmin', 'pmin']

    return dict(zip(lst, out))

def str_or_path(fpath: Union[Path, str], check_exists:bool=True,
                check_is_dir:bool=False, check_is_file:bool=False) -> Path:

    """Converts fpath to a Path object if necessay, do some checks and return the Path object"""

    is_path = isinstance(fpath, (Path))
    is_str = isinstance(fpath, (str))
    is_str_or_path = is_str or is_path

    assert is_str_or_path, "fpath must be a string or a Path object"
    _val_ = fpath if is_path else Path(fpath)

    if check_exists:
        assert _val_.exists(), f"File/directory not found: {_val_}"
    if check_is_dir:
        assert not check_is_file, "Cannot check if a path is a file and a directory at the same time"
        assert _val_.is_dir(), f"Path is not a directory: {_val_}"
    if check_is_file:
        assert not check_is_dir, "Cannot check if a path is a file and a directory at the same time"
        assert _val_.is_file(), f"Path is not a file: {_val_}"

    return _val_

def get_co2_concentration(filename:Union[Path, str]):
    """_summary_

    Args:
        filename (Union[Path, str]): _description_

    Returns:
        _type_: _description_
    """
    fname = str_or_path(filename, check_is_file=True)
    with open(fname, 'r') as file:
        # Use the Sniffer class to detect the dialect
        sample = file.read(1024)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        file.seek(0)
        data = list(csv.reader(file, dialect))
    if sniffer.has_header:
        data = data[1:]
    return dict(map(lambda x: (int(x[0]), float(x[1])), data))

def read_bz2_file(filepath:Union[Path, str]):
    fpath = str_or_path(filepath)
    with bz2.BZ2File(fpath, mode='r') as fh:
        data = pkl.load(fh)
    return data

def parse_date(date_string):
    """_summary_

    Args:
        date_string (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    formats = ['%Y%m%d', '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d']
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return cftime.real_datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        except ValueError:
            pass
    raise ValueError('No valid date format found')

# These functions are JIT compiled and cached by numba.
# If you change any of the cached functions, you should delete the cache
# folder in the src folder, generally named __pycache__. This will force numba
# to recompile the functions and cache them again.
@numba.jit(nopython=True, cache=True)
def neighbours_index(pos: Union[List, NDArray], matrix: NDArray) -> List:
    neighbours = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for i in range(max(0, pos[0] - 1), min(rows, pos[0] + 2)):
        for j in range(max(0, pos[1] - 1), min(cols, pos[1] + 2)):
            if (i, j) != pos:
                neighbours.append((i, j))
    return neighbours

@numba.njit(cache=True)
def inflate_array(nsize: int, partial:NDArray[np.float32], id_living:NDArray[np.intp]):
    """_summary_

    Args:
        nsize (int): _description_
        partial (NDArray[np.float32]): _description_
        id_living (NDArray[np.intp]): _description_

    Returns:
        _type_: _description_
    """
    c = 0
    complete = np.zeros(nsize, dtype=np.float32)
    for n in id_living:
        complete[n] = partial[c]
        c += 1
    return complete

@numba.jit(nopython=True, cache=True)
def linear_func(temp: float,
                vpd: float,
                T_max: float = 45.0,
                VPD_max : float = 3.8) -> float:
    """Linear function to calculate the coupling between the atmosphere and the canopy"""
    linear_func = (temp / T_max + vpd / VPD_max) / 2.0

    # Ensure the output is between 0 and 1
    if linear_func > 1.0:
        linear_func = 1.0
    elif linear_func < 0.0:
        linear_func = 0.0

    linear_func = 0.0 if linear_func < 0.0 else linear_func
    linear_func = 1.0 if linear_func > 1.0 else linear_func

    return linear_func

@numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32), nopython=True, cache=True)
def atm_canopy_coupling(emaxm: float, evapm: float, air_temp: float, vpd: float) -> float:
    """Calculate the coupling between the atmosphere and the canopy based on a simple linear function
    of the air temperature and the vapor pressure deficit.
    Args:
        emaxm: float -> maximum evaporation rate mm/day
        evapm: float -> evaporation rate mm/day
        air_temp: float -> air temperature in Celsius
        vpd: float -> vapor pressure deficit in kPa
    Returns:
        float: Evapotranspiration rate mm/day
        """

    omega = linear_func(air_temp, vpd)
    return emaxm * omega + evapm * (1 - omega)

@numba.jit(numba.float32(numba.int8[:], numba.float32[:]), nopython=True, cache=True)
def masked_mean(mask: NDArray[np.int8], values: NDArray[np.float32]) -> float:
    """Calculate the mean of the values array ignoring the masked values"""
    mean = 0.0
    count = np.logical_not(mask).sum()
    if count == 0:
        return np.nan

    for i in range(mask.size):
        if mask[i] == 0:
            mean += values[i] / count
    return mean

@numba.jit(numba.float32[:](numba.int8[:], numba.float32[:,:]), nopython=True, cache=True)
def masked_mean_2D(mask: NDArray[np.int8], values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate the mean of the values array ignoring the masked values"""
    integrate_dim = values.shape[0]
    dim_sum = np.zeros(integrate_dim, dtype=np.float32)
    count = np.zeros(integrate_dim, dtype=np.int32)
    for i in range(mask.size):
        if mask[i] == 0:
            for j in range(integrate_dim):
                dim_sum[j] += values[j, i]
                count[j] += 1
    return dim_sum / count

@numba.jit(numba.float32(numba.float64[:], numba.float32[:]), nopython=True, cache=True)
def cw_mean(ocp: NDArray[np.float64], values: NDArray[np.float32]) -> np.float32:
    """
    Calculate the Community weighted mean for values using an
    array of area occupation (0 (empty) -1 (Total dominance))"""

    return np.sum(ocp * values, dtype = np.float32)

@numba.jit(numba.float32(numba.float64[:], numba.float32[:], numba.float32), nopython=True, cache=True)
def cw_variance(ocp: NDArray[np.float64], values: NDArray[np.float32], mean: float) -> float:
    """Calculate the Community weighted variance for values using an
    array of area occupation (0 (empty) -1 (Total dominance))"""

    variance = 0.0
    for i in range(ocp.size):
        variance += ocp[i] * ((values[i] - mean) ** 2)
    return variance

# Some functions to calculate diversity and evenness indices coded by copilot
# TODO: Check the implementation of these functions
# These functions are not used yet in the code
@numba.jit(nopython=True, cache=True)
def shannon_entropy(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon entropy for a community"""
    entropy = 0.0
    for i in range(ocp.size):
        if ocp[i] > 0:
            entropy -= ocp[i] * np.log(ocp[i])
    return entropy

@numba.jit(nopython=True, cache=True)
def shannon_evenness(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon evenness for a community"""
    max_entropy = np.log(ocp.size)
    return shannon_entropy(ocp) / max_entropy

@numba.jit(nopython=True, cache=True)
def shannon_diversity(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon diversity for a community"""
    return np.exp(shannon_entropy(ocp))

@numba.jit(nopython=True, cache=True)
def simpson_diversity(ocp: NDArray[np.float64]) -> float:
    """Calculate the Simpson diversity for a community"""
    simpson = 0.0
    for i in range(ocp.size):
        simpson += ocp[i] ** 2
    return simpson

@numba.jit(nopython=True, cache=True)
def simpson_evenness(ocp: NDArray[np.float64]) -> float:
    """Calculate the Simpson evenness for a community"""
    return simpson_diversity(ocp) / ocp.size


class state_zero:
    """base class with input/output related data (paths, filenames, etc)
    """

    def __init__(self, y:Union[int, float], x:Union[int, float],
                output_dump_folder:str | Path,
                get_main_table:Callable)->None:
        """Construct the basic gridcell object. Fetch the configuration data and set the CRS data.

        if you give a pair of integers, the gridcell will understand that you are giving the indices of the
        gridcell in a 2D numpy array of gridcells representing the area of simulation
        if you give a pair of floats, the gridcell will understand that you are giving the coordinates
        of the gridcell in the real world (latitude and longitude). The indices are used to locate
        the input file that contains the climatic and soil data. The files must be named as grd_x-y.pbz2 where x and y are the indices of the gridcell

        Args:

        y: int | float -> index in the 0 axis [zero-indexed] or geographic latitude coordinate [degrees North]
        x: int | float -> index in the 1 axis [zero-indexed] or geographic longitude coordinate [degrees East]
        output_dump_folder: str -> a string with a valid name to an output folder. This will be used to create a
        child directory in the output location for the region that contains this gridcell.

        """

        assert type(y) == type(x), "x and y must be of the same type"


        # Configuration data
        self.config:Config = fetch_config("caete.toml")
        self.afex_config: config_type = self.config.fertilization # type: ignore
        self.co2_data: Optional[Dict[int, float]] = None

        # CRS
        self.yres = self.config.crs.yres # type: ignore
        self.xres = self.config.crs.xres # type: ignore

        self.y, self.x = find_indices_xy(N = y, W = x, res_y=self.yres,
                                         res_x=self.xres,rounding=2) if isinstance(x, float) else (y, x)

        self.lat, self.lon = find_coordinates_xy(self.y, self.x, res_y=self.yres,                   # type: ignore
                                                 res_x=self.xres) if isinstance(x, int) else (y, x)

        self.cell_area = calculate_area(self.lat, self.lon,
                                        dx=self.xres, dy=self.yres)

        # Files & IO
        self.xyname = f"{self.y}-{self.x}"
        self.grid_filename = f"gridcell{self.xyname}"
        self.input_fname = f"input_data_{self.xyname}.pbz2"
        self.input_fpath = None
        self.data = None
        self.doy_months = set(self.config.doy_months) # type: ignore

        # Name of the dump folder where this gridcell will dump model outputs.
        # It is a child from ../outputs - defined in caete.toml
        self.plot_name = output_dump_folder

        # Plant life strategies table
        self.get_from_main_array = get_main_table
        self.ncomms = None
        self.metacomm = None

        # Store start and end date for each "spin"
        self.executed_iterations: List[Tuple[str,str]] = []

        # OUTPUT FOLDER STRUCTURE
        self.outputs = {}       # dict, store filepaths of output data generated by this
        # Root dir for the region outputs
        self.out_dir = output_path/Path(output_dump_folder)
        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = None

        # counts the execution of a time slice (a call of self.run_spinup)
        self.run_counter = 0


class climate:
    """class with climate data"""

    def __init__(self):
        """_summary_
        """
        self.pr: NDArray[np.float64]
        self.ps: NDArray[np.float64]
        self.rsds: NDArray[np.float64]
        self.tas: NDArray[np.float64]
        self.rhs: NDArray[np.float64]


    def _set_clim(self, data:Dict):
        """_summary_

        Args:
            data (Dict): _description_
        """
        self.pr = data['pr']
        self.ps = data['ps']
        self.rsds = data['rsds']
        self.tas = data['tas']
        self.rhs = data['hurs']


    def _set_tas(self, data:Dict):
        """_summary_

        Args:
            data (Dict): _description_
        """
        self.tas = data['tas']


    def _set_pr(self, data:Dict):
        """_summary_"""
        self.pr = data['pr']


    def _set_ps(self, data:Dict):
        """_summary_"""
        self.ps = data['ps']


    def _set_rsds(self, data:Dict):
        """_summary_"""
        self.rsds = data['rsds']


    def _set_rhs(self, data:Dict):
        """_summary_"""
        self.rhs = data['hurs']


    def _set_co2(self, fpath:Union[Path, str]):
        """_summary_"""
        self.co2_path = str_or_path(fpath, check_is_file=True)
        self.co2_data = get_co2_concentration(self.co2_path)


class time:
    """_summary_
    """
    def __init__(self):
        """Time attributes"""
        self.time_index:np.ndarray
        self.calendar:str
        self.time_unit:str
        self.start_date:str
        self.end_date:str
        self.sind: int
        self.eind: int


    def _set_time(self, stime_i:Dict):
        """_summary_

        Args:
            stime_i (Dict): _description_
        """
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)


class soil:
    """_summary_
    """

    def __init__(self):
        """_summary_
        """
        self.sp_csoil = None
        self.sp_snc = None
        self.input_nut = None
        self.sp_available_p = None
        self.sp_available_n = None
        self.sp_so_n = None
        self.sp_in_n = None
        self.sp_so_p = None
        self.sp_in_p = None
        self.sp_csoil = None
        self.sp_snr = None
        self.sp_uptk_costs = None
        self.sp_organic_n = None
        self.sp_sorganic_n = None
        self.sp_organic_p = None
        self.sp_sorganic_p = None

        # Water
        # Water content for each soil layer
        self.wp_water_upper_mm = None  # mm
        self.wp_water_lower_mm = None  # mm
        self.wmax_mm = None  # mm
        self.theta_sat = None
        self.psi_sat = None
        self.soil_texture = None


    def _init_soil_cnp(self, data:Dict):
        """_summary_

        Args:
            data (Dict): _description_
        """
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 0.001
        self.sp_snc = np.zeros(shape=(8,), order='F') + 0.0001
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
        self.sp_organic_n = 0.1 * self.soil_dict['tn']
        self.sp_sorganic_n = 0.1 * self.soil_dict['tn']
        self.sp_organic_p = 0.5 * self.soil_dict['op']
        self.sp_sorganic_p = self.soil_dict['op'] - self.sp_organic_p


    def _init_soil_water(self, tsoil:Tuple, ssoil:Tuple, hsoil:Tuple):
        """Initializes the soil pools

        Args:
            tsoil (Tuple): tuple with the soil water content for the upper layer
            ssoil (Tuple): tuple with the soil water content for the lower layer
            hsoil (Tuple): tuple with the soil texture, saturation point and water potential at saturation
        """
        assert self.tas is not None, "Climate data not loaded" # type: ignore
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)  # type: ignore

        self.tsoil = []
        self.emaxm = []

        # GRIDCELL STATE
        # Water
        self.ws1 = tsoil[0][self.y, self.x].copy() # type: ignore
        self.fc1 = tsoil[1][self.y, self.x].copy() # type: ignore
        self.wp1 = tsoil[2][self.y, self.x].copy() # type: ignore

        self.ws2 = ssoil[0][self.y, self.x].copy() # type: ignore
        self.fc2 = ssoil[1][self.y, self.x].copy() # type: ignore
        self.wp2 = ssoil[2][self.y, self.x].copy() # type: ignore

        self.swp = soil_water(self.ws1, self.ws2, self.fc1, self.fc2, self.wp1, self.wp2)
        self.wp_water_upper_mm = self.swp.w1
        self.wp_water_lower_mm = self.swp.w2
        self.wmax_mm = np.float64(self.swp.w1_max + self.swp.w2_max)

        self.theta_sat = hsoil[0][self.y, self.x].copy() # type: ignore
        self.psi_sat = hsoil[1][self.y, self.x].copy() # type: ignore
        self.soil_texture = hsoil[2][self.y, self.x].copy() # type: ignore


    def add_soil_nutrients(self, afex_mode:str):
        """_summary_

        Args:
            afex_mode (str): _description_
        """
        if afex_mode == 'N':
            self.sp_available_n += self.afex_config.n # type: ignore
        elif afex_mode == 'P':
            self.sp_available_p += self.afex_config.p # type: ignore
        elif afex_mode == 'NP':
            self.sp_available_n += self.afex_config.n # type: ignore
            self.sp_available_p += self.afex_config.p # type: ignore


    def add_soil_water(self, data:Dict):
        """_summary_

        Args:
            data (Dict): _description_
        """

        # will deal with irrigation experiments and possibly water table depth
        pass


class gridcell_output:
    """Class to manage gridcell outputs
    """
    def __init__(self):
        """_summary_
        """
        self.run_counter: int = 0
        self.flush_data:Optional[Dict]
        self.emaxm: List = []
        self.tsoil: List = []
        self.soil_temp:NDArray
        self.photo:NDArray
        self.ls :NDArray
        self.aresp:NDArray
        self.npp:NDArray
        self.rnpp: NDArray
        self.lai:NDArray
        self.csoil:NDArray
        self.inorg_n:NDArray
        self.inorg_p:NDArray
        self.sorbed_n:NDArray
        self.sorbed_p:NDArray
        self.snc:NDArray
        self.hresp:NDArray
        self.rcm:NDArray
        self.f5:NDArray
        self.runom:NDArray
        self.evapm:NDArray
        self.wsoil:NDArray
        self.swsoil:NDArray
        self.rm:NDArray
        self.rg:NDArray
        self.cleaf:NDArray
        self.cawood:NDArray
        self.cfroot:NDArray
        self.ocp_area:NDArray
        self.wue:NDArray
        self.cue:NDArray
        self.cdef:NDArray
        self.nmin:NDArray
        self.pmin:NDArray
        self.vcmax:NDArray
        self.specific_la:NDArray
        self.nupt:NDArray
        self.pupt:NDArray
        self.litter_l:NDArray
        self.cwd:NDArray
        self.litter_fr:NDArray
        self.lnc:NDArray
        self.storage_pool:NDArray
        self.lim_status:NDArray

        self.uptake_strategy:NDArray
        self.carbon_costs:NDArray


    def _allocate_output(self, n, npls, ncomms, save=True):
        """allocate space for the outputs
        n: int NUmber of days being simulated"""

        self.evapm = np.zeros(shape=(n,), order='F')
        self.runom = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(2, n), order='F')
        self.pupt = np.zeros(shape=(3, n), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')
        self.rnpp = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))

        if not save:
            return None

        # Daily outputs
        self.emaxm = []
        self.tsoil = []
        self.photo = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.aresp = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.npp = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))

        self.inorg_n = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.inorg_p = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.sorbed_n = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.sorbed_p = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.snc = np.zeros(shape=(8, n), order='F', dtype=np.dtype("float32"))
        self.hresp = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.rcm = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.f5 = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.rm = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.rg = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.wue = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.cue = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.cdef = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.nmin = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.pmin = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.vcmax = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))

        self.carbon_costs = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.ocp_area = np.zeros(shape=(npls, ncomms, n), dtype=('int32'), order='F')
        self.lim_status = np.zeros(
            shape=(3, npls, ncomms, n), dtype=np.dtype('int8'), order='F')
        self.uptake_strategy = np.zeros(
            shape=(2, npls, ncomms, n), dtype=np.dtype('int8'), order='F')

        self.lai = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.csoil = np.zeros(shape=(4, n), order='F', dtype=np.dtype("float32"))
        self.wsoil = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.swsoil = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.cleaf = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.cawood = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.cfroot = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.specific_la = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))


    def _flush_output(self, run_descr, index):
        """1 - Clean variables that receive outputs from the fortran subroutines
           2 - Fill self.outputs dict with filepats of output data
           3 - Returns the output data to be writen

           runs_descr: str a name for the files
           index = tuple or list with the first and last values of the index time variable"""
        # This code uses some attribute that is not defined in the class

        to_pickle = {}
        self.run_counter += 1
        if self.run_counter < 10:
            spiname = run_descr + "000" + str(self.run_counter) + out_ext
        elif self.run_counter < 100:
            spiname = run_descr +  "00" + str(self.run_counter) + out_ext
        elif self.run_counter < 1000:
            spiname = run_descr +  "0" + str(self.run_counter) + out_ext
        else:
            spiname = run_descr + str(self.run_counter) + out_ext

        self.outputs[spiname] = os.path.join(self.out_dir, spiname) # type: ignore
        to_pickle = {'emaxm': np.array(self.emaxm),
                     "tsoil": np.array(self.tsoil),
                     "photo": self.photo,
                     "aresp": self.aresp,
                     'npp': self.npp,
                     'rnpp': self.rnpp,
                     'lai': self.lai,
                     'csoil': self.csoil,
                     'inorg_n': self.inorg_n,
                     'inorg_p': self.inorg_p,
                     'sorbed_n': self.sorbed_n,
                     'sorbed_p': self.sorbed_p,
                     'snc': self.snc,
                     'hresp': self.hresp,
                     'rcm': self.rcm,
                     'f5': self.f5,
                     'runom': self.runom,
                     'evapm': self.evapm,
                     'wsoil': self.wsoil,
                     'swsoil': self.swsoil,
                     'rm': self.rm,
                     'rg': self.rg,
                     'cleaf': self.cleaf,
                     'cawood': self.cawood,
                     'cfroot': self.cfroot,
                     'area': self.ocp_area,
                     'wue': self.wue,
                     'cue': self.cue,
                     'cdef': self.cdef,
                     'nmin': self.nmin,
                     'pmin': self.pmin,
                     'vcmax': self.vcmax,
                     'specific_la': self.specific_la,
                     'nupt': self.nupt,
                     'pupt': self.pupt,
                     'litter_l': self.litter_l,
                     'cwd': self.cwd,
                     'litter_fr': self.litter_fr,
                     'lnc': self.lnc,
                     'ls': self.ls,
                     'lim_status': self.lim_status,
                     'c_cost': self.carbon_costs,
                     'u_strat': self.uptake_strategy,
                     'storage_pool': self.storage_pool,
                     'calendar': self.calendar,    # Calendar name # type: ignore
                     'time_unit': self.time_unit,  # Time unit # type: ignore
                     'sind': index[0],
                     'eind': index[1]}
        # Flush attrs
        dummy_array = np.empty(0, dtype=np.float32)
        self.emaxm: List = []
        self.tsoil: List = []
        self.photo: NDArray = dummy_array
        self.aresp: NDArray = dummy_array
        self.npp: NDArray = dummy_array
        self.rnpp: NDArray = dummy_array
        self.lai: NDArray = dummy_array
        self.csoil: NDArray = dummy_array
        self.inorg_n: NDArray = dummy_array
        self.inorg_p: NDArray = dummy_array
        self.sorbed_n: NDArray = dummy_array
        self.sorbed_p: NDArray = dummy_array
        self.snc: NDArray = dummy_array
        self.hresp: NDArray = dummy_array
        self.rcm: NDArray = dummy_array
        self.f5: NDArray = dummy_array
        self.runom: NDArray = dummy_array
        self.evapm: NDArray = dummy_array
        self.wsoil: NDArray = dummy_array
        self.swsoil: NDArray = dummy_array
        self.rm: NDArray = dummy_array
        self.rg: NDArray = dummy_array
        self.cleaf: NDArray = dummy_array
        self.cawood: NDArray = dummy_array
        self.cfroot: NDArray = dummy_array
        self.area: NDArray = dummy_array
        self.wue: NDArray = dummy_array
        self.cue: NDArray = dummy_array
        self.cdef: NDArray = dummy_array
        self.nmin: NDArray = dummy_array
        self.pmin: NDArray = dummy_array
        self.vcmax: NDArray = dummy_array
        self.specific_la: NDArray = dummy_array
        self.nupt: NDArray = dummy_array
        self.pupt: NDArray = dummy_array
        self.litter_l: NDArray = dummy_array
        self.cwd: NDArray = dummy_array
        self.litter_fr: NDArray = dummy_array
        self.lnc: NDArray = dummy_array
        self.storage_pool: NDArray = dummy_array
        self.ls: NDArray = dummy_array
        self.lim_status: NDArray = dummy_array
        self.carbon_costs: NDArray = dummy_array
        self.uptake_strategy: NDArray = dummy_array
        return to_pickle


    def _save_output(self, data_obj: Dict[str, Union[NDArray, str, int]]):
        """Compress and save output data
        data_object: dict; the dict returned from _flush_output"""
        if self.run_counter < 10: # type: ignore
            fpath = "spin{}{}{}{}{}".format(0, 0, 0, self.run_counter, out_ext) # type: ignore
        elif self.run_counter < 100: # type: ignore
            fpath = "spin{}{}{}{}".format(0, 0, self.run_counter, out_ext) # type: ignore
        elif self.run_counter < 1000: # type: ignore
            fpath = "spin{}{}{}".format(0, self.run_counter, out_ext)
        else:
            fpath = "spin{}{}".format(self.run_counter, out_ext) # type: ignore
        with open(self.outputs[fpath], 'wb') as fh: # type: ignore
            dump(data_obj, fh, compress=('lz4', 6), protocol=4) # type: ignore
        self.flush_data = None


class grd_mt(state_zero, climate, time, soil, gridcell_output):

    """A gridcell object to run the model in the meta-community mode

    Args:
        base classes with climatic, soil data, and some common methods to manage gridcells
    """


    def __init__(self, y: int | float,
                x: int | float,
                data_dump_directory: str | Path,
                get_main_table:Callable
                )->None:

        """Construct the gridcell object

        Args:
            y (int | float): latitude(float) or index(int) in the y dimension
            x (int | float): longitude(float) or index(int) in the x dimension
            data_dump_directory (str): Where this gridcell will dump model outputs
            get_main_table (callable): a region method used to get PLS from the main table
            to create the metacommunity.
        """

        super().__init__(y, x, data_dump_directory, get_main_table)
        # self.spin_data: Optional[Dict] = None #


    def find_co2(self, year:int)->float:
        """Reads the CO₂ data for a given year. The units are expected to be in ppm

        Args:
            year (int): Year to read

        Raises:
            ValueError: If the year is not in the CO₂ data

        Returns:
            float: CO₂ concentration for the given year in ppm (parts per million) eq. to µmol mol⁻¹
        """

        assert isinstance(year, int), "year must be an integer"
        assert self.co2_data, "CO2 data not loaded"
        _val_ = self.co2_data.get(year)
        if _val_ is None:
            raise ValueError(f"Year {year} not in ATM[CO₂] data")
        return _val_


    def find_index(self, start:int, end:int)->list:
        """
        Used to find the indices of the time array that correspond to the start and end dates
        It finds the lower and upper indices (zero-based) of the time array that correspond to
        the start and end dates

        Args:
            start (int): Time index of the start date (based on calendar and time unit)
            end (int): Time index of the end date (based on calendar and time unit)

        Raises:
            ValueError: If the start or end date is out of bounds

        Returns:
            List(int,int): lower bound, upper bound

        """

        # Ensure start and end are within the bounds
        if start < self.sind or end > self.eind:
            raise ValueError("start or end out of bounds")

        # Find the indices
        start_index = np.where(np.arange(self.sind, self.eind + 1) == start)[0]
        end_index = np.where(np.arange(self.sind, self.eind + 1) == end)[0]

        # Combine and return the results
        return np.concatenate((start_index, end_index)).tolist()


    def change_input(self,
                    input_fpath:Union[Path, str, None]=None,
                    stime_i:Union[Dict, None]=None,
                    co2:Union[Dict, str, Path, None]=None)->None:
        """modify the input data for the gridcell

        Args:
            input_fpath (Union[Path, str], optional): _description_. Defaults to None.
            stime_i (Union[Dict, None], optional): _description_. Defaults to None.
            co2 (Union[Dict, str, Path, None], optional): _description_. Defaults to None.

        Returns:
            None: Changes the input data for the gridcell
        """
        if input_fpath is not None:
            #TODO prevent errors here
            self.input_fpath = Path(os.path.join(input_fpath, self.input_fname))
            assert self.input_fpath.exists()

            with bz2.BZ2File(self.input_fpath, mode='r') as fh:
                self.data = pkl.load(fh)

            self._set_clim(self.data)

        if stime_i is not None:
            self._set_time(stime_i)

        if co2 is not None:
            if isinstance(co2, (str, Path)):
                self._set_co2(co2)
            elif isinstance(co2, dict):
                self.co2_data = copy.deepcopy(co2)

        return None


    def set_gridcell(self,
                      input_fpath:Union[Path, str],
                      stime_i: Dict,
                      co2: Dict,
                      tsoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                      ssoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                      hsoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
                      )->None:
        """ PREPARE A GRIDCELL TO RUN in the meta-community mode

        Args:
            input_fpath (Union[Path, str]): path to the input file with climatic and soil data
            stime_i (Dict): dictionary with the time index and units
            co2 (Dict): dictionary with the CO2 data
            pls_table (np.ndarray): np.array with the functional traits data
            tsoil (Tuple[np.ndarray]):
            ssoil (Tuple[np.ndarray]):
            hsoil (Tuple[np.ndarray]):
        """
        # Input data
        self.input_fpath = str_or_path(input_fpath)

        # # Meta-community
        # We want to run queues of gridcells in parallel. So each gridcell receives a copy of the PLS table object

        # Number of communities in the metacommunity. Defined in the config file {caete.toml}
        # Each gridcell has one metacommunity wuth ncomms communities
        self.ncomms:int = self.config.metacomm.n  #type: ignore # Number of communities

        # Metacommunity object
        self.metacomm:mc.metacommunity = mc.metacommunity(self.ncomms, self.get_from_main_array)


        # Read climate drivers and soil characteristics, incl. nutrients, for this gridcell
        # Having all data to one gridcell in a file enables to create/start the gricells in parallel (threading)
        # TODO: implement this multithreading in the region class to start all gridcells in parallel
        self.data = read_bz2_file(self.input_fpath)

        # Read climate data
        self._set_clim(self.data)

        # get CO2 data
        self.co2_data = copy.deepcopy(co2)

        # SOIL: NUTRIENTS and WATER
        self._init_soil_cnp(self.data)
        self._init_soil_water(tsoil, ssoil, hsoil)

        # TIME
        self._set_time(stime_i)

        return None

    # @profile
    def run_gridcell(self,
                  start_date: str,
                  end_date: str,
                  spinup: int = 0,
                  fixed_co2_atm_conc: Optional[str] = None,
                  save: bool = True,
                  nutri_cycle: bool = True,
                  afex: bool = False,
                  reset_community: bool = False,
                  kill_and_reset: bool = False,
                  env_filter: bool = False,
                  verbose: bool = True):
        """
        Run the model for a specific grid cell.

        CAETÊ-DVM execution in the start_date - end_date period, can be used for spinup or transient runs.

        Args:
            start_date (str): Start date for model execution in "yyyymmdd" format.
            end_date (str): End date for model execution in "yyyymmdd" format.
            spinup (int, optional): Number of repetitions in spinup. Set to 0 for a transient run between start_date and end_date. Default is 0.
            fixed_co2_atm_conc (Optional[float], optional): Fixed atmospheric CO2 concentration. If None, use dynamic CO2 levels. Default is None.
            save (bool, optional): Whether to save the results. Default is True.
            nutri_cycle (bool, optional): Whether to include nutrient cycling in the model. Default is True.
            afex (bool, optional): Whether to apply additional effects (AFEX) in the model. Default is False.
            reset_community (bool, optional): Whether to reset the community structure at the start. Default is False.
            kill_and_reset (bool, optional): Whether to kill and reset the community structure during the run. Default is False.
            env_filter (bool, optional): Whether to apply environmental filtering. Default is False.
            verbose (bool, optional): Whether to print detailed logs during execution. Default is True.

        Returns:
            None

        Notes:
            - If reset_community is true a new community will be set (reset) when there is no PLSs remaining.
            - If the kill_and_reset is true, after n spins (integer given by spinup parameter - i.e. in the end
              of function execution) all the communities in a gridcell are reset. The reset_community and
              kill_and_reset  arguments are not mutually exclusive. You can use both as true at the same time.
            - The env_filter argument is used to define if new unique PLSs from the main table will be
              seed in the communities that have free slots (PLSs that are not producing). At the moment, the
              interval for the env_filter to add a new PLS to the community is set to  ~30 days.
              If env filter argument is true, then the reset_community argument will have a very low
              probability to trigger a reset because the communities will be constantly filled with new PLS.
              Nonetheless, the reset_community argument will still be able to trigger a reset if the community loses all PLSs.
              With the probability of a reset_community increasing as the interval between new seeds increases.

              TODO: Implement a more flexible way to define the interval for
                    the env_filter to add a new PLS to the community.
        """

        assert not fixed_co2_atm_conc or\
            isinstance(fixed_co2_atm_conc, str) or\
            fixed_co2_atm_conc > 0,\
                "A fixed value for ATM[CO2] must be a positive number greater than zero or a proper string with the year - e.g., 'yyyy'"

        # Define start and end dates (read parameters)
        start = parse_date(start_date)
        end = parse_date(end_date)

        # Check dates sanity
        assert start < end, "Start date must be before end date"
        assert start >= self.start_date, "initial date out of bounds for the time array"
        assert end <= self.end_date, f"Final date out of bounds for the time array"


        # Define time index bounds for this run
        # During a run we are in general using a slice ov the available time span
        # to run the model. For example, we can run the model for a year or a decade
        # at the begining of the input data time series to spin up. This slice is defined
        # by the start and end dates provided in the arguments. HEre we get the indices.
        start_index = int(cftime.date2num(start, self.time_unit, self.calendar))
        end_index =   int(cftime.date2num(end, self.time_unit, self.calendar))

        # Find the indices in the time array [used to slice the timeseries with driver data  - tas, pr, etc.]
        lower_bound, upper_bound = self.find_index(start_index, end_index)

        # Define the time steps range
        # From zero to the last day of simulation
        steps = np.arange(lower_bound, upper_bound + 1, dtype=np.int64)

        # Define the number of repetitions for the spinup
        spin = 1 if spinup == 0 else spinup

        # Define the AFEX mode
        afex_mode = self.afex_config.afex_mode # type: ignore

        # Slice&Catch climatic input and make conversions
        cv = self.config.conversion_factors_isimip # type: ignore

        temp: NDArray[np.float64] = self.tas[lower_bound: upper_bound + 1] - cv.tas   # Air temp: model uses °C
        prec: NDArray[np.float64] = self.pr[lower_bound: upper_bound + 1] * cv.pr     # Precipitation: model uses  mm/day
        p_atm: NDArray[np.float64] = self.ps[lower_bound: upper_bound + 1] * cv.ps    # Atmospheric pressure: model uses hPa
        ipar: NDArray[np.float64] = self.rsds[lower_bound: upper_bound + 1] * cv.rsds # PAR: model uses  mol(photons) m-2 s-1
        ru: NDArray[np.float64] = self.rhs[lower_bound: upper_bound + 1] *  cv.rhs    # Relative humidity: model uses 0-1

        # Define the daily values for co2 concentrations
        co2_daily_values = np.zeros(steps.size, dtype=np.float32)

        if fixed_co2_atm_conc is None:
            # In this case, the co2 concentration will be updated daily.
            # We interpolate linearly between the yearly values of the atm co2 data
            co2 = self.find_co2(start.year)
            today = datetime(start.year, start.month, start.day, start.hour, start.minute, start.second)
            time_step = timedelta(days=1) # Define the time step
            today -= time_step # The first thing we do next is to add a day to the date. So we go back one day
            # Loop over the days and calculate the co2 concentration for each day
            for step in range(steps.size):
                today += time_step
                remaining = (datetime(today.year, 12, 31) - today).days + 1
                daily_fraction = (self.find_co2(today.year + 1) - co2) / (remaining + 1)
                co2 += daily_fraction
                co2_daily_values[step] = co2
        elif isinstance(fixed_co2_atm_conc, int) or isinstance(fixed_co2_atm_conc, float):
            # In this case, the co2 concentration will be fixed according to the numeric value provided in the argument
            co2 = fixed_co2_atm_conc
            co2_daily_values += co2
        elif isinstance(fixed_co2_atm_conc, str):
            # In this case, the co2 concentration will be fixed
            # According to the year provided in the argument
            # as a string. Format "yyyy".
            try:
                co2_year = int(fixed_co2_atm_conc)
            except ValueError:
                raise ValueError(
                    "The string(\"yyyy\") must be a number in the {self.start_date.year} - {self.end_date.year} interval")
            co2 = self.find_co2(co2_year)
            co2_daily_values += co2
        else:
            raise ValueError("Invalid value for fixed_co2_atm_conc")

        # Start loops
        # THis outer loop is used to run the model for a number
        # of times defined by the spinup argument. THe model is
        # executed repeatedly between the start and end dates
        # provided in the arguments
        first_day_of_simulation = datetime(start.year, start.month, start.day, start.hour, start.minute, start.second)
        # Define the time step
        time_step = timedelta(days=1)
        for s in range(spin):

            self._allocate_output(steps.size, self.metacomm.comm_npls, len(self.metacomm), save)

            # Loop over the days
            # Create a datetime object to track the dates
            today = first_day_of_simulation

            # Go back one day
            today -= time_step

            # Arrays to store values for each community in a simulated day
            sto =        np.zeros(shape=(3, self.metacomm.comm_npls), order='F')
            cleaf_in =   np.zeros(self.metacomm.comm_npls, order='F')
            cwood_in =   np.zeros(self.metacomm.comm_npls, order='F')
            croot_in =   np.zeros(self.metacomm.comm_npls, order='F')
            uptk_costs = np.zeros(self.metacomm.comm_npls, order='F')
            rnpp_in =    np.zeros(self.metacomm.comm_npls, order='F')

            # Arrays to store values for each community in a simulated day
            # There are two modes of operation: save and not save.
            # In the save mode, the arrays are used to store the values that are
            # needed for model iteration, i.e., the values that are used in the next
            # time step. In the save mode, an extra number arrays are created to be used
            # to store the outputs.
            xsize: int = len(self.metacomm) # Number of communities
            evavg: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)
            epavg: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)
            rnpp_mt: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)

            # We keep track of these to input in SOM dynamics later. They are used for output also
            leaf_litter: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)
            cwd: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)
            root_litter: NDArray[np.float32] = np.zeros(xsize, dtype=np.float32)
            lnc: NDArray[np.float32] = np.zeros(shape=(6, xsize), dtype=np.float32)
            # THis is added to leaf litter pool (that is basicaly a fast SOM pool)
            c_to_nfixers: NDArray[np.float32]= np.zeros(xsize, dtype=np.float32)

            if save:
                nupt = np.zeros(shape=(2, xsize), dtype=np.float32)
                pupt = np.zeros(shape=(3, xsize), dtype=np.float32)
                cc = np.zeros(xsize, dtype=np.float32)
                photo = np.zeros(xsize, dtype=np.float32)
                aresp = np.zeros(xsize, dtype=np.float32)
                npp = np.zeros(xsize, dtype=np.float32)
                lai = np.zeros(xsize, dtype=np.float32)
                rcm = np.zeros(xsize, dtype=np.float32)
                f5 = np.zeros(xsize, dtype=np.float32)
                rm = np.zeros(xsize, dtype=np.float32)
                rg = np.zeros(xsize, dtype=np.float32)
                cleaf = np.zeros(xsize, dtype=np.float32)
                cawood = np.zeros(xsize, dtype=np.float32)
                cfroot = np.zeros(xsize, dtype=np.float32)
                wue = np.zeros(xsize, dtype=np.float32)
                cue = np.zeros(xsize, dtype=np.float32)
                cdef = np.zeros(xsize, dtype=np.float32)
                vcmax = np.zeros(xsize, dtype=np.float32)
                specific_la = np.zeros(xsize, dtype=np.float32)
                storage_pool = np.zeros(shape=(3, xsize))
                ocp_area = np.ma.masked_all(shape=(self.metacomm.comm_npls, xsize), dtype='int32')
                lim_status = np.ma.masked_all(shape=(3, self.metacomm.comm_npls, xsize), dtype=np.dtype('int8'))
                uptake_strategy = np.ma.masked_all(shape=(2, self.metacomm.comm_npls, xsize), dtype=np.dtype('int8'))

            # <- Daily loop
            for step in range(steps.size):
                today += time_step # Now it is today
                julian_day = today.timetuple().tm_yday

                # Get the co2 concentration for the day
                co2 = co2_daily_values[step]
                # Update soil temperature
                self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

                # AFEX
                if afex and julian_day == 364:
                    self.add_soil_nutrients(afex_mode)

                # Loop over communities
                # Create these arrays outside and just reuse it
                living_pls = 0 # Sum of living PLS in the communities
                for i, community in enumerate(self.metacomm):
                    if community.masked:
                        # skip this one
                        continue
                    sto[0, :] = inflate_array(community.npls, community.vp_sto[0, :], community.vp_lsid)
                    sto[1, :] = inflate_array(community.npls, community.vp_sto[1, :], community.vp_lsid)
                    sto[2, :] = inflate_array(community.npls, community.vp_sto[2, :], community.vp_lsid)

                    cleaf_in[:] = inflate_array(community.npls, community.vp_cleaf, community.vp_lsid)
                    cwood_in[:] = inflate_array(community.npls, community.vp_cwood, community.vp_lsid)
                    croot_in[:] = inflate_array(community.npls, community.vp_croot, community.vp_lsid)
                    uptk_costs[:] = inflate_array(community.npls, community.sp_uptk_costs, community.vp_lsid)
                    rnpp_in[:] = inflate_array(community.npls, community.construction_npp, community.vp_lsid)

                    ton = self.sp_organic_n #+ self.sp_sorganic_n
                    top = self.sp_organic_p #+ self.sp_sorganic_p

                    # Community daily budget calculation
                    out = model.daily_budget(community.pls_array, self.wp_water_upper_mm,
                                            self.wp_water_lower_mm, self.soil_temp, temp[step],
                                            p_atm[step], ipar[step], ru[step], self.sp_available_n,
                                            self.sp_available_p, ton, top, self.sp_organic_p,
                                            co2, sto, cleaf_in, cwood_in, croot_in, uptk_costs,self.wmax_mm,
                                            rnpp_in)

                    # get daily budget results
                    daily_output = budget_daily_result(out)

                    # Update the community status
                    community.update_lsid(daily_output.ocpavg)
                    community.vp_ocp = daily_output.ocpavg[community.vp_lsid]
                    community.ls = community.vp_lsid.size
                    community.vp_cleaf = daily_output.cleafavg_pft[community.vp_lsid]
                    community.vp_cwood = daily_output.cawoodavg_pft[community.vp_lsid]
                    community.vp_croot = daily_output.cfrootavg_pft[community.vp_lsid]
                    community.vp_sto = daily_output.stodbg[:, community.vp_lsid].astype('float32')
                    community.sp_uptk_costs = daily_output.npp2pay[community.vp_lsid]
                    community.construction_npp = daily_output.rnpp_out[community.vp_lsid]
                    living_pls += community.ls

                    # Restore or seed PLS
                    if env_filter and community.ls < self.metacomm.comm_npls:
                        if julian_day in self.doy_months:
                            if verbose:
                                print(f"PLS seed in Community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")
                            new_id, new_PLS = community.get_unique_pls(self.get_from_main_array)
                            community.seed_pls(new_id, new_PLS)
                    if community.vp_lsid.size < 1:
                        print(f"Empty community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")
                        if reset_community:
                            assert not save, "Cannot save data when resetting communities"
                            if verbose:
                                print(f"Reseting community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")
                            # Get the new life strategies. This is a method from the region class
                            # with lock:
                            new_life_strategies = self.get_from_main_array(community.npls)
                            community.restore_from_main_table(new_life_strategies)
                            continue
                        else:
                            # In the transiant run - i.e., when reset_community is false and
                            # kill_and_reset is false; we mask the community if there is no PLS
                            self.metacomm.mask[i] = np.int8(1)
                             # Set mask to true for this community, will not run in the next steps
                            community.masked = np.int8(1)
                            # if the reset_community is true
                            continue # cycle

                    # Store values for each community
                    rnpp_mt[i] = cw_mean(community.vp_ocp, community.construction_npp.astype(np.float32)) # Community Weighted rNPP
                    leaf_litter[i] = daily_output.litter_l
                    root_litter[i] = daily_output.litter_fr
                    cwd[i] = daily_output.cwd
                    lnc[:, i] = daily_output.lnc.astype(np.float32)
                    c_to_nfixers[i] = daily_output.cp[3]
                    evavg[i] = daily_output.evavg
                    epavg[i] = daily_output.epavg

                    if save:
                        nupt[:, i] = daily_output.nupt
                        pupt[:, i] = daily_output.pupt
                        cc[i] = daily_output.c_cost_cwm
                        npp[i] = daily_output.nppavg
                        photo[i] = daily_output.phavg
                        aresp[i] = daily_output.aravg
                        lai[i] = daily_output.laiavg
                        rcm[i] = daily_output.rcavg
                        f5[i] = daily_output.f5avg
                        rm[i] = daily_output.rmavg
                        rg[i] = daily_output.rgavg
                        cleaf[i] = daily_output.cp[0]
                        cawood[i] = daily_output.cp[1]
                        cfroot[i] = daily_output.cp[2]
                        wue[i] = daily_output.wueavg
                        cue[i] = daily_output.cueavg
                        cdef[i] = daily_output.c_defavg
                        vcmax[i] = daily_output.vcmax
                        specific_la[i] = daily_output.specific_la
                        ocp_area[:, i] = np.array(daily_output.ocpavg * 1e6, dtype='int32') # Quantize it
                        lim_status[:, :, i] = daily_output.limitation_status
                        uptake_strategy[:, :, i] = daily_output.uptk_strat

                        for j in range(daily_output.stodbg.shape[0]):
                            storage_pool[j, i] = cw_mean(community.vp_ocp, community.vp_sto[j, :])


                    # del daily_output
                #<- Out of the community loop
                vpd = m.vapor_p_deficit(temp[step], ru[step])
                et_pot = masked_mean(self.metacomm.mask, np.array(epavg).astype(np.float32)) #epavg.mean()
                et = masked_mean(self.metacomm.mask, epavg) #evavg.mean()

                # Update water pools
                # TODO add a type signature to jit these functions
                self.evapm[step] = atm_canopy_coupling(et_pot, et, temp[step], vpd)
                self.runom[step] = self.swp._update_pool(prec[step], self.evapm[step])
                self.swp.w1 = 0.0 if self.swp.w1 < 0.0 else self.swp.w1
                self.swp.w2 = 0.0 if self.swp.w2 < 0.0 else self.swp.w2
                self.wp_water_upper_mm = self.swp.w1
                self.wp_water_lower_mm = self.swp.w2
                wtot = self.wp_water_upper_mm + self.wp_water_lower_mm

                # Update cflux to the soil for output, mean values over the communities
                # Values are also used to update SOM dynamics
                self.litter_l[step] = masked_mean(self.metacomm.mask, leaf_litter) +\
                                      masked_mean(self.metacomm.mask, c_to_nfixers)
                self.cwd[step] = masked_mean(self.metacomm.mask, cwd)
                self.litter_fr[step] = masked_mean(self.metacomm.mask, root_litter)
                self.lnc[:, step] = masked_mean_2D(self.metacomm.mask, lnc)

                # Soil C:N:P balance and OM decomposition
                s_out = soil_dec.carbon3(self.soil_temp, wtot / self.wmax_mm, self.litter_l[step],
                                         self.cwd[step], self.litter_fr[step], self.lnc[:, step],
                                         self.sp_csoil, self.sp_snc)
                soil_out = catch_out_carbon3(s_out)

                # Organic C N & P
                self.sp_csoil = soil_out['cs']
                self.sp_snc = soil_out['snc']
                idx = np.where(self.sp_snc < 0.0)[0]
                if len(idx) > 0:
                    self.sp_snc[idx] = 0.0

                # <- Out of the community loop

                # IF NUTRICYCLE:
                if nutri_cycle:
                    # UPDATE ORGANIC POOLS
                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()
                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()
                    self.sp_available_p += soil_out['pmin']
                    self.sp_available_n += soil_out['nmin']
                    # NUTRIENT DINAMICS
                    # Inorganic N
                    self.sp_in_n += self.sp_available_n + self.sp_so_n
                    self.sp_so_n = soil_dec.sorbed_n_equil(self.sp_in_n)
                    self.sp_available_n = soil_dec.solution_n_equil(
                        self.sp_in_n)
                    self.sp_in_n -= self.sp_so_n + self.sp_available_n
                    # Inorganic P
                    self.sp_in_p += self.sp_available_p + self.sp_so_p
                    self.sp_so_p = soil_dec.sorbed_p_equil(self.sp_in_p)
                    self.sp_available_p = soil_dec.solution_p_equil(
                        self.sp_in_p)
                    self.sp_in_p -= self.sp_so_p + self.sp_available_p
                    # Sorbed P
                    if self.pupt[1, step] > 0.75:
                        rwarn(
                            f"Puptk_SO > soP_max - 987 | in spin{s}, step{step} - {self.pupt[1, step]}")
                        self.pupt[1, step] = 0.0

                    if self.pupt[1, step] > self.sp_so_p:
                        rwarn(
                            f"Puptk_SO > soP_pool - 992 | in spin{s}, step{step} - {self.pupt[1, step]}")
                    self.sp_so_p -= self.pupt[1, step]
                    try:
                        t1 = np.all(self.sp_snc > 0.0)
                    except:
                        if self.sp_snc is None:
                            self.sp_snc = np.zeros(shape=8,)
                            t1 = True
                        elif self.sp_snc is not None:
                            t1 = True
                        rwarn(f"Exception while handling sp_snc pool")
                    if not t1:
                        self.sp_snc[np.where(self.sp_snc < 0)[0]] = 0.0
                    # ORGANIC nutrients uptake
                    # N
                    if self.nupt[1, step] < 0.0:
                        rwarn(
                            f"NuptkO < 0 - 1003 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0
                    if self.nupt[1, step] > 2.5:
                        rwarn(
                            f"NuptkO  > max - 1007 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0
                    total_on = self.sp_snc[:4].sum()
                    if total_on > 0.0:
                        frsn = [i / total_on for i in self.sp_snc[:4]]
                    else:
                        frsn = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsn):
                        self.sp_snc[i] -= self.nupt[1, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        self.sp_snc[idx] = 0.0

                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()

                    # P
                    if self.pupt[2, step] < 0.0:
                        rwarn(
                            f"PuptkO < 0  in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    if self.pupt[2, step] > 1.0:
                        rwarn(
                            f"PuptkO > max  in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    total_op = self.sp_snc[4:].sum()
                    if total_op > 0.0:
                        frsp = [i / total_op for i in self.sp_snc[4:]]
                    else:
                        frsp = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsp):
                        self.sp_snc[i + 4] -= self.pupt[2, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        self.sp_snc[idx] = 0.0

                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()

                    # Raise some warnings
                    if self.sp_organic_n < 0.0:
                        self.sp_organic_n = 0.0
                        rwarn(f"ON negative in spin{s}, step{step}")
                    if self.sp_sorganic_n < 0.0:
                        self.sp_sorganic_n = 0.0
                        rwarn(f"SON negative in spin{s}, step{step}")
                    if self.sp_organic_p < 0.0:
                        self.sp_organic_p = 0.0
                        rwarn(f"OP negative in spin{s}, step{step}")
                    if self.sp_sorganic_p < 0.0:
                        self.sp_sorganic_p = 0.0
                        rwarn(f"SOP negative in spin{s}, step{step}")

                    # CALCULATE THE EQUILIBTIUM IN SOIL POOLS
                    # Soluble and inorganic pools
                    if self.pupt[0, step] > 1e2:
                        rwarn(
                            f"Puptk > max - 786 | in spin{s}, step{step} - {self.pupt[0, step]}")
                        self.pupt[0, step] = 0.0
                    self.sp_available_p -= self.pupt[0, step]

                    if self.nupt[0, step] > 1e3:
                        rwarn(
                            f"Nuptk > max - 792 | in spin{s}, step{step} - {self.nupt[0, step]}")
                        self.nupt[0, step] = 0.0
                    self.sp_available_n -= self.nupt[0, step]
                # END SOIL NUTRIENT DYNAMICS

                if save:
                    # Plant uptake and Carbon costs of nutrient uptake
                    self.nupt[:, step] = masked_mean_2D(self.metacomm.mask, nupt)
                    self.pupt[:, step] = masked_mean_2D(self.metacomm.mask, pupt)
                    self.storage_pool[:, step] = masked_mean_2D(self.metacomm.mask, storage_pool.astype(np.float32))
                    self.carbon_costs[step] = masked_mean(self.metacomm.mask, cc)
                    self.tsoil.append(self.soil_temp)
                    self.photo[step] = masked_mean(self.metacomm.mask, photo)
                    self.aresp[step] = masked_mean(self.metacomm.mask, aresp)
                    self.npp[step] = masked_mean(self.metacomm.mask, npp)
                    self.rnpp[step] = masked_mean(self.metacomm.mask, rnpp_mt)
                    self.lai[step] = masked_mean(self.metacomm.mask, lai)
                    self.rcm[step] = masked_mean(self.metacomm.mask, rcm)
                    self.f5[step] = masked_mean(self.metacomm.mask, f5)
                    self.rm[step] = masked_mean(self.metacomm.mask, rm)
                    self.rg[step] = masked_mean(self.metacomm.mask, rg)
                    self.wue[step] = masked_mean(self.metacomm.mask, wue)
                    self.cue[step] = masked_mean(self.metacomm.mask, cue)
                    self.cdef[step] = masked_mean(self.metacomm.mask, cdef)
                    self.vcmax[step] = masked_mean(self.metacomm.mask, vcmax)
                    self.specific_la[step] = masked_mean(self.metacomm.mask, specific_la)
                    self.cleaf[step] = masked_mean(self.metacomm.mask, cleaf)
                    self.cawood[step] = masked_mean(self.metacomm.mask, cawood)
                    self.cfroot[step] = masked_mean(self.metacomm.mask, cfroot)
                    self.hresp[step] = soil_out['hr']
                    self.csoil[:, step] = soil_out['cs']
                    self.wsoil[step] = self.wp_water_upper_mm + self.wp_water_lower_mm
                    self.inorg_n[step] = self.sp_in_n
                    self.inorg_p[step] = self.sp_in_p
                    self.sorbed_n[step] = self.sp_so_n
                    self.sorbed_p[step] = self.sp_so_p
                    self.snc[:, step] = soil_out['snc']
                    self.nmin[step] = self.sp_available_n
                    self.pmin[step] = self.sp_available_p
                    self.ocp_area[:,:, step] = ocp_area
                    self.lim_status[:, :, :, step] = lim_status
                    self.uptake_strategy[:, :, :, step] = uptake_strategy
                    self.ls[step] = living_pls

            # <- Out of the daily loop
            sv: Thread
            if save:
                if s > 0:
                    while True:
                        if sv.is_alive(): # type: ignore
                            sleep(0.5)
                        else:
                            self.flush_data = None
                            break
                self.executed_iterations.append((start_date, end_date))
                self.flush_data = self._flush_output(
                    'spin', (start_index, end_index))
                sv = Thread(target=self._save_output, args=(self.flush_data,))
                sv.start()
        # Finish the last thread
        # <- Out of spin loop
        if save:
            while True:
                if sv.is_alive():
                    sleep(0.5)
                else:
                    self.flush_data = None
                    break
        # Restablish new communities in the end, if applicable
        if kill_and_reset:
            for community in self.metacomm:
                # with lock:
                new_life_strategies = self.get_from_main_array(community.npls)
                community.restore_from_main_table(new_life_strategies)
            # Here we update the metacomm mask to ensure that all communities are active again
            self.metacomm.update_mask()
        return None


    def __fetch_spin_data(self, spin) -> dict:
        """Get the data from a spin file"""
        if len(self.outputs) == 0:
            raise AssertionError("No output data available. Run the model first")
        if spin < 10:
            name = f'spin000{spin}.pkz'
        elif spin < 100:
            name = f'spin00{spin}.pkz'
        elif spin < 1000:
            name = f'spin0{spin}.pkz'
        else:
            name = f'spin{spin}.pkz'

        with open(self.outputs[name], 'rb') as fh:
            spin_dt = load(fh)
        return spin_dt


    def _read_daily_output(self, period: Union[int, Tuple[int, int], None] = None) -> Union[Tuple, List[Any]]:
        """Read the daily output for this gridcell.

        Warning: This method assumes that the ouptut files are time-ordered
        """
        assert len(self.outputs) > 0, "No output data available. Run the model first"

        if isinstance(period, int):
            assert period > 0, "Period must be positive"
            assert period <= self.run_counter, "Period must be less than the number of spins"
            with ThreadPoolExecutor(max_workers=1) as executor:
                return [executor.submit(self.__fetch_spin_data, period)]
            # return data, self.executed_iterations[period]
        elif isinstance(period, tuple):
            assert period[1] <= self.run_counter, "Period must be less than the number of spins" # type: ignore
            assert period[0] < period[1], "Period must be a tuple with the start and end spins" # type: ignore
            spins = range(period[0], period[1] + 1) # type: ignore
            files = tuple((f"spin{x}" for x in spins))
        else:
            files:Tuple[str, ...] = tuple(path_str for path_str in self.outputs.values())
            spins = range(1, len(files) + 1)

        with ThreadPoolExecutor(max_workers=len(files)) as executor:
            futures = [executor.submit(self.__fetch_spin_data, spin) for spin in spins]
        return futures


    def _reader_array(self, ndim:int, sample_data:NDArray) -> NDArray:
        """
        Return a numpy array with one extra dimension.
        This array is used to concatenate results comming from different spins

        Args:
            ndim (int): Number of dimensions in the array
            sample_data (np.ndarray): Sample data to build the array

        Raises:
            ValueError: If the number of dimensions is less than 1
            NotImplementedError: If the number of dimensions is greater than 4

        Returns:
            np.ndarray: An array with one extra dimension and same dtype as sample_data
        """
        if not ndim:
            raise ValueError("Need the array dimensions to process. Stopping")
        if ndim > 4:
            raise NotImplementedError("Arrays with more than 4 axis are not supported")
        if ndim == 1:
            output = np.zeros(0, dtype=sample_data.dtype)
        elif ndim == 2:
            s = (sample_data.shape[0], 0)
            output = np.zeros(s, dtype=sample_data.dtype)
        elif ndim == 3:
            s = (sample_data.shape[0],
                self.ncomms,
                0)
            output = np.zeros(s, dtype=sample_data.dtype)
        elif ndim == 4:
            s = (sample_data.shape[0],
                sample_data.shape[1],
                self.ncomms,
                0)
            output = np.zeros(s, dtype=sample_data.dtype)
        return output


    def _get_daily_data(self, variable: Union [str, Collection[str]] = "npp",
                            spin_slice: Union[int, Tuple[int, int], None] = None,
                            pp: bool=False,
                            return_time: bool=False,
                            return_array: bool=False
                            ) -> Union[List, NDArray, Tuple[NDArray, NDArray], Tuple[Dict[str, NDArray], NDArray], List[NDArray]]:
        """_summary_

        Args:
            variable (Union[str, Collection[str]], optional): variable name or names. Defaults to "npp".
            spin_slice (Optional[Tuple[int, int]], optional): Slice of spins. Defaults to None.
            pp (bool, optional): Print available variable names in the output data and exits. Defaults to False.
            return_time (bool, optional): Return a collection of time objects with the days of simulation. Defaults to False.
            return_array (bool, optional): Returns one array or a tuple of arrays. Defaults to False.


        Returns:
            Union[NDArray, List, Tuple, None]: There are four possible returns:
            - If return_array is True and return_time is False, returns an array with the values of the variable
            - If return_array is False and return_time is True, returns a tuple with a np.array of the variable
            and a list of time objects
            - If return_array is False and return_time is False, returns a list with the values of the variable.
            - If return_array is False and return_time is False, and a slice is provided, an array or
              list of arrays is returned
            - If return_array is True and return_time is True, returns a tuple with the values of the variable
            and a list of time objects
        """

        if isinstance(variable, str):
            variable = [variable,]
        assert isinstance(variable, Collection), "Variable must be a string or a collection of strings"


        result = []
        f = self._read_daily_output(period=None) if spin_slice is None else self._read_daily_output(period=spin_slice) # type: ignore
        for _read_ in f:
            result.append(_read_.result())

        # GEt start and end dates (by index)
        if len(result) == 1:
            eind = result[0]["eind"]
            sind = result[0]["sind"]
        elif len(result) > 1:
            eind = result[-1]["eind"]
            sind = result[0]["sind"]

        variable_names: Set[str] = set(result[0].keys()) # Available variable names in the output data
        variable_set: Set[str] = set(variable)

        if pp:
            print(f"Available variables: {variable_names}")
            return [] # print and exit

        not_in = variable_set - variable_names
        assert len(not_in) == 0, f"No variable(s): {not_in} found in the output data"

        output_list:List[NDArray] = []
        varnames = []
        for var in variable_set:
            sample_data = result[0][var]

            vtype = type(sample_data)

            try:
                ndim = len(sample_data.shape)
            except:
                ndim = 0

            if vtype == np.ndarray:
                varnames.append(var)
                output = self._reader_array(ndim, sample_data)
                for r in result:
                    if not ndim:
                        raise ValueError(f"Need the array dimensions to process. Stopping")
                    output = np.concatenate((output, r[var]), axis=ndim - 1)
                output_list.append(output)
            else:
                print(f"Variable {var} is not a numpy array. Skipping")
                continue

        if return_array:
            assert len(variable_set) == 1, "Only one variable can be returned as an array"
            if return_time:
                pass
            else:
                return output_list[0] # return the array

        if not return_time:
            if len(output_list) == 1:
                return output_list[0] # return the array
            return output_list        # return a list of arrays List[NDArray]
        else:
            # # Build date index. datelist will have the same length as the arrays
            # # if the files were saved in a transient run. Files saved in a spinup
            # # will result in a datelist that corresponds to the start_date-end_date range -
            # i.e., the lenght of the datelist divides the lenght of the arrays n spinup times

            datelist = cftime.num2date(np.arange(sind, eind + 1),
                                       units=self.time_unit,
                                       calendar=self.calendar)

            if len(output_list) == 1:
                return output_list[0], datelist # return the Tuple[NDArray, NDArray[datetime]]

            output_dict:Dict[str, NDArray] = dict(zip(varnames, output_list))
            return output_dict, datelist  # return the Tuple[Dict[str, NDArray], NDArray[datetime]]


    def print_available_periods(self):
        assert len(self.executed_iterations) > 0, "No output data available. Run the model first"
        for i, period in enumerate(self.executed_iterations):
            print(f"Period {i}: {period[0]} - {period[1]}")


class region:
    """Region class containing the gridcells for a given region
    """

    def __init__(self,
                name:str,
                clim_data:Union[str,Path],
                soil_data:Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
                co2:Union[str, Path],
                pls_table:NDArray)->None:
        """_summary_

        Args:
            name (str): this will be the name of the region and the name of the output folder
            clim_data (Union[str,Path]): Path for the climate data
            soil_data (Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]]): _description_
            output_folder (Union[str, Path]): _description_
            co2 (Union[str, Path]): _description_
            pls_table (np.ndarray): _description_
        """
        self.config: Config = fetch_config("caete.toml")
        self.nproc = self.config.multiprocessing.nprocs # type: ignore
        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # IO
        self.climate_files = []
        self.input_data = str_or_path(clim_data)
        self.soil_data = copy.deepcopy(soil_data)
        self.pls_table = mc.pls_table(pls_table)

        # calculate_matrix dimesnion size size from grid resolution
        self.nx = len(np.arange(0, 180, self.config.crs.xres/2)) # type: ignore
        self.ny = len(np.arange(0,  90, self.config.crs.yres/2)) # type: ignore

        self.epsg = f"EPSG:{self.config.crs.epsg_id}" # type: ignore
        self.datum = self.config.crs.datum # type: ignore

        # Grid mask of the region
        self.grid_mask = np.ones((self.ny, self.nx), dtype=bool)

        # Number of PLS in the main table (global table)
        self.npls_main_table = self.pls_table.npls

        try:
            metadata_file = list(self.input_data.glob("*_METADATA.pbz2"))[0]
        except:
            raise FileNotFoundError("Metadata file not found in the input data folder")

        try:
            mtd = str_or_path(metadata_file, check_is_file=True)
        except:
            raise AssertionError("Metadata file path could not be resolved. Cannot proceed without metadata")

        # Read metadata from climate files
        self.metadata = read_bz2_file(mtd)
        self.stime = copy.deepcopy(self.metadata[0])

        for file_path in self.input_data.glob("input_data_*-*.pbz2"):
            self.climate_files.append(file_path)

        # This is used to define the gridcells output paths
        self.yx_indices = []
        self.lats = np.zeros(len(self.climate_files), dtype=np.float32)
        self.lons = np.zeros(len(self.climate_files), dtype=np.float32)
        for f in self.climate_files:
            y, x = f.stem.split("_")[-1].split("-")
            self.yx_indices.append((int(y), int(x)))
            # Update the grid mask
            self.grid_mask[int(y), int(x)] = False

        # create the output folder structure
        # This is the output path for the regions, Create it if it does not exist
        # output_path is a global defined in parameters.py The region object will
        # create the internal output folder structure into this directory
        os.makedirs(output_path, exist_ok=True)

        # This is the output path for this region
        self.output_path = output_path/self.name
        os.makedirs(self.output_path, exist_ok=True)

        # A list to store this region's gridcells
        # Some magic methods are defined to deal with this list
        self.gridcells:List[grd_mt] = []


    def _update_config(self):
        """Update the configuration file"""
        self.config = fetch_config("caete.toml")


    def get_from_main_table(self, comm_npls, lock = lock) -> Tuple[Union[int, NDArray[np.intp]], NDArray[np.float32]]:
        """Returns a number of IDs (in the main table) and the respective
        functional identities (PLS table) to set or reset a community

        This method is passed as an argument for the gridcell class. It is used to read
        the main table and the PLS table to set or reset a community. This method
        must be called with a lock.

        Args:
        comm_npls: (int) Number of PLS in the output table (must match npls_max (see caete.toml))"""

        assert comm_npls > 0, "Number of PLS must be greater than 0"

        if comm_npls == 1:
            idx = np.random.randint(0, self.npls_main_table - 1)
            with lock:
                return idx, self.pls_table.table[:, idx]

        assert comm_npls <= self.npls_main_table, "Number of PLS must be less than the number of PLS in the main table"

        idx = np.random.randint(0, self.npls_main_table - 1, comm_npls)
        with lock:
            return idx, self.pls_table.table[:, idx]


    def set_gridcells(self):
        """_summary_
        """
        print("Starting gridcells")
        i = 0
        print_progress(i, len(self.yx_indices), prefix='Progress:', suffix='Complete')
        for f,pos in zip(self.climate_files, self.yx_indices):
            y, x = pos
            gridcell_dump_directory = self.output_path/Path(f"grd_{y}-{x}")
            grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
            grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                    tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
            self.gridcells.append(grd_cell)
            self.lats[i] = grd_cell.lat
            self.lons[i] = grd_cell.lon
            print_progress(i+1, len(self.yx_indices), prefix='Progress:', suffix='Complete')
            i += 1


    def run_region_map(self, func:Callable):
        """_summary_

        Args:
            func (Callable): _description_

        Returns:
            _type_: _description_
        """
        with mp.Pool(processes=self.nproc, maxtasksperchild=1) as p:
            self.gridcells = p.map(func, self.gridcells, chunksize=1)
        gc.collect()
        return None


    def run_region_starmap(self, func:Callable, args):
        """_summary_

        Args:
            func (Callable): _description_
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        with mp.Pool(processes=self.nproc, maxtasksperchild=1) as p:
            self.gridcells = p.starmap(func, [(gc, args) for gc in self.gridcells], chunksize=1)
        gc.collect()
        return None


    # Methods to deal with model outputs
    def clean_model_state(self):
        """
        Clean state of the region. Deletes all attributes that are not necessary

        This is useful to access model outputs after a run

        Warning: This method will erase all data related to the state of the region.
        The region cannot be used directly to run the model after this method is called.
        However, you can still use it to access the output data generated by the model.

        """
        attributes_to_keep = {'calendar',
                              'time_unit',
                              'cell_area',
                              'config',
                              'executed_iterations',
                              'lat',
                              'lon',
                              'ncomms',
                              'out_dir',
                              'outputs',
                              'run_counter',
                              'x',
                              'xres',
                              'xyname',
                              'y',
                              'yres'}

        for gridcell in self:
            all_attributes = set(gridcell.__dict__.keys())

            # # Delete attributes that are not in the subset
            for attr in all_attributes - attributes_to_keep:
                delattr(gridcell, attr)


    def get_mask(self)->np.ndarray:
        """returns region mask
        True is masked, not masked otherwise

        Returns:
            np.ndarray: region mask
        """
        return self.grid_mask


    def get_crs(self) -> Dict[str, Union[str, int, float]]:
        """Get region CRS information

        Returns:
            Dict[str, Union[str, int, float]]: Dictionary with the CRS information
        """
        return {
            'proj4': f'{self.config.crs.proj4}', # type: ignore
            'lats' : self.lats,
            'lons' : self.lons,
            'epsg' : self.epsg,
            'datum': self.datum,
            'lat_units': self.config.crs.lat_units, # type: ignore
            'lon_units': self.config.crs.lon_units, # type: ignore
            'lat_zero' : self.config.crs.lat_zero, # type: ignore
            'lon_zero' : self.config.crs.lon_zero, # type: ignore
        }

    # Magic methods
    def __getitem__(self, idx:int):
        try:
            _val_ = self.gridcells[idx]
        except IndexError:
            raise IndexError(f"Cannot get item at index {idx}. Region has {self.__len__()} gridcells")
        return _val_


    def __len__(self):
        return len(self.gridcells)


    def __iter__(self):
        yield from self.gridcells


class worker:

    """Worker functions used to run the model in parallel"""

    @staticmethod
    def create_run_breaks(start_year:int, end_year:int, interval:int):
        run_breaks_hist = []
        current_year = start_year

        # Create intervals
        while current_year + interval - 1 <= end_year:
            start_date = f"{current_year}0101"
            end_date = f"{current_year + interval - 1}1231"
            run_breaks_hist.append((start_date, end_date))
            current_year += interval

        # Adjust the last interval if it is not uniform
        if current_year <= end_year:
            start_date = f"{current_year}0101"
            end_date = f"{end_year}1231"
            run_breaks_hist.append((start_date, end_date))

        return run_breaks_hist


    @staticmethod
    def soil_pools_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in soil pools, In this phase the communities are reset if there are no PLS

        CALL:\n

        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=9, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=False, reset_community=True, kill_and_reset=True)
        """
        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=5, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def community_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in the community, In this phase, communities can be reset if there are no PLS

        CALL:\n

        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=6, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True, reset_community=True)

        """
        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=5, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def env_filter_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in the community while adding new PLS if there are free slots

        CALL:\n
        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=20, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        """
        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=10, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def final_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in the community while adding new PLS if there are free slots

        CALL:\n

        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=10, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True)
        """
        gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=3, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True)
        gc.collect()
        return gridcell


    @staticmethod
    def transient_run_brk(gridcell:grd_mt, interval:Tuple[str, str]):
        """transient run

        Args:
            gridcell (grd_mt): Gridcell object
            interval (Tuple[str, str]): Tuple with the start and end date of the interval: Date format "YYYYMMDD"


        """
        start_date, end_date = interval
        gridcell.run_gridcell(start_date, end_date, spinup=0, fixed_co2_atm_conc=None,
                              save=True, nutri_cycle=True, reset_community=False, kill_and_reset=False)
        gc.collect()
        return gridcell


    @staticmethod
    def save_state_zstd(region: Any, fname: Union[str, Path]):
        """Save a python serializable object using zstd compression with 12 threads

        Args:
            region (Any): python object to be compressed and saved. Must be serializable
            fname (Union[str, Path]): filename to save the compressed object

        Returns:
            None: None
        """
        with open(fname, 'wb') as fh:
            compressor = zstd.ZstdCompressor(level=22, threads=12)
            with compressor.stream_writer(fh) as compressor_writer:
                pkl.dump(region, compressor_writer)


    @staticmethod
    def load_state_zstd(fname:Union[str, Path]):
        """Load a python object from a zstd compressed file

        Args:
            fname (Union[str, Path]): filename of the compressed object
        """
        with open(fname, 'rb') as fh:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(fh) as decompressor_reader:
                region = pkl.load(decompressor_reader)
        return region


# OLD CAETÊ
# This is the prototype of CAETÊ that I created during my PhD.
# Will continue here for some time until I can migrate everything to the new version (ABOVE)
# This is horrible code, but it works. I will try to make it better in the new version
#

# GLOBAL variables
out_ext = ".pkz"
npls = gp.npls

NO_DATA = [-9999.0, -9999.0]


run_breaks_hist = [('19790101', '19801231'),
                   ('19810101', '19821231'),
                   ('19830101', '19841231'),
                   ('19850101', '19861231'),
                   ('19870101', '19881231'),
                   ('19890101', '19901231'),
                   ('19910101', '19921231'),
                   ('19930101', '19941231'),
                   ('19950101', '19961231'),
                   ('19970101', '19981231'),
                   ('19990101', '20001231'),
                   ('20010101', '20021231'),
                   ('20030101', '20041231'),
                   ('20050101', '20061231'),
                   ('20070101', '20081231'),
                   ('20090101', '20101231'),
                   ('20110101', '20121231'),
                   ('20130101', '20141231'),
                   ('20150101', '20161231')]

run_breaks_CMIP5_hist = [('19300101', '19391231'),
                        ('19400101', '19491231'),
                        ('19500101', '19591231'),
                        ('19600101', '19691231'),
                        ('19700101', '19791231'),
                        ('19800101', '19891231'),
                        ('19900101', '19991231'),
                        ('20000101', '20051231')]

run_breaks_CMIP5_proj = [('20060101', '20091231'),
                         ('20100101', '20191231'),
                         ('20200101', '20291231'),
                         ('20300101', '20391231'),
                         ('20400101', '20491231'),
                         ('20500101', '20591231'),
                         ('20600101', '20691231'),
                         ('20700101', '20791231'),
                         ('20800101', '20891231'),
                         ('20900101', '20991231')]

# historical and projection periods respectively
rbrk = [run_breaks_hist, run_breaks_CMIP5_hist, run_breaks_CMIP5_proj]


class grd:

    """
    Defines the gridcell object - This object stores all the input data,
    the data comming from model runs for each grid point, all the state variables and all the metadata
    describing the life cycle of the gridcell and the filepaths to the generated model outputs
    This class also provides several methods to apply the CAETÊ model with proper formated climatic and soil variables
    """

    def __init__(self, x, y, dump_folder):
        """Construct the gridcell object"""

        # CELL Identifiers
        self.x = x                            # Grid point x coordinate
        self.y = y                            # Grid point y coordinate
        self.xyname = str(y) + '-' + str(x)   # IDENTIFIES GRIDCELLS
        self.plot_name = dump_folder
        self.plot = None
        self.input_fname = f"input_data_{self.xyname}.pbz2"
        self.input_fpath = None
        self.data = None
        self.pos = (int(self.x), int(self.y))
        self.pls_table = None   # will receive the np.array with functional traits data
        self.outputs = {}       # dict, store filepaths of output data
        self.realized_runs = []
        self.experiments = 1
        # counts the execution of a time slice (a call of self.run_spinup)
        self.run_counter = 0
        self.neighbours = None

        self.ls = None          # Number of surviving plss//
        self.grid_filename = f"gridcell{self.xyname}"
        self.out_dir = Path(
            "../outputs/{}/gridcell{}/".format(dump_folder, self.xyname)).resolve()
        self.flush_data = None

        # Time attributes
        self.time_index = None  # Array with the time stamps
        self.calendar = None    # Calendar name
        self.time_unit = None   # Time unit
        self.start_date = None
        self.end_date = None
        self.ssize = None
        self.sind = None
        self.eind = None

        # Input data
        self.filled = False     # Indicates when the gridcell is filled with input data
        self.pr = None
        self.ps = None
        self.rsds = None
        self.tas = None
        self.rhs = None

        # OUTPUTS
        self.soil_temp = None
        self.emaxm = None
        self.tsoil = None
        self.photo = None
        self.ls  = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.swsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.lim_status = None
        self.uptake_strategy = None
        self.carbon_costs = None

        # WATER POOLS
        # Water content for each soil layer
        self.wp_water_upper_mm = None  # mm
        self.wp_water_lower_mm = None  # mm
        # Saturation point
        self.wmax_mm = None  # mm

        # SOIL POOLS
        self.input_nut = None
        self.sp_available_p = None
        self.sp_available_n = None
        self.sp_so_n = None
        self.sp_in_n = None
        self.sp_so_p = None
        self.sp_in_p = None
        self.sp_csoil = None
        self.sp_snr = None
        self.sp_uptk_costs = None
        self.sp_organic_n = None
        self.sp_sorganic_n = None
        self.sp_organic_p = None
        self.sp_sorganic_p = None

        # CVEG POOLS
        self.vp_cleaf = None
        self.vp_croot = None
        self.vp_cwood = None
        self.vp_dcl = None
        self.vp_dca = None
        self.vp_dcf = None
        self.vp_ocp = None
        self.vp_wdl = None
        self.vp_sto = None
        self.vp_lsid = None

        # Hydraulics
        self.theta_sat = None
        self.psi_sat = None
        self.soil_texture = None

    def _allocate_output_nosave(self, n):
        """allocate space for some tracked variables during spinup
        n: int NUmber of days being simulated"""

        self.runom = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(2, n), order='F')
        self.pupt = np.zeros(shape=(3, n), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')

    def _allocate_output(self, n, npls=npls):
        """allocate space for the outputs
        n: int NUmber of days being simulated"""
        self.emaxm = []
        self.tsoil = []
        self.photo = np.zeros(shape=(n,), order='F')
        self.aresp = np.zeros(shape=(n,), order='F')
        self.npp = np.zeros(shape=(n,), order='F')
        self.lai = np.zeros(shape=(n,), order='F')
        self.csoil = np.zeros(shape=(4, n), order='F')
        self.inorg_n = np.zeros(shape=(n,), order='F')
        self.inorg_p = np.zeros(shape=(n,), order='F')
        self.sorbed_n = np.zeros(shape=(n,), order='F')
        self.sorbed_p = np.zeros(shape=(n,), order='F')
        self.snc = np.zeros(shape=(8, n), order='F')
        self.hresp = np.zeros(shape=(n,), order='F')
        self.rcm = np.zeros(shape=(n,), order='F')
        self.f5 = np.zeros(shape=(n,), order='F')
        self.runom = np.zeros(shape=(n,), order='F')
        self.evapm = np.zeros(shape=(n,), order='F')
        self.wsoil = np.zeros(shape=(n,), order='F')
        self.swsoil = np.zeros(shape=(n,), order='F')
        self.rm = np.zeros(shape=(n,), order='F')
        self.rg = np.zeros(shape=(n,), order='F')
        self.cleaf = np.zeros(shape=(n,), order='F')
        self.cawood = np.zeros(shape=(n,), order='F')
        self.cfroot = np.zeros(shape=(n,), order='F')
        self.wue = np.zeros(shape=(n,), order='F')
        self.cue = np.zeros(shape=(n,), order='F')
        self.cdef = np.zeros(shape=(n,), order='F')
        self.nmin = np.zeros(shape=(n,), order='F')
        self.pmin = np.zeros(shape=(n,), order='F')
        self.vcmax = np.zeros(shape=(n,), order='F')
        self.specific_la = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(2, n), order='F')
        self.pupt = np.zeros(shape=(3, n), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')
        self.carbon_costs = np.zeros(shape=(n,), order='F')

        self.area = np.zeros(shape=(npls, n), order='F')
        self.lim_status = np.zeros(
            shape=(3, npls, n), dtype=np.dtype('int16'), order='F')
        self.uptake_strategy = np.zeros(
            shape=(2, npls, n), dtype=np.dtype('int32'), order='F')

    def _flush_output(self, run_descr, index):
        """1 - Clean variables that receive outputs from the fortran subroutines
           2 - Fill self.outputs dict with filepats of output data
           3 - Returns the output data to be writen

           runs_descr: str a name for the files
           index = tuple or list with the first and last values of the index time variable"""
        to_pickle = {}
        self.run_counter += 1
        if self.run_counter < 10:
            spiname = run_descr + "0" + str(self.run_counter) + out_ext
        else:
            spiname = run_descr + str(self.run_counter) + out_ext

        self.outputs[spiname] = os.path.join(self.out_dir, spiname)
        to_pickle = {'emaxm': np.array(self.emaxm),
                     "tsoil": np.array(self.tsoil),
                     "photo": self.photo,
                     "aresp": self.aresp,
                     'npp': self.npp,
                     'lai': self.lai,
                     'csoil': self.csoil,
                     'inorg_n': self.inorg_n,
                     'inorg_p': self.inorg_p,
                     'sorbed_n': self.sorbed_n,
                     'sorbed_p': self.sorbed_p,
                     'snc': self.snc,
                     'hresp': self.hresp,
                     'rcm': self.rcm,
                     'f5': self.f5,
                     'runom': self.runom,
                     'evapm': self.evapm,
                     'wsoil': self.wsoil,
                     'swsoil': self.swsoil,
                     'rm': self.rm,
                     'rg': self.rg,
                     'cleaf': self.cleaf,
                     'cawood': self.cawood,
                     'cfroot': self.cfroot,
                     'area': self.area,
                     'wue': self.wue,
                     'cue': self.cue,
                     'cdef': self.cdef,
                     'nmin': self.nmin,
                     'pmin': self.pmin,
                     'vcmax': self.vcmax,
                     'specific_la': self.specific_la,
                     'nupt': self.nupt,
                     'pupt': self.pupt,
                     'litter_l': self.litter_l,
                     'cwd': self.cwd,
                     'litter_fr': self.litter_fr,
                     'lnc': self.lnc,
                     'ls': self.ls,
                     'lim_status': self.lim_status,
                     'c_cost': self.carbon_costs,
                     'u_strat': self.uptake_strategy,
                     'storage_pool': self.storage_pool,
                     'calendar': self.calendar,    # Calendar name
                     'time_unit': self.time_unit,   # Time unit
                     'sind': index[0],
                     'eind': index[1]}
        # Flush attrs
        self.emaxm = []
        self.tsoil = []
        self.photo = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.swsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.ls = None
        self.ls_id = None
        self.lim_status = None
        self.carbon_costs = None,
        self.uptake_strategy = None

        return to_pickle

    def _save_output(self, data_obj):
        """Compress and save output data
        data_object: dict; the dict returned from _flush_output"""
        if self.run_counter < 10:
            fpath = "spin{}{}{}".format(0, self.run_counter, out_ext)
        else:
            fpath = "spin{}{}".format(self.run_counter, out_ext)
        with open(self.outputs[fpath], 'wb') as fh:
            dump(data_obj, fh, compress=('lz4', 9), protocol=4) # type: ignore
        self.flush_data = 0

    def init_caete_dyn(self, input_fpath, stime_i, co2, pls_table, tsoil, ssoil, hsoil):
        """ PREPARE A GRIDCELL TO RUN
            input_fpath:(str or pathlib.Path) path to Files with climate and soil data
            co2: (list) a alist (association list) with yearly cCO2 ATM data(yyyy\t[CO2]atm\n)
            pls_table: np.ndarray with functional traits of a set of PLant life strategies
        """

        assert self.filled == False, "already done"
        self.input_fpath = Path(os.path.join(input_fpath, self.input_fname))
        assert self.input_fpath.exists()

        with bz2.BZ2File(self.input_fpath, mode='r') as fh:
            self.data = pkl.load(fh)

        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = 0

        # # Metacomunity
        # self.metacomm = metacommunity(pls_table=self.pls_table)

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # OTHER INPUTS
        self.pls_table = copy.deepcopy(pls_table)
        # self.neighbours = neighbours_index(self.pos, mask)
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        self.tsoil = []
        self.emaxm = []

        # STATE
        # Water
        self.ws1 = tsoil[0][self.y, self.x].copy()
        self.fc1 = tsoil[1][self.y, self.x].copy()
        self.wp1 = tsoil[2][self.y, self.x].copy()
        self.ws2 = ssoil[0][self.y, self.x].copy()
        self.fc2 = ssoil[1][self.y, self.x].copy()
        self.wp2 = ssoil[2][self.y, self.x].copy()

        self.swp = soil_water(self.ws1, self.ws2, self.fc1, self.fc2, self.wp1, self.wp2)
        self.wp_water_upper_mm = self.swp.w1
        self.wp_water_lower_mm = self.swp.w2
        self.wmax_mm = np.float64(self.swp.w1_max + self.swp.w2_max)

        self.theta_sat = hsoil[0][self.y, self.x].copy()
        self.psi_sat = hsoil[1][self.y, self.x].copy()
        self.soil_texture = hsoil[2][self.y, self.x].copy()

        # Biomass
        self.vp_cleaf = np.random.uniform(0.3,0.4,npls)#np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_croot = np.random.uniform(0.3,0.4,npls)#np.zeros(shape=(npls,), order='F') + 0.1
        self.vp_cwood = np.random.uniform(5.0,6.0,npls)#np.zeros(shape=(npls,), order='F') + 0.1

        self.vp_cwood[pls_table[6,:] == 0.0] = 0.0

        a, b, c, d = m.pft_area_frac(
            self.vp_cleaf, self.vp_croot, self.vp_cwood, self.pls_table[6, :])
        del b # not used
        del c # not used
        del d # not used
        self.vp_lsid = np.where(a > 0.0)[0]
        self.ls = self.vp_lsid.size
        self.vp_dcl = np.zeros(shape=(npls,), order='F')
        self.vp_dca = np.zeros(shape=(npls,), order='F')
        self.vp_dcf = np.zeros(shape=(npls,), order='F')
        self.vp_ocp = np.zeros(shape=(npls,), order='F')
        self.vp_sto = np.zeros(shape=(3, npls), order='F')

        # # # SOIL
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 0.001
        self.sp_snc = np.zeros(shape=(8,), order='F') + 0.0001
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
        self.sp_uptk_costs = np.zeros(npls, order='F')
        self.sp_organic_n = 0.1 * self.soil_dict['tn']
        self.sp_sorganic_n = 0.1 * self.soil_dict['tn']
        self.sp_organic_p = 0.5 * self.soil_dict['op']
        self.sp_sorganic_p = self.soil_dict['op'] - self.sp_organic_p

        self.outputs = dict()
        self.filled = True
        return None

    def clean_run(self, dump_folder, save_id):
        abort = False
        mem = str(self.out_dir)
        self.out_dir = Path(
            "../outputs/{}/gridcell{}/".format(dump_folder, self.xyname)).resolve()
        try:
            os.makedirs(str(self.out_dir), exist_ok=False)
        except FileExistsError:
            abort = True
            print(
                f"Folder {dump_folder} already exists. You cannot orerwrite its contents")
        finally:
            assert self.out_dir.exists(), f"Failed to create {self.out_dir}"

        if abort:
            print("ABORTING")
            self.out_dir = Path(mem)
            print(
                f"Returning the original grd_{self.xyname}.out_dir to {self.out_dir}")
            raise RuntimeError

        self.realized_runs.append((save_id, self.outputs.copy()))
        self.outputs = {}
        self.run_counter = 0
        self.experiments += 1

    def change_clim_input(self, input_fpath, stime_i, co2):

        self.input_fpath = Path(os.path.join(input_fpath, self.input_fname))
        assert self.input_fpath.exists()

        with bz2.BZ2File(self.input_fpath, mode='r') as fh:
            self.data = pkl.load(fh)

        self.flush_data = 0

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        return None

    def run_caete(self,
                  start_date,
                  end_date,
                  spinup=0,
                  fix_co2=None,
                  save=True,
                  nutri_cycle=True,
                  afex=False):
        """ start_date [str]   "yyyymmdd" Start model execution

            end_date   [str]   "yyyymmdd" End model execution

            spinup     [int]   Number of repetitions in spinup. 0 for no spinup

            fix_co2    [Float] Fixed value for ATM [CO2]
                       [int]   Fixed value for ATM [CO2]
                       [str]   "yyyy" Corresponding year of an ATM [CO2]

            This function run the fortran subroutines and manage data flux. It
            is the proper CAETÊ-DVM execution in the start_date - end_date period
        """

        assert self.filled, "The gridcell has no input data"
        assert not fix_co2 or type(
            fix_co2) == str or fix_co2 > 0, "A fixed value for ATM[CO2] must be a positive number greater than zero or a proper string "
        ABORT = 0
        if self.plot is True:
            splitter = ","
        else:
            splitter = "\t"

        def find_co2(year):
            for i in self.co2_data:
                if int(i.split(splitter)[0]) == year:
                    return float(i.split(splitter)[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates (read actual arguments)
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)
        spin = 1 if spinup == 0 else spinup

        # Catch climatic input and make conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # transforamando de Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.5 / 2.18e5
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0
        next_year = 0.0

        fix_co2_p = False
        if fix_co2 is None:
            fix_co2_p = False
        elif type(fix_co2) == int or type(fix_co2) == float:
            co2 = fix_co2
            fix_co2_p = True
        elif type(fix_co2) == str:
            assert type(int(
                fix_co2)) == int, "The string(\"yyyy\") for the fix_co2 argument must be an year between 1901-2016"
            co2 = find_co2(int(fix_co2))
            fix_co2_p = True

        for s in range(spin):
            if ABORT:
                pID = os.getpid()
                print(f'Closed process PID = {pID}\nGRD = {self.plot_name}\nCOORD = {self.pos}')
                break
            if save:
                self._allocate_output(steps.size)
                self.save = True
            else:
                self._allocate_output_nosave(steps.size)
                self.save = False
            for step in range(steps.size):
                if fix_co2_p:
                    pass
                else:
                    loop += 1
                    count_days += 1
                    # CAST CO2 ATM CONCENTRATION
                    days = 366 if m.leap(year0) == 1 else 365
                    if count_days == days:
                        count_days = 0
                        year0 = cftime.num2date(day_indexes[step],
                                                self.time_unit, self.calendar).year
                        co2 = find_co2(year0)
                        next_year = (find_co2(year0 + 1) - co2) / days

                    elif loop == 1 and count_days < days:
                        year0 = start.year
                        next_year = (find_co2(year0 + 1) - co2) / \
                            (days - count_days)

                    co2 += next_year

                # Update soil temperature
                self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

                # AFEX
                if count_days == 364 and afex:
                    with open("afex.cfg", 'r') as afex_cfg:
                        afex_exp = afex_cfg.readlines()
                    afex_exp = afex_exp[0].strip()
                    if afex_exp == 'N':
                        # (12.5 g m-2 y-1 == 125 kg ha-1 y-1)
                        self.sp_available_n += 12.5
                    elif afex_exp == 'P':
                        # (5 g m-2 y-1 == 50 kg ha-1 y-1)
                        self.sp_available_p += 5.0
                    elif afex_exp == 'NP':
                        self.sp_available_n += 12.5
                        self.sp_available_p += 5.0

                # INFLATe VARS
                sto = np.zeros(shape=(3, npls), order='F')
                cleaf = np.zeros(npls, order='F')
                cwood = np.zeros(npls, order='F')
                croot = np.zeros(npls, order='F')
                # dcl = np.zeros(npls, order='F')
                # dca = np.zeros(npls, order='F')
                # dcf = np.zeros(npls, order='F')
                uptk_costs = np.zeros(npls, order='F')

                sto[0, self.vp_lsid] = self.vp_sto[0, :]
                sto[1, self.vp_lsid] = self.vp_sto[1, :]
                sto[2, self.vp_lsid] = self.vp_sto[2, :]
                # Just Check the integrity of the data
                assert self.vp_lsid.size == self.vp_cleaf.size, 'different array sizes'
                c = 0
                for n in self.vp_lsid:
                    cleaf[n] = self.vp_cleaf[c]
                    cwood[n] = self.vp_cwood[c]
                    croot[n] = self.vp_croot[c]
                    # dcl[n] = self.vp_dcl[c]
                    # dca[n] = self.vp_dca[c]
                    # dcf[n] = self.vp_dcf[c]
                    uptk_costs[n] = self.sp_uptk_costs[c]
                    c += 1
                ton = self.sp_organic_n #+ self.sp_sorganic_n
                top = self.sp_organic_p #+ self.sp_sorganic_p
                # TODO need to adapt no assimilation with water content lower than thw wilting point
                # self.swp.w1_max ...
                out = model.daily_budget(self.pls_table, self.wp_water_upper_mm, self.wp_water_lower_mm,
                                         self.soil_temp, temp[step], p_atm[step],
                                         ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                         ton, top, self.sp_organic_p, co2, sto, cleaf, cwood, croot,
                                         uptk_costs, self.wmax_mm)

                # del sto, cleaf, cwood, croot, dcl, dca, dcf, uptk_costs
                # Create a dict with the function output
                daily_output = catch_out_budget(out)

                self.vp_lsid = np.where(daily_output['ocpavg'] > 0.0)[0]
                self.vp_ocp = daily_output['ocpavg'][self.vp_lsid]
                self.ls[step] = self.vp_lsid.size

                if self.vp_lsid.size < 1 and not save:
                    self.vp_lsid = np.sort(
                        np.array(
                            rd.sample(list(np.arange(gp.npls)), int(gp.npls - 5))))
                    rwarn(
                        f"Gridcell {self.xyname} has no living Plant Life Strategies - Re-populating")
                    # REPOPULATE]
                    # UPDATE vegetation pools
                    self.vp_cleaf = np.zeros(shape=(self.vp_lsid.size,)) + 0.01
                    self.vp_cwood = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_croot = np.zeros(shape=(self.vp_lsid.size,)) + 0.01
                    awood = self.pls_table[6, :]
                    for i0, i in enumerate(self.vp_lsid):
                        if awood[i] > 0.0:
                            self.vp_cwood[i0] = 0.01

                    self.vp_dcl = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_dca = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_dcf = np.zeros(shape=(self.vp_lsid.size,))
                    self.vp_sto = np.zeros(shape=(3, self.vp_lsid.size))
                    self.sp_uptk_costs = np.zeros(shape=(self.vp_lsid.size,))

                    self.vp_ocp = np.zeros(shape=(self.vp_lsid.size,))
                    del awood
                    self.ls[step] = self.vp_lsid.size
                else:
                    if self.vp_lsid.size < 1:
                        ABORT = 1
                        rwarn(f"Gridcell {self.xyname} has"  + \
                               " no living Plant Life Strategies")
                    # UPDATE vegetation pools
                    self.vp_cleaf = daily_output['cleafavg_pft'][self.vp_lsid]
                    self.vp_cwood = daily_output['cawoodavg_pft'][self.vp_lsid]
                    self.vp_croot = daily_output['cfrootavg_pft'][self.vp_lsid]
                    # self.vp_dcl = daily_output['delta_cveg'][0][self.vp_lsid]
                    # self.vp_dca = daily_output['delta_cveg'][1][self.vp_lsid]
                    # self.vp_dcf = daily_output['delta_cveg'][2][self.vp_lsid]
                    self.vp_sto = daily_output['stodbg'][:, self.vp_lsid]
                    self.sp_uptk_costs = daily_output['npp2pay'][self.vp_lsid]

                # UPDATE STATE VARIABLES
                # WATER CWM
                self.runom[step] = self.swp._update_pool(
                    prec[step], daily_output['evavg'])
                self.swp.w1 = np.float64(
                    0.0) if self.swp.w1 < 0.0 else self.swp.w1
                self.swp.w2 = np.float64(
                    0.0) if self.swp.w2 < 0.0 else self.swp.w2
                self.wp_water_upper_mm = self.swp.w1
                self.wp_water_lower_mm = self.swp.w2

                # Plant uptake and Carbon costs of nutrient uptake
                self.nupt[:, step] = daily_output['nupt']
                self.pupt[:, step] = daily_output['pupt']

                # CWM of STORAGE_POOL
                for i in range(3):
                    self.storage_pool[i, step] = np.sum(
                        self.vp_ocp * self.vp_sto[i])

                # OUTPUTS for SOIL CWM
                self.litter_l[step] = daily_output['litter_l'] + \
                    daily_output['cp'][3]
                self.cwd[step] = daily_output['cwd']
                self.litter_fr[step] = daily_output['litter_fr']
                self.lnc[:, step] = daily_output['lnc']
                wtot = self.wp_water_upper_mm + self.wp_water_lower_mm
                s_out = soil_dec.carbon3(self.soil_temp, wtot / self.wmax_mm, self.litter_l[step],
                                         self.cwd[step], self.litter_fr[step], self.lnc[:, step],
                                         self.sp_csoil, self.sp_snc)

                soil_out = catch_out_carbon3(s_out)

                # Organic C N & P
                self.sp_csoil = soil_out['cs']
                self.sp_snc = soil_out['snc']
                idx = np.where(self.sp_snc < 0.0)[0]
                if len(idx) > 0:
                    for i in idx:
                        self.sp_snc[i] = 0.0

                # IF NUTRICYCLE:
                if nutri_cycle:
                    # UPDATE ORGANIC POOLS
                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()
                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()
                    self.sp_available_p += soil_out['pmin']
                    self.sp_available_n += soil_out['nmin']
                    # NUTRIENT DINAMICS
                    # Inorganic N
                    self.sp_in_n += self.sp_available_n + self.sp_so_n
                    self.sp_so_n = soil_dec.sorbed_n_equil(self.sp_in_n)
                    self.sp_available_n = soil_dec.solution_n_equil(
                        self.sp_in_n)
                    self.sp_in_n -= self.sp_so_n + self.sp_available_n

                    # Inorganic P
                    self.sp_in_p += self.sp_available_p + self.sp_so_p
                    self.sp_so_p = soil_dec.sorbed_p_equil(self.sp_in_p)
                    self.sp_available_p = soil_dec.solution_p_equil(
                        self.sp_in_p)
                    self.sp_in_p -= self.sp_so_p + self.sp_available_p

                    # Sorbed P
                    if self.pupt[1, step] > 0.75:
                        rwarn(
                            f"Puptk_SO > soP_max - 987 | in spin{s}, step{step} - {self.pupt[1, step]}")
                        self.pupt[1, step] = 0.0

                    if self.pupt[1, step] > self.sp_so_p:
                        rwarn(
                            f"Puptk_SO > soP_pool - 992 | in spin{s}, step{step} - {self.pupt[1, step]}")

                    self.sp_so_p -= self.pupt[1, step]

                    try:
                        t1 = np.all(self.sp_snc > 0.0)
                    except:
                        if self.sp_snc is None:
                            self.sp_snc = np.zeros(shape=8,)
                            t1 = True
                        elif self.sp_snc is not None:
                            t1 = True
                        rwarn(f"Exception while handling sp_snc pool")
                    if not t1:
                        self.sp_snc[np.where(self.sp_snc < 0)[0]] = 0.0
                    # ORGANIC nutrients uptake
                    # N
                    if self.nupt[1, step] < 0.0:
                        rwarn(
                            f"NuptkO < 0 - 1003 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0
                    if self.nupt[1, step] > 2.5:
                        rwarn(
                            f"NuptkO  > max - 1007 | in spin{s}, step{step} - {self.nupt[1, step]}")
                        self.nupt[1, step] = 0.0

                    total_on = self.sp_snc[:4].sum()

                    if total_on > 0.0:
                        frsn = [i / total_on for i in self.sp_snc[:4]]
                    else:
                        frsn = [0.0, 0.0, 0.0, 0.0]

                    for i, fr in enumerate(frsn):
                        self.sp_snc[i] -= self.nupt[1, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        for i in idx:
                            self.sp_snc[i] = 0.0

                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()

                    # P
                    if self.pupt[2, step] < 0.0:
                        rwarn(
                            f"PuptkO < 0 - 1020 | in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    if self.pupt[2, step] > 1.0:
                        rwarn(
                            f"PuptkO  > max - 1024 | in spin{s}, step{step} - {self.pupt[2, step]}")
                        self.pupt[2, step] = 0.0
                    total_op = self.sp_snc[4:].sum()
                    if total_op > 0.0:
                        frsp = [i / total_op for i in self.sp_snc[4:]]
                    else:
                        frsp = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsp):
                        self.sp_snc[i + 4] -= self.pupt[2, step] * fr

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        for i in idx:
                            self.sp_snc[i] = 0.0

                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()

                    # Raise some warnings
                    if self.sp_organic_n < 0.0:
                        self.sp_organic_n = 0.0
                        rwarn(f"ON negative in spin{s}, step{step}")
                    if self.sp_sorganic_n < 0.0:
                        self.sp_sorganic_n = 0.0
                        rwarn(f"SON negative in spin{s}, step{step}")
                    if self.sp_organic_p < 0.0:
                        self.sp_organic_p = 0.0
                        rwarn(f"OP negative in spin{s}, step{step}")
                    if self.sp_sorganic_p < 0.0:
                        self.sp_sorganic_p = 0.0
                        rwarn(f"SOP negative in spin{s}, step{step}")

                    # CALCULATE THE EQUILIBTIUM IN SOIL POOLS
                    # Soluble and inorganic pools
                    if self.pupt[0, step] > 1e2:
                        rwarn(
                            f"Puptk > max - 786 | in spin{s}, step{step} - {self.pupt[0, step]}")
                        self.pupt[0, step] = 0.0
                    self.sp_available_p -= self.pupt[0, step]

                    if self.nupt[0, step] > 1e3:
                        rwarn(
                            f"Nuptk > max - 792 | in spin{s}, step{step} - {self.nupt[0, step]}")
                        self.nupt[0, step] = 0.0
                    self.sp_available_n -= self.nupt[0, step]

                # END SOIL NUTRIENT DYNAMICS

                # # #  store (np.array) outputs
                if save:
                    assert self.save == True
                    self.carbon_costs[step] = daily_output['c_cost_cwm']
                    self.emaxm.append(daily_output['epavg'])
                    self.tsoil.append(self.soil_temp)
                    self.photo[step] = daily_output['phavg']
                    self.aresp[step] = daily_output['aravg']
                    self.npp[step] = daily_output['nppavg']
                    self.lai[step] = daily_output['laiavg']
                    self.rcm[step] = daily_output['rcavg']
                    self.f5[step] = daily_output['f5avg']
                    self.evapm[step] = daily_output['evavg']
                    self.wsoil[step] = self.wp_water_upper_mm
                    self.swsoil[step] = self.wp_water_lower_mm
                    self.rm[step] = daily_output['rmavg']
                    self.rg[step] = daily_output['rgavg']
                    self.wue[step] = daily_output['wueavg']
                    self.cue[step] = daily_output['cueavg']
                    self.cdef[step] = daily_output['c_defavg']
                    self.vcmax[step] = daily_output['vcmax']
                    self.specific_la[step] = daily_output['specific_la']
                    self.cleaf[step] = daily_output['cp'][0]
                    self.cawood[step] = daily_output['cp'][1]
                    self.cfroot[step] = daily_output['cp'][2]
                    self.hresp[step] = soil_out['hr']
                    self.csoil[:, step] = soil_out['cs']
                    self.inorg_n[step] = self.sp_in_n
                    self.inorg_p[step] = self.sp_in_p
                    self.sorbed_n[step] = self.sp_so_n
                    self.sorbed_p[step] = self.sp_so_p
                    self.snc[:, step] = soil_out['snc']
                    self.nmin[step] = self.sp_available_n
                    self.pmin[step] = self.sp_available_p
                    self.area[self.vp_lsid, step] = self.vp_ocp
                    self.lim_status[:, self.vp_lsid,
                                    step] = daily_output['limitation_status'][:, self.vp_lsid]
                    self.uptake_strategy[:, self.vp_lsid,
                                         step] = daily_output['uptk_strat'][:, self.vp_lsid]
                if ABORT:
                    rwarn("NO LIVING PLS - ABORT")
            if save:
                if s > 0:
                    while True:
                        if sv.is_alive():
                            sleep(0.5)
                        else:
                            break

                self.flush_data = self._flush_output(
                    'spin', (start_index, end_index))
                sv = Thread(target=self._save_output, args=(self.flush_data,))
                sv.start()
        if save:
            while True:
                if sv.is_alive():
                    sleep(0.5)
                else:
                    break
        return None

    def bdg_spinup(self, start_date, end_date):
        """SPINUP SOIL POOLS - generate soil OM and Organic nutrients inputs for soil spinup
        - Side effect - Start soil water pools pools """

        assert self.filled, "The gridcell has no input data"
        self.budget_spinup = True

        if self.plot:
            splitter = ","
        else:
            splitter = "\t"

        def find_co2(year):
            for i in self.co2_data:
                if int(i.split(splitter)[0]) == year:
                    return float(i.split(splitter)[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)

        # Catch climatic input and make conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # transforamando de Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.5 / 2.18e5
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0
        next_year = 0
        wo = []
        llo = []
        cwdo = []
        rlo = []
        lnco = []

        sto = self.vp_sto
        cleaf = self.vp_cleaf
        cwood = self.vp_cwood
        croot = self.vp_croot
        dcl = self.vp_dcl
        dca = self.vp_dca
        dcf = self.vp_dcf
        uptk_costs = np.zeros(npls, order='F')

        for step in range(steps.size):
            loop += 1
            count_days += 1
            # CAST CO2 ATM CONCENTRATION
            days = 366 if m.leap(year0) == 1 else 365
            if count_days == days:
                count_days = 0
                year0 = cftime.num2date(day_indexes[step],
                                        self.time_unit, self.calendar).year
                co2 = find_co2(year0)
                next_year = (find_co2(year0 + 1) - co2) / days

            elif loop == 1 and count_days < days:
                year0 = start.year
                next_year = (find_co2(year0 + 1) - co2) / \
                    (days - count_days)

            co2 += next_year
            self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

            out = model.daily_budget(self.pls_table, self.wp_water_upper_mm, self.wp_water_lower_mm,
                                     self.soil_temp, temp[step], p_atm[step],
                                     ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                     self.sp_snc[:4].sum(
                                     ), self.sp_so_p, self.sp_snc[4:].sum(),
                                     co2, sto, cleaf, cwood, croot, uptk_costs, self.wmax_mm)

            # Create a dict with the function output
            daily_output = catch_out_budget(out)
            runoff = self.swp._update_pool(prec[step], daily_output['evavg'])

            self.wp_water_upper_mm = self.swp.w1
            self.wp_water_lower_mm = self.swp.w2
            # UPDATE vegetation pools

            wo.append(np.float64(self.wp_water_upper_mm + self.wp_water_lower_mm))
            llo.append(daily_output['litter_l'])
            cwdo.append(daily_output['cwd'])
            rlo.append(daily_output['litter_fr'])
            lnco.append(daily_output['lnc'])

        f = np.array
        def x(a): return a * 1.0

        return x(f(wo).mean()), x(f(llo).mean()), x(f(cwdo).mean()), x(f(rlo).mean()), x(f(lnco).mean(axis=0,))

    def sdc_spinup(self, water, ll, cwd, rl, lnc):
        """SOIL POOLS SPINUP"""

        for x in range(3000):

            s_out = soil_dec.carbon3(self.soil_temp, water / self.wmax_mm, ll, cwd, rl, lnc,
                                     self.sp_csoil, self.sp_snc)

            soil_out = catch_out_carbon3(s_out)
            self.sp_csoil = soil_out['cs']
            self.sp_snc = soil_out['snc']


class plot(grd):
    """i and j are the latitude and longitude (in that order) of plot location in decimal degrees"""

    def __init__(self, latitude, longitude, dump_folder):
        y, x = find_indices(latitude, longitude, res=0.5)
        super().__init__(x, y, dump_folder)

        self.plot = True

    def init_plot(self, sdata, stime_i, co2, pls_table, tsoil, ssoil, hsoil):
        """ PREPARE A GRIDCELL TO RUN With PLOT OBSERVED DATA
            sdata : python dict with the proper structure - see the input files e.g. CAETE-DVM/input/central/input_data_175-235.pbz2
            stime_i:  python dict with the proper structure - see the input files e.g. CAETE-DVM/input/central/ISIMIP_HISTORICAL_METADATA.pbz2
            These dicts are build upon .csv climatic data in the file CAETE-DVM/src/k34_experiment.py where you can find an application of the plot class
            co2: (list) a alist (association list) with yearly cCO2 ATM data(yyyy\t[CO2]\n)
            pls_table: np.ndarray with functional traits of a set of PLant life strategies
            tsoil, ssoil, hsoil: numpy arrays with soil parameters see the file CAETE-DVM/src/k34_experiment.py
        """

        assert self.filled == False, "already done"

        self.data = sdata

        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = 0

        self.pr = self.data['pr']
        self.ps = self.data['ps']
        self.rsds = self.data['rsds']
        self.tas = self.data['tas']
        self.rhs = self.data['hurs']

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(self.data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))
        self.data = None

        # TIME
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_index = self.stime['time_index']
        self.time_unit = self.stime['units']
        self.ssize = self.time_index.size
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # OTHER INPUTS
        self.pls_table = pls_table.copy()
        # self.neighbours = neighbours_index(self.pos, mask)
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = copy.deepcopy(co2)

        self.tsoil = []
        self.emaxm = []

        # STATE
        # Water
        self.ws1 = tsoil[0][self.y, self.x].copy()
        self.fc1 = tsoil[1][self.y, self.x].copy()
        self.wp1 = tsoil[2][self.y, self.x].copy()
        self.ws2 = ssoil[0][self.y, self.x].copy()
        self.fc2 = ssoil[1][self.y, self.x].copy()
        self.wp2 = ssoil[2][self.y, self.x].copy()

        self.swp = soil_water(self.ws1, self.ws2, self.fc1, self.fc2, self.wp1, self.wp2)
        self.wp_water_upper_mm = self.swp.w1
        self.wp_water_lower_mm = self.swp.w2
        self.wmax_mm = self.swp.w1_max + self.swp.w2_max

        self.theta_sat = hsoil[0][self.y, self.x].copy()
        self.psi_sat = hsoil[1][self.y, self.x].copy()
        self.soil_texture = hsoil[2][self.y, self.x].copy()

        # Biomass
        self.vp_cleaf = np.zeros(shape=(npls,), order='F') + 0.3
        self.vp_croot = np.zeros(shape=(npls,), order='F') + 0.3
        self.vp_cwood = np.zeros(shape=(npls,), order='F') + 0.01
        self.vp_cwood[pls_table[6,:] == 0.0] = 0.0

        a, b, c, d = m.pft_area_frac(
            self.vp_cleaf, self.vp_croot, self.vp_cwood, self.pls_table[6, :])
        self.vp_lsid = np.where(a > 0.0)[0]
        self.ls = self.vp_lsid.size
        del a, b, c, d
        self.vp_dcl = np.zeros(shape=(npls,), order='F')
        self.vp_dca = np.zeros(shape=(npls,), order='F')
        self.vp_dcf = np.zeros(shape=(npls,), order='F')
        self.vp_ocp = np.zeros(shape=(npls,), order='F')
        self.vp_sto = np.zeros(shape=(3, npls), order='F')

        # # # SOIL
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 1.0
        self.sp_snc = np.zeros(shape=(8,), order='F') + 0.1
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
        self.sp_uptk_costs = np.zeros(npls, order='F')
        self.sp_organic_n = 0.1 * self.soil_dict['tn']
        self.sp_sorganic_n = 0.1 * self.soil_dict['tn']
        self.sp_organic_p = 0.5 * self.soil_dict['op']
        self.sp_sorganic_p = self.soil_dict['op'] - self.sp_organic_p

        self.outputs = dict()
        self.filled = True

        return None


if __name__ == '__main__':

    # Short example of how to run the new version of the model. Also used to do some profiling

    from metacommunity import pls_table
    from parameters import *

    # # Read CO2 data
    co2_path = Path("../input/co2/historical_CO2_annual_1765_2018.txt")
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))


    r = region("region_test",
                   "../input/test_input",
                   (tsoil, ssoil, hsoil),
                   co2_path,
                   main_table)

    c = r.set_gridcells()

    gridcell = r[0]
    try:
        prof = sys.argv[1] == "cprof"
    except:
        prof = False
    if prof:
        import cProfile
        command = "gridcell.run_gridcell('1901-01-01', '1930-12-31', spinup=2, fixed_co2_atm_conc=1901, save=True, nutri_cycle=True, reset_community=True)"
        cProfile.run(command, sort="cumulative", filename="profile.prof")

    else:
        # gridcell.run_gridcell("1901-01-01", "1930-12-31", spinup=2, fixed_co2_atm_conc=None,
        #                                save=False, nutri_cycle=True, reset_community=True, env_filter=True, verbose=False)

        gridcell.run_gridcell("1901-01-01", "1901-12-31", spinup=5, fixed_co2_atm_conc=None,
                                       save=True, nutri_cycle=True)

        comm = gridcell.metacomm[0]
