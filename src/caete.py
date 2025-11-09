# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

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
# This module contains the classes that define the gridcell object.
# The gridcell object is the basic unit of the simulation. It contains the data and the methods to
# run the simulation for a single gridcell. It also has methods to save/process model outputs.

# The gridcell object is used by the region object (defined in region.py).
# The region object represents a collection of gridcells.
# It the methods to run the simulation for a collection of gridcells in parallel (multiprocessing).

# THe architecture of the gridcell is defined by the following classes:
# - state_zero: base class with input/output related data (paths, filenames, etc)
# - climate: class with climate data and related methods
# - time: class with time data and related methods
# - soil: class with soil data and related methods
# - gridcell_output: class to manage gridcell outputs

#  All these classes have some particular methods to manage the data
#  They are isolated in these classes to make the code more readable and maintainable.
#  All the above classes are used as base for the class that regresents a gricell
#  in the simulation:

# - grd_mt: class to manage the gridcell simulation
#     It has the methods to set up, run the simulation and save the outputs in a
#     folder. It also has some methods to process output data after model execution.
#     Each gridcell has one plant metacommunity. The plant metacommunity is a collection of
#     plant communities. Each plant community is a collection of plant life strategies (PLS).
#     At creation time, a community receives a sample of PLS from a global table. A community is not
#     allowed to have duplicated PLS but the same PLS can be in more than one community in a
#     metacommunity. Both the number of communities and the number of PLS in each community
#     are defined by the user in the configuration file called caete.toml. The number of
#     communities can be changed freely between different simulations. However, If you want
#     to change the number of PLS in a community, you must not only change the configuration
#     in the caete.toml file but also recompile the shared library using gnu-make or nmake.
#     The shared library here is the fortran code that we use to run the daily processes.

#     The global PLS table is a collection of PLS that is shared by all gridcells in the region. There is a
#     script called plsgen.py that can create a global PLS table. Run it in the src folder like this:

#     ```$ python plsgen.py -n 25000 -f ./MyPLSDataFolder```

#     This will save a file named pls_attrs<n>.csv (where n = 25000) in the local folder MyPLSDataFolder.
#     The script uses data from literature to generate a quasi-random sample of PLS based on 17 plant
#     functional traits. There is a plsgen.toml file that contains some of the parameters used
#     to generate the PLS table. There are comments in the script. Check it for more info.

#     During the simulation of a gridcell, it is possible to control the number of PLS in a community.
#     You can reset a community to a new set of PLS. This is useful when the set of PLS initially designated
#     to a community is not able to accumulate biomass in the initial stage of the simulation.
#     It is also possible to reset the entire metacommunity at once. This is useful to reset the
#     distributions of PLS after a initial spinup. In this case, te spin  aims to reach a stable state in the soil pools.
#     Right after this soil spinup, it is possible to seed new PLS in the communities while runing the
#     model. This is useful to simulate the filtering of the gridcell "environment"
#     along time while adding new PLS to the communities.
#     Given that our sample space has 17 traits, these strategies are important do start
#     the model (spin it up) and to keep it running while simulating the filtering of the
#     gridcell "environment". The grd_mt.run_gridcell method has several parameters to control
#     the simulation. You can have a look at the method signature and docstring to understand the options.


# Other important classes and functions are defined in the files region.py, worker.py, output.py,
# metacommmunity.py, and caete_jit.py.

# The region class is defined in region.py. It represents collection of gridcells.
# It has the methods to run the simulation for a collection of gridcells
# in parallel (multiprocessing).
# - region: class to manage the region simulation, the global PLS table, IO, multiprocessing etc.

# - worker (only @staticmethods): class grouping worker functions defining different phases of simulation.
#   These functions are called by the region object to run the simulation in parallel. The worker class
#   also have some utility functions to load and save data. You can save a region in a state file and
#   restart the simulation from this point. Note that in this case a entire region is saved. All relatled data
#   for each gridcell is saved and the final file can become huge. This tradeoff with the facitily to restart
#   the simulation from a specific point with a very low amount of code. The state file is compressed with zsdt

#   I am testing the model with a script called caete_driver.py. This script is a good example of how to use the model.
#   In the end of this source file (caete.py) there is some code used to test/profile the python code.

#   The old implementation of the model is in a separate branch in this repositopy: CAETE-DVM-v0.1.
#   I started to code the new implementation in the master branch while keeping the old implementation along
#   the way (Working). Several parts of the old implementation were used here.
#
# Last but not least:
#  The model uses a shared library coded in Fortran. This shared library is compiled
#  using gnu-make (Linux/MacOS) or nmake (Windows). There is a Makefile, for gnu-make, and Makefile_win for nmake.
# The main subroutine that runs the daily processes is called budget and is in the caete_module module.
# There are also other modules in the shared library: soil_dec (soil decomposition processes) and water (soil water processes).

# """

import bz2
import copy
import csv
import os
import pickle as pkl
import sys
import warnings
import time as tm

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import Thread

from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Collection, Set, TypeVar

import cftime
import numpy as np
from joblib import dump, load
from numpy.typing import NDArray

import metacommunity as mc
from _geos import calculate_area, find_coordinates_xy, find_indices_xy
from config import Config, fetch_config
from hydro_caete import soil_water
from output import DailyBudget
from caete_jit import inflate_array, masked_mean, masked_mean_2D, cw_mean
from caete_jit import shannon_entropy, shannon_evenness, shannon_diversity
from caete_jit import pft_area_frac64


# This code is only relevant in Windows systems. It adds the fortran compiler dlls to the PATH
# so the shared library can find the fortran runtime libraries of the intel one API compiler (ifx)
# Note: This is only necessary in Windows systems
if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib, caete_libs_path
    update_sys_pathlib(fortran_runtime)
    update_sys_pathlib(caete_libs_path)

# shared library
from caete_module import budget as model # type: ignore
from caete_module import soil_dec        # type: ignore
from caete_module import water as st     # type: ignore

#from memory_profiler import profile

# logging.basicConfig(filename='execution.log', level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# Output file extension
out_ext = ".pkz"

warnings.simplefilter("default")

T = TypeVar('T')

# Define some util functions #
def get_args(variable: Union[T, Collection[T]]) -> Collection[Union[T,Any]]:
    """Ensure the input is returned as a collection."""
    if isinstance(variable, Collection) and not isinstance(variable, str):
        return variable
    else:
        return [variable,]

def rwarn(txt:str='RuntimeWarning'):
    """Raise a RuntimeWarning"""
    warnings.warn(f"{txt}", RuntimeWarning)

def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=30, nl='\r'): # type: ignore
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
    percents = str_format.format(100 * (iteration / float(total))) # type: ignore
    filled_length = int(round(bar_length * iteration / float(total))) # type: ignore
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('%s%s |%s| %s%s %s' %
                     (nl, prefix, bar, percents, '%', suffix)), # type: ignore

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

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
    """Extracts co2 concentration data from a formatted text file

    Args:
        filename (Union[Path, str]): File name or path to the co2 file

    Returns:
        _type_: dictionary with the co2 data
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
    """Read a bz2 compressed file"""
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

# Timer wrapper
def timer(method):
    """_summary_

    Args:
        method (_type_): _description_

    Returns:
        _type_: _description_
    """
    @wraps(method)
    def timed(*args, **kwargs):
        from time import time
        start_time = time()
        result = method(*args, **kwargs)
        end_time = time()
        # logger.info(f"{method.__name__} took {end_time - start_time:.4f} seconds")
        # print(f"{method.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return timed


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

        y: int | float -> index in the 0 axis [zero-indexed] or geographic latitude coordinate [degrees North]  89.75 (center of northernmost cell) to -89.75 (center of southernmost cell)
        x: int | float -> index in the 1 axis [zero-indexed] or geographic longitude coordinate [degrees East] -179.75 (centrer of westernmost cell) to 179.75 (center of easternmost cell) to 179.75 
        output_dump_folder: str -> a string with a valid name to an output folder. This will be used to create a
        child directory in the output location for the region that contains this gridcell.

        """

        assert type(y) == type(x), "x and y must be of the same type"


        # TODO: Check grid data from the caete.toml file
        # Center of eg. easternmost cell is -179.75 degrees East at 0.5 degree resolution
        # Center of westernmost cell is 179.75 degrees East ...
        # the same for the latitudes
        # Center of northernmost cell is 89.75 degrees North
        # Center of southernmost cell is -89.75 degrees North
        # TODO: Test the model in a different config (res_x, res_y)
        # defined in the config file (caete.toml)

        # Configuration data
        self.config:Config = fetch_config()
        self.afex_config = self.config.fertilization # type: ignore
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

        self.input_fname = f"input_data_{self.xyname}.pbz2"
        self.station_name = f"station_{self.xyname}" # station name for the gridcell, used to fetch the climate data
        self.input_fpath = None
        self.data = None
        self.doy_months = set(self.config.doy_months) # type: ignore

        # Name of the dump folder where this gridcell will dump model outputs.
        # It is a child from ../outputs - defined in caete.toml

        # Plant life strategies table
        self.get_from_main_array = get_main_table # function to get PLS from the main table. A method of the region object is passed here at creation time.
        self.ncomms = None
        self.metacomm = None

        # Store start and end date for each "spin"
        self.executed_iterations: List[Tuple[str,str]] = []

        # OUTPUT FOLDER STRUCTURE

        self.outputs = {}       # dict, store filepaths of output data generated by this
        # Root dir for the region outputs
        self.out_dir = str_or_path(Path(output_dump_folder), check_exists=False) # gridcell output folder
        self.parent_dir = self.out_dir.parent # region output folder
        self.main_outdir = self.parent_dir.parent # main output folder

        os.makedirs(self.out_dir, exist_ok=True)
        self.flush_data = None
        self.plot_name = self.out_dir.name

        # This will be used to save the annual state of the metacommunity
        # It is a dictionary with the year as key and the path to the metacommunity state file as value
        self.metacomm_output = {}

        # counts the execution of a time slice (a call of self.run_spinup)
        # TODO: check if this is still necessary. It seems to be unused now. 
        self.run_counter = 0

class climate:
    """class with climate data"""

    def __init__(self):
        """_summary_
        """
        self.pr: NDArray[np.float32] # precipitation kg m-2 s-1
        self.ps: NDArray[np.float32]
        self.rsds: NDArray[np.float32]
        self.tas: NDArray[np.float32]
        self.rhs: NDArray[np.float32]


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


    def _set_co2(self, fpath:Union[Path, str]):
        """_summary_"""
        self.co2_path = str_or_path(fpath, check_is_file=True)
        self.co2_data = get_co2_concentration(self.co2_path)

class time:
    """_summary_
    """
    def __init__(self):
        """Time attributes"""
        self.time_index:NDArray[Union[np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]]
        self.calendar:str
        self.time_unit:str
        self.start_date:str
        self.end_date:str
        self.sind: int
        self.eind: int


    def _set_time(self, stime_i:Dict[str, Any]):
        """_summary_

        Args:
            stime_i (Dict): _description_
        """
        self.stime = copy.deepcopy(stime_i)
        self.calendar = self.stime['calendar']
        self.time_unit = self.stime['units']

        # Extract the time index from the time variable
        # Sometimes this time index can come messed from the input file
        # We get the number of days and the first day and create a new time index
        metadata_idx = self.stime['time_index'][:]
        self.ndays_total = metadata_idx.size
        self.day_zero = int(np.floor(metadata_idx[0])) # Back to midnight
        self.time_index = np.arange(self.ndays_total, dtype=np.int64) + self.day_zero

        self.sind = self.time_index[0]
        self.eind = self.time_index[-1]

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
        self.sp_csoil = None # soil carbon pools g m-2 --> 4 pools, litter 1 and 2, soil 1 and 2
        self.sp_snc = None # soil nutrient content g m-2 8 pools (N an P pools )
        self.input_nut = None # Initial values for the soil pools of nutrients (N, P)
        self.sp_available_p = None # available phosphorus
        self.sp_available_n = None # available nitrogen
        self.sp_so_n = None # soil organic nitrogen
        self.sp_in_n = None # inorganic nitrogen
        self.sp_so_p = None # soil organic phosphorus
        self.sp_in_p = None # inorganic phosphorus
        self.sp_uptk_costs = None # uptake costs g(C)m-2
        self.sp_organic_n = None # organic nitrogen 1 (Litter)
        self.sp_sorganic_n = None # organic nitrogen 2 (Soil)
        self.sp_organic_p = None # organic phosphorus 1 (Litter)
        self.sp_sorganic_p = None # organic phosphorus 2 (Soil)

        # Water
        # Water content for each soil layer. Water content under wilt point must be deducted from the water content under field capacity
        self.wp_water_upper_mm = None  # mm (water content under field capacity)
        self.wp_water_lower_mm = None  # mm (water content under field capacity)

        self.wmax_mm = None # mm (maximum water content)
        # Unused:
        # self.theta_sat = None # saturation point (RWC?)
        # self.psi_sat = None # water potential at saturation
        # self.soil_texture = None # soil texture


    def _init_soil_cnp(self, data:Dict):
        """_summary_

        Args:
            data (Dict): _description_
        """
        self.sp_csoil = np.zeros(shape=(4,), order='F') # g C m-2
        self.sp_csoil[0] = 28.155628 # Fast pool (litter)
        self.sp_csoil[1] = 315.31412 # Intermediate pool
        self.sp_csoil[2] = 771.3141  # Slow pool
        self.sp_csoil[3] = 6690.124  # Passive pool

        self.sp_snc = np.zeros(shape=(8,), order='F')
        # N:C ratios (first 4 elements) - typically ranges from 0.1 to 0.05
        self.sp_snc[0] = 0.1    # Fast N pool (litter) ~C:N = 10:1
        self.sp_snc[1] = 0.08   # Intermediate N pool ~C:N = 12.5:1
        self.sp_snc[2] = 0.067  # Slow N pool ~C:N = 15:1 
        self.sp_snc[3] = 0.05   # Passive N pool ~C:N = 20:1

        # P:C ratios (last 4 elements) - typically ranges from 0.008 to 0.002
        self.sp_snc[4] = 0.008  # Fast P pool (litter) ~C:P = 125:1
        self.sp_snc[5] = 0.005  # Intermediate P pool ~C:P = 200:1
        self.sp_snc[6] = 0.003  # Slow P pool ~C:P = 333:1
        self.sp_snc[7] = 0.002  # Passive P pool ~C:P = 500:1

        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in self.nutlist:
            self.input_nut.append(data[nut])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))

        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.4 * self.soil_dict['tn']
        self.sp_so_n = 0.2 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - (self.soil_dict['ap'] + self.soil_dict['op'] + self.soil_dict['ip'])
        self.sp_in_p = self.soil_dict['ip'] * 1.0 
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
        self.wp_water_upper_mm = self.swp.awc1
        self.wp_water_lower_mm = self.swp.awc2
        self.wmax_mm = np.float64(self.swp.w1_max + self.swp.w2_max)

        # self.theta_sat = hsoil[0][self.y, self.x].copy() # type: ignore
        # self.psi_sat = hsoil[1][self.y, self.x].copy() # type: ignore
        # self.soil_texture = hsoil[2][self.y, self.x].copy() # type: ignore


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
        self.rm:NDArray
        self.rg:NDArray
        self.wue:NDArray
        self.cue:NDArray
        self.carbon_deficit:NDArray
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
        self.carbon_deficit = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.nmin = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.pmin = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.vcmax = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.carbon_costs = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.lai = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
        self.csoil = np.zeros(shape=(4, n), order='F', dtype=np.dtype("float32"))
        self.wsoil = np.zeros(shape=(n,), order='F', dtype=np.dtype("float32"))
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
                     'rm': self.rm,
                     'rg': self.rg,
                     'wue': self.wue,
                     'cue': self.cue,
                     'cdef': self.carbon_deficit,
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
                     'c_cost': self.carbon_costs,
                     'storage_pool': self.storage_pool,
                     'calendar': self.calendar,    # type: ignore # Calendar name
                     'time_unit': self.time_unit,  # type: ignore # Time unit
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
        self.rm: NDArray = dummy_array
        self.rg: NDArray = dummy_array
        self.wue: NDArray = dummy_array
        self.cue: NDArray = dummy_array
        self.carbon_deficit: NDArray = dummy_array
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
        self.carbon_costs: NDArray = dummy_array

        return to_pickle


    def _save_output(self, data_obj: Dict[str, Union[NDArray, str, int]]):
        """Compress and save output data
        data_object: dict; the dict returned from _flush_output
        Can deal with spin file names up to 9999 files"""
        if self.run_counter < 10: # type: ignore
            fpath = "spin{}{}{}{}{}".format(0, 0, 0, self.run_counter, out_ext) # type: ignore
        elif self.run_counter < 100: # type: ignore
            fpath = "spin{}{}{}{}".format(0, 0, self.run_counter, out_ext) # type: ignore
        elif self.run_counter < 1000: # type: ignore
            fpath = "spin{}{}{}".format(0, self.run_counter, out_ext)
        else:
            fpath = "spin{}{}".format(self.run_counter, out_ext) # type: ignore
        with open(self.outputs[fpath], 'wb') as fh: # type: ignore
            dump(data_obj, fh, compress=('lz4', 5), protocol=5) # type: ignore
            fh.flush()
        # gc.collect() # type: ignore
        self.flush_data = None

class grd_mt(state_zero, climate, time, soil, gridcell_output):
    """A gridcell object to run the model in the meta-community mode

    Args:
        base classes with climatic, soil data, and some common methods to manage gridcells
    """


    def __init__(self,
                y: int | float,
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
        # Init state_zero
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
            ValueError: If the start or end date or both are out of bounds

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
                    input_data:Union[Path, str, dict, None]=None,
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
        if input_data is not None:
            if self.config.input_handler.input_method == 'legacy':
                assert isinstance(input_data, (Path, str)), "Input path must be a Path or string for legacy input mode"
                self.input_fpath = Path(os.path.join(input_data, self.input_fname))
                assert self.input_fpath.exists()

                with bz2.BZ2File(self.input_fpath, mode='r') as fh:
                    self.data = pkl.load(fh)
            else:
                assert self.config.input_handler.input_method == 'ih', "Improper input method"
                assert isinstance(input_data, dict), "Input_fpath should be a dict for non-legacy input mode"
                self.input_fpath = None
                self.data = input_data

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
                      climate_data_source:Union[Path, str, Dict[str, NDArray[np.float32]]],
                      stime_i: Dict,
                      co2: Dict,
                      tsoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                      ssoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                      hsoil: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
                      )->None:
        """ PREPARE A GRIDCELL TO RUN in the meta-community mode

        Args:
            climate_data_source (Union[Path, str]): path to the input file or dictionary with climatic and soil data
            stime_i (Dict): dictionary with the time index metadata
            co2 (Dict): dictionary with the CO2 data
            pls_table (np.ndarray): np.array with the functional traits data
            tsoil (Tuple[np.ndarray]):
            ssoil (Tuple[np.ndarray]):
            hsoil (Tuple[np.ndarray]):
        """
        # Input data
        if isinstance(climate_data_source, dict):
            # If input_fpath is a dict, it is expected to contain the data
            # in the same format as the data read from the input file
            self.data = climate_data_source
            self.input_fpath = None
        elif isinstance(climate_data_source, (str, Path)):
            self.input_fpath = str_or_path(climate_data_source)
            self.data = read_bz2_file(self.input_fpath)

        # # Meta-community
        # We want to run queues of gridcells in parallel. So each gridcell receives a copy of the PLS table object

        # Number of communities in the metacommunity. Defined in the config file {caete.toml}
        # Each gridcell has one metacommunity wuth ncomms communities
        self.ncomms:int = self.config.metacomm.n  #type: ignore # Number of communities

        # Metacommunity object
        self.metacomm:mc.metacommunity = mc.metacommunity(self.ncomms, self.get_from_main_array)

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
    # @timer
    def run_gridcell(self,
                  start_date: str,
                  end_date: str,
                  spinup: int = 0,
                  fixed_co2_atm_conc: Optional[str] | Optional[int] | Optional[float] = None,
                  save: bool = True,
                  nutri_cycle: bool = True,
                  afex: bool = False,
                  reset_community: bool = False,
                  kill_and_reset: bool = False,
                  env_filter: bool = False,
                  process_limitation: bool = False,
                  verbose: bool = False):
        """
        Run the model for a grid cell.

        CAETÊ-DVM execution in the start_date - end_date period, can be used for spinup or transient runs.

        Args:
            start_date (str): Start date for model execution in "yyyymmdd" format.
            end_date (str): End date for model execution in "yyyymmdd" format.
            spinup (int, optional): Number of repetitions in spinup. Set to 0 for a transient run between start_date and end_date. Default is 0.
            fixed_co2_atm_conc (Optional[Union[str, int, float]]): Fixed atmospheric CO2 concentration. If None, use dynamic CO2 levels from a predefined file. If a string with a year (e.g., "1987") That year's value in the provided file will be used. Use a float to set a fixed level in ppm. Default is None.
            save (bool, optional): Whether to save the results. Default is True.
            nutri_cycle (bool, optional): Whether to include nutrient cycling in the model. Default is True.
            afex (bool, optional): Whether to apply nutrient addition to soil in the model. Default is False.
            reset_community (bool, optional): Whether to restart a new community if there are not viable PLS. Default is False.
            kill_and_reset (bool, optional): Whether to kill and reset the community structure at the end of execution (only CVEG pools and PLS IDs). Default is False.
            env_filter (bool, optional): Whether to apply environmental filtering (Include new PLS periodically) []. Default is False.
            verbose (bool, optional): Whether to print detailed logs during execution. Default is False.

        Returns:
            None

        Notes:
            - If reset_community is true a new community will be set (reset) when there is no PLSs remaining.
            - If the kill_and_reset is true, after n spins (integer given by spinup parameter - i.e. in the end
              of function execution) all the communities in a gridcell are reset. The reset_community and
              kill_and_reset  arguments are not mutually exclusive. You can use both as true at the same time.
            - The env_filter argument is used to define if new unique PLSs from the main table will be
              seed in the communities that have free slots (PLSs that are not producing). At the moment, the
              interval for the env_filter to add a new PLS to the community is set to  ~15 days.
              If env filter argument is true, then the reset_community argument will have a very low
              probability to trigger a reset because the communities will be constantly filled with new PLS.
              Nonetheless, the reset_community argument will still be able to trigger a reset if the community loses all PLSs.
              With the probability of a reset_community increasing as the interval between new seeds increases. The parameter doy_months
              in the config file (caete.toml) is used to define the interval for the env_filter to add a new PLS to the community.

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

        # Check dates
        assert start < end, "Start date must be before end date"
        assert start >= self.start_date, "initial date out of bounds for the time array"
        assert end <= self.end_date, f"Final date out of bounds for the time array"


        # Define time index bounds for this run
        # During a run we are in general using a slice ov the available time span
        # to run the model. For example, we can run the model for a year or a decade
        # at the begining of the input data time series to spin up. This slice is defined
        # by the start and end dates provided in the arguments. Here we get the indices.
        self.start_index = int(cftime.date2num(start, self.time_unit, self.calendar))
        self.end_index =   int(cftime.date2num(end, self.time_unit, self.calendar))

        # Find the indices in the time array [used to slice the timeseries with driver data  - tas, pr, etc.]
        lower_bound, upper_bound = self.find_index(self.start_index, self.end_index)

        # Define the time steps range (days)
        steps = np.arange(lower_bound, upper_bound + 1, dtype=np.int64)

        # Define the number of repetitions for the spinup
        spin = 1 if spinup == 0 else spinup

        # Define the AFEX mode
        afex_mode = self.afex_config.afex_mode # type: ignore

        # Slice&Catch climatic input and make conversions
        cv = self.config.conversion_factors_isimip # type: ignore


        if self.config.input_handler.input_method == "ih" and self.config.input_handler.input_type == "netcdf":
            # Variables in Netcdf files are already in proprer units
            temp: NDArray[np.float32] = self.tas[lower_bound: upper_bound + 1]   # Air temp: model uses °C
            prec: NDArray[np.float32] = self.pr[lower_bound: upper_bound + 1]    # Precipitation: model uses  mm/day
            p_atm: NDArray[np.float32] = self.ps[lower_bound: upper_bound + 1]   # Atmospheric pressure: model uses hPa
            ipar: NDArray[np.float32] = self.rsds[lower_bound: upper_bound + 1]  # PAR: model uses  mol(photons) m-2 s-1
            ru: NDArray[np.float32] = self.rhs[lower_bound: upper_bound + 1]     # Relative humidity: model uses 0-1
        else:
            temp: NDArray[np.float32] = self.tas[lower_bound: upper_bound + 1] - cv.tas    # Air temp: model uses °C
            prec: NDArray[np.float32] = self.pr[lower_bound: upper_bound + 1] * cv.pr      # Precipitation: model uses  mm/day
            p_atm: NDArray[np.float32] = self.ps[lower_bound: upper_bound + 1] * cv.ps     # Atmospheric pressure: model uses hPa
            ipar: NDArray[np.float32] = self.rsds[lower_bound: upper_bound + 1] * cv.rsds  # PAR: model uses  mol(photons) m-2 s-1
            ru: NDArray[np.float32] = self.rhs[lower_bound: upper_bound + 1] * cv.rhs      # Relative humidity: model uses 0-1

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

        # Define variables to track dates
        first_day_of_simulation = datetime(start.year, start.month, start.day, start.hour, start.minute, start.second)
        # Define the time step
        time_step = timedelta(days=1)

        # Define the object to store the outputs from daily_budget
        daily_output = DailyBudget()

        # Start loops
        # THis outer loop is used to run the model for a number
        # of times defined by the spinup argument. THe model is
        # executed repeatedly between the start and end dates
        # provided in the arguments

        for s in range(spin):

            self._allocate_output(steps.size, self.metacomm.comm_npls, len(self.metacomm), save)

            # Loop over the days
            today = first_day_of_simulation

            # Go back one day
            today -= time_step

            # Arrays to store & pass values for each community in a simulated day
            sto =        np.zeros(shape=(3, self.metacomm.comm_npls), order='F')
            cleaf_in =   np.zeros(self.metacomm.comm_npls, order='F')
            cwood_in =   np.zeros(self.metacomm.comm_npls, order='F')
            croot_in =   np.zeros(self.metacomm.comm_npls, order='F')
            uptk_costs = np.zeros(self.metacomm.comm_npls, order='F')
            rnpp_in =    np.zeros(self.metacomm.comm_npls, order='F')

            # There are two modes of operation: save and not save.
            # In the save == False mode, the arrays are used to store the values that are
            # needed for model iteration, i.e., the values that are used in the next
            # time step. In the save mode, an extra number arrays is created to be used
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
            # This is added to leaf litter pool (that is basicaly a fast SOM pool)
            c_to_nfixers: NDArray[np.float32]= np.zeros(xsize, dtype=np.float32)
            nupt = np.zeros(shape=(2, xsize), dtype=np.float32)
            pupt = np.zeros(shape=(3, xsize), dtype=np.float32)

            if save:
                cc = np.zeros(xsize, dtype=np.float32)
                photo = np.zeros(xsize, dtype=np.float32)
                aresp = np.zeros(xsize, dtype=np.float32)
                npp = np.zeros(xsize, dtype=np.float32)
                lai = np.zeros(xsize, dtype=np.float32)
                rcm = np.zeros(xsize, dtype=np.float32)
                f5 = np.zeros(xsize, dtype=np.float32)
                rm = np.zeros(xsize, dtype=np.float32)
                rg = np.zeros(xsize, dtype=np.float32)
                wue = np.zeros(xsize, dtype=np.float32)
                cue = np.zeros(xsize, dtype=np.float32)
                carbon_deficit = np.zeros(xsize, dtype=np.float32)
                vcmax = np.zeros(xsize, dtype=np.float32)
                specific_la = np.zeros(xsize, dtype=np.float32)
                storage_pool = np.zeros(shape=(3, xsize))

                if process_limitation:
                    lim_status_y_leaf = np.ma.masked_all((xsize, self.metacomm.comm_npls, 366), dtype=np.int8)
                    lim_status_y_stem = np.ma.masked_all((xsize, self.metacomm.comm_npls, 366), dtype=np.int8)
                    lim_status_y_root = np.ma.masked_all((xsize, self.metacomm.comm_npls, 366), dtype=np.int8)
                    uptake_strategy_n = np.ma.masked_all((xsize, self.metacomm.comm_npls, 366), dtype=np.int8)
                    uptake_strategy_p = np.ma.masked_all((xsize, self.metacomm.comm_npls, 366), dtype=np.int8)

            # <- Daily loop
            for step in range(steps.size):
                today += time_step
                julian_day = today.timetuple().tm_yday

                # Get the co2 concentration for the day
                co2 = co2_daily_values[step]
                # Update soil temperature
                self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

                # AFEX
                if afex and julian_day == 365:
                    self.add_soil_nutrients(afex_mode)

                # Loop over communities
                living_pls = 0 # Sum of living PLS in the communities
                for i, community in enumerate(self.metacomm):
                    # if i >= len(self.metacomm):
                    #     break
                    # if community.masked:
                    #     # skip this one
                    #     continue
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
                    daily_output.update(out)

                    # Update the community status
                    community.update_lsid(daily_output.ocpavg)

                    if community.masked and save:
                        continue

                    community.ls = community.vp_lsid.size
                    # # Restore or seed PLS TODO: Error here need to be fixed ()
                    if env_filter and (community.ls < self.metacomm.comm_npls) and not save:
                        # if julian_day in self.doy_months:
                        if julian_day % 2 == 0:  # For testing purposes we add a new PLS every 2 days
                            new_id, new_PLS = community.get_unique_pls(self.get_from_main_array)
                            community.seed_pls(new_id, new_PLS, daily_output.cleafavg_pft,
                                               daily_output.cfrootavg_pft, daily_output.cawoodavg_pft)
                            if verbose: print(f"PLS seed in Community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")

                            daily_output.ocpavg = pft_area_frac64(daily_output.cleafavg_pft,
                                                            daily_output.cfrootavg_pft,
                                                            daily_output.cawoodavg_pft)
                            community.update_lsid(daily_output.ocpavg)
                            community.ls = community.vp_lsid.size
                        # endif


                    community.vp_ocp = daily_output.ocpavg[community.vp_lsid]
                    community.vp_cleaf = daily_output.cleafavg_pft[community.vp_lsid]
                    community.vp_cwood = daily_output.cawoodavg_pft[community.vp_lsid]
                    community.vp_croot = daily_output.cfrootavg_pft[community.vp_lsid]
                    community.vp_sto = daily_output.stodbg[:, community.vp_lsid].astype('float32')
                    community.sp_uptk_costs = daily_output.npp2pay[community.vp_lsid]
                    community.construction_npp = daily_output.rnpp_out[community.vp_lsid]
                    living_pls += community.ls
                    # print(f"Community storage: {community.vp_sto[0, :]}")
                    # print(f"Sto_budget_out {daily_output.stodbg[0, :]}")

                    # Limiting nutrient organization:
                    # dim1 = leaf wood root, code: 1=N 2=P 4=N,COLIM 5=P,COLIM 6=COLIM 0=NOLIM
                    if save:
                        if process_limitation:
                            lim_status_y_leaf[i, :, julian_day - 1] = daily_output.limitation_status[0,:]# type: ignore
                            lim_status_y_stem[i, :, julian_day - 1] = daily_output.limitation_status[1,:]# type: ignore
                            lim_status_y_root[i, :, julian_day - 1] = daily_output.limitation_status[2,:]# type: ignore
                            uptake_strategy_n[i, :, julian_day - 1] = daily_output.uptk_strat[0,:]# type: ignore
                            uptake_strategy_p[i, :, julian_day - 1] = daily_output.uptk_strat[1,:]# type: ignore

                        community.anpp += cw_mean(community.vp_ocp, community.construction_npp.astype(np.float32))
                        community.uptake_costs += cw_mean(community.vp_ocp, community.sp_uptk_costs.astype(np.float32))

                    if save and julian_day == 365:
                        community.cleaf = cw_mean(community.vp_ocp, community.vp_cleaf.astype(np.float32))
                        community.cwood = cw_mean(community.vp_ocp, community.vp_cwood.astype(np.float32))
                        community.croot = cw_mean(community.vp_ocp, community.vp_croot.astype(np.float32))
                        community.csto = cw_mean(community.vp_ocp, community.vp_sto[0, :])
                        community.shannon_diversity = shannon_diversity(community.vp_ocp)
                        community.shannon_entropy = shannon_entropy(community.vp_ocp)
                        community.shannon_evenness = shannon_evenness(community.vp_ocp)

                        # process limitation data
                        # Filter non living PLS from the limitation status
                        if process_limitation:
                            _data_leaf = lim_status_y_leaf[i, [community.vp_lsid], :] # type: ignore
                            _data_stem = lim_status_y_stem[i, [community.vp_lsid], :] # type: ignore
                            _data_root = lim_status_y_root[i, [community.vp_lsid], :] # type: ignore

                            _data_uptake_n = uptake_strategy_n[i, [community.vp_lsid], :]# type: ignore
                            _data_uptake_p = uptake_strategy_p[i, [community.vp_lsid], :]# type: ignore

                            # Loop over the living PLS to get the unique values and counts
                            pls_lim_leaf = []
                            pls_lim_stem = []
                            pls_lim_root = []
                            pls_uptake_n = []
                            pls_uptake_p = []

                            for k in range(community.vp_lsid.size):
                                # Get the unique values and counts for leaf limitation
                                unique, counts = np.unique(_data_leaf[0, k, :], return_counts=True)
                                unique = unique.data[unique.mask == False] # type: ignore
                                pls_lim_leaf.append((unique, counts[:unique.size])) # type: ignore

                                # Stem limitation
                                unique, counts = np.unique(_data_stem[0, k, :], return_counts=True)
                                unique = unique.data[unique.mask == False] # type: ignore
                                pls_lim_stem.append((unique, counts[:unique.size])) # type: ignore

                                # Root limitation
                                unique, counts = np.unique(_data_root[0, k, :], return_counts=True)
                                unique = unique.data[unique.mask == False]
                                pls_lim_root.append((unique, counts[:unique.size])) # type: ignore

                                # Uptake strategy N
                                unique, counts = np.unique(_data_uptake_n[0, k, :], return_counts=True)
                                unique = unique.data[unique.mask == False] # type: ignore
                                pls_uptake_n.append((unique, counts[:unique.size])) # type: ignore

                                # Uptake strategy P
                                unique, counts = np.unique(_data_uptake_p[0, k, :], return_counts=True)
                                unique = unique.data[unique.mask == False]  # type: ignore
                                pls_uptake_p.append((unique, counts[:unique.size])) # type: ignore

                            community.limitation_status_leaf = pls_lim_leaf
                            community.limitation_status_wood = pls_lim_stem
                            community.limitation_status_root = pls_lim_root
                            community.uptake_strategy_n = pls_uptake_n
                            community.uptake_strategy_p = pls_uptake_p

                            # Reset the limitation masked arrays
                            lim_status_y_leaf.mask[i, :, :] = np.ones((self.metacomm.comm_npls, 366), dtype=bool) # type: ignore
                            lim_status_y_stem.mask[i, :, :] = np.ones((self.metacomm.comm_npls, 366), dtype=bool) # type: ignore
                            lim_status_y_root.mask[i, :, :] = np.ones((self.metacomm.comm_npls, 366), dtype=bool) # type: ignore
                            uptake_strategy_n.mask[i, :, :] = np.ones((self.metacomm.comm_npls, 366), dtype=bool) # type: ignore
                            uptake_strategy_p.mask[i, :, :] = np.ones((self.metacomm.comm_npls, 366), dtype=bool) # type: ignore
                        else:
                            pass

                    if community.vp_lsid.size < 1:
                        if verbose: print(f"Empty community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")
                        if reset_community:
                            assert not save, "Cannot save data when resetting communities"
                            if verbose: print(f"Reseting community {i}: Gridcell: {self.lat} °N, {self.lon} °E: In spin:{s}, step:{step}")

                            new_life_strategies = self.get_from_main_array(community.npls)
                            community.restore_from_main_table(new_life_strategies)
                            # continue

                        else:
                            # In the transiant run - i.e., when reset_community is false and
                            # kill_and_reset is false; we mask the community if there is no PLS
                            self.metacomm.mask[i] = np.int8(1)
                            # Set mask to true for this community, will not run in the next steps
                            # Set annual values to zero
                            community.masked = np.int8(1)
                            community.cleaf = np.float32(0.0)
                            community.cwood = np.float32(0.0)
                            community.croot = np.float32(0.0)
                            community.csto  = np.float32(0.0)
                            community.shannon_diversity = -9999.0
                            community.shannon_entropy = -9999.0
                            community.shannon_evenness = -9999.0
                            # if the reset_community is true
                            # continue # cycle

                    # Store values for each community
                    rnpp_mt[i] = cw_mean(community.vp_ocp, community.construction_npp.astype(np.float32)) # Community Weighted rNPP
                    leaf_litter[i] = daily_output.litter_l
                    root_litter[i] = daily_output.litter_fr
                    cwd[i] = daily_output.cwd
                    lnc[:, i] = daily_output.lnc.astype(np.float32)
                    c_to_nfixers[i] = daily_output.cp[3]
                    evavg[i] = daily_output.evavg
                    epavg[i] = daily_output.epavg
                    nupt[:, i] = daily_output.nupt #type: ignore
                    pupt[:, i] = daily_output.pupt #type: ignore

                    if save:
                        cc[i] = daily_output.c_cost_cwm #type: ignore
                        npp[i] = daily_output.nppavg #type: ignore
                        photo[i] = daily_output.phavg #type: ignore
                        aresp[i] = daily_output.aravg #type: ignore
                        lai[i] = daily_output.laiavg #type: ignore
                        rcm[i] = daily_output.rcavg #type: ignore
                        f5[i] = daily_output.f5avg #type: ignore
                        rm[i] = daily_output.rmavg #type: ignore
                        rg[i] = daily_output.rgavg #type: ignore
                        wue[i] = daily_output.wueavg #type: ignore
                        cue[i] = daily_output.cueavg #type: ignore
                        carbon_deficit[i] = daily_output.c_defavg #type: ignore
                        vcmax[i] = daily_output.vcmax #type: ignore
                        specific_la[i] = daily_output.specific_la #type: ignore

                        for j in range(daily_output.stodbg.shape[0]):
                            storage_pool[j, i] = cw_mean(community.vp_ocp, community.vp_sto[j, :]) #type: ignore

                #<- Out of the community loop
                # Save annual state of the metacommunity
                if save:
                    if julian_day == 365:
                        y = today.year
                        # m = today.month
                        # d = today.day
                        filename = self.out_dir/f"metacommunity_{y}.pkz"
                        # filename = self.out_dir/f"metacommunity_{d}{m}{y}.pkz"
                        self.metacomm.save_state(filename, y, process_limitation)
                        self.metacomm_output[y] = filename

                        for community in self.metacomm:
                            # Set annual accumulators to zero
                            community.anpp = np.float32(0.0)
                            community.uptake_costs = np.float32(0.0)

                # ------------
                # Evapotranspiration
                et = masked_mean(self.metacomm.mask, evavg) #evavg.mean()
                ## Canopy-atmosphere coupling [EXPERIMENTAL]
                # vpd = m.vapor_p_deficit(temp[step], ru[step])
                # et_pot = masked_mean(self.metacomm.mask, np.array(epavg).astype(np.float32)) #epavg.mean()
                # self.evapm[step] = atm_canopy_coupling(et_pot, et, temp[step], vpd)
                self.evapm[step] = et

                # Update water pools

                self.runom[step] = self.swp._update_pool(prec[step], self.evapm[step])
                self.swp.w1 = 0.0 if self.swp.w1 < 0.0 else self.swp.w1
                self.swp.w2 = 0.0 if self.swp.w2 < 0.0 else self.swp.w2
                self.wp_water_upper_mm = self.swp.awc1
                self.wp_water_lower_mm = self.swp.awc2
                wtot = self.swp.w1 + self.swp.w2

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
                self.sp_snc = np.zeros(shape=8)
                self.sp_snc = soil_out['snc']
                idx = np.where(self.sp_snc < 0.0)[0]
                if len(idx) > 0:
                    self.sp_snc[idx] = 0.0

                # <- Out of the community loop
                self.nupt[:, step] = masked_mean_2D(self.metacomm.mask, nupt)
                self.pupt[:, step] = masked_mean_2D(self.metacomm.mask, pupt)
                
                # TODO: Soil nutrient dynamics. Isolate this if branch into a separate method/class/function
                # Critical part of soil nutrient dynamics and availability for plants
                # IF NUTRICYCLE
                if nutri_cycle:
                    # UPDATE ORGANIC POOLS
                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()
                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()

                    # Update inorganic pools
                    self.sp_available_p += soil_out['pmin']
                    self.sp_available_n += soil_out['nmin']
                    self.sp_available_p -= self.pupt[0, step]
                    self.sp_available_n -= self.nupt[0, step]

                    # NUTRIENT DINAMICS
                    
                    # Inorganic N
                    # TODO: NaNs are being sourced upstream , need to track the source and fix it

                    if not np.isfinite(self.sp_in_n):
                        # rwarn(f"Non-finite value detected in sp_in_n pool at step {step}. Resetting to zero.")
                        self.sp_in_n = 0.0
                    if not np.isfinite(self.sp_available_n):
                        # rwarn(f"Non-finite value detected in sp_available_n pool at step {step}. Resetting to zero.")
                        self.sp_available_n = 0.0
                    if not np.isfinite(self.sp_so_n):
                        # rwarn(f"Non-finite value detected in sp_so_n pool at step {step}. Resetting to zero.")
                        self.sp_so_n = 0.0
                    
                    self.sp_in_n += self.sp_available_n + self.sp_so_n
                    self.sp_so_n = soil_dec.sorbed_n_equil(self.sp_in_n)
                    self.sp_available_n = soil_dec.solution_n_equil(self.sp_in_n)
                    self.sp_in_n -= (self.sp_so_n + self.sp_available_n)

                    # Inorganic P
                    if not np.isfinite(self.sp_in_p):
                        # rwarn(f"Non-finite value detected in sp_in_p pool at step {step}. Resetting to zero.")
                        self.sp_in_p = 0.0
                    if not np.isfinite(self.sp_available_p):
                        # rwarn(f"Non-finite value detected in sp_available_p pool at step {step}. Resetting to zero.")
                        self.sp_available_p = 0.0
                    if not np.isfinite(self.sp_so_p):
                        # rwarn(f"Non-finite value detected in sp_so_p pool at step {step}. Resetting to zero.")
                        self.sp_so_p = 0.0
                    
                    self.sp_in_p += self.sp_available_p + self.sp_so_p
                    # sp_so_p is the occluded P in the inorganic pool
                    self.sp_so_p = soil_dec.sorbed_p_equil(self.sp_in_p)
                    # THe fraction that can be dissolved in soil solution (passive uptake uses transpiration 
                    # to estimate the amount of P that can be taken up from the soil solution pool)
                    self.sp_available_p = soil_dec.solution_p_equil(self.sp_in_p)
                    # Inorganic pool that is adsorbed
                    self.sp_in_p -= (self.sp_so_p + self.sp_available_p)

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
                    total_on = self.sp_snc[:4].sum()
                    if total_on > 0.0 and np.isfinite(total_on):
                        frsn = [i / total_on for i in self.sp_snc[:4]]
                    else:
                        frsn = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsn):
                        self.sp_snc[i] -= (self.nupt[1, step] * fr)

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        self.sp_snc[idx] = 0.0

                    self.sp_organic_n = self.sp_snc[:2].sum()
                    self.sp_sorganic_n = self.sp_snc[2:4].sum()

                    # P
                    total_op = self.sp_snc[4:].sum()
                    if total_op > 0.0 and np.isfinite(total_op):
                        frsp = [i / total_op for i in self.sp_snc[4:]]
                    else:
                        frsp = [0.0, 0.0, 0.0, 0.0]
                    for i, fr in enumerate(frsp):
                        self.sp_snc[i + 4] -= (self.pupt[2, step] * fr)

                    idx = np.where(self.sp_snc < 0.0)[0]
                    if len(idx) > 0:
                        self.sp_snc[idx] = 0.0

                    self.sp_organic_p = self.sp_snc[4:6].sum()
                    self.sp_sorganic_p = self.sp_snc[6:].sum()
                # END SOIL NUTRIENT DYNAMICS

                if save:
                    # Plant uptake and Carbon costs of nutrient uptake
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
                    self.carbon_deficit[step] = masked_mean(self.metacomm.mask, carbon_deficit)
                    self.vcmax[step] = masked_mean(self.metacomm.mask, vcmax)
                    self.specific_la[step] = masked_mean(self.metacomm.mask, specific_la)
                    self.hresp[step] = soil_out['hr']
                    self.csoil[:, step] = soil_out['cs']
                    self.wsoil[step] = self.swp.calc_total_water()
                    self.inorg_n[step] = self.sp_in_n
                    self.inorg_p[step] = self.sp_in_p
                    self.sorbed_n[step] = self.sp_so_n
                    self.sorbed_p[step] = self.sp_so_p
                    self.snc[:, step] = soil_out['snc']
                    self.nmin[step] = self.sp_available_n
                    self.pmin[step] = self.sp_available_p
                    self.ls[step] = living_pls

            # <- Out of the daily loop
            # Save the spin data
            sv: Thread
            if save:
                if s > 0:
                    sv.join()  # Wait for the previous thread to finish
                    self.flush_data = None
                self.executed_iterations.append((start_date, end_date))
                self.flush_data = self._flush_output(
                    'spin', (self.start_index, self.end_index))
                sv = Thread(target=self._save_output, args=(self.flush_data,))
                sv.start()
        # <- Out of spin loop
        # Manage the last thread
        if save:
            sv.join()  # Wait for the last thread to finish
            self.flush_data = None


        # Restablish new communities in the end, if applicable
        if kill_and_reset:
            assert not save, "Cannot save data when resetting communities"
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


    def _fetch_metacommunity_data(self, year) -> dict:
        """Get the data from a metacommunity output file"""
        filename = self.metacomm_output.get(year)
        if filename is None:
            raise KeyError(f"No data available for year {year}")
        with open(filename, 'rb') as fh:
            metacomm_dt = load(fh)
        return metacomm_dt


    def _get_years(self) -> List[int]:
        """Get the sequence of years for which data is available"""
        return sorted(list(self.metacomm_output.keys()))


    def _read_annual_output(self, variables: str) -> Dict[int, NDArray[np.float32]]:
        """Read the annual output for the gridcell (only metacommunity aggregated data)"""

        # Variables that are aggregated over the metacommunities
        aggregated_variables: Set[str] = {"cveg",
                                          "cleaf",
                                          "croot",
                                          "cwood",
                                          "anpp",
                                          "uptake_costs"}

        vnames = get_args(variables)

        for v in vnames:
            assert v in aggregated_variables, f"Variable {v} not available"

        years = self._get_years()
        fetched_data = []
        for y in years:
            fetched_data.append(self._fetch_metacommunity_data(y))
        output = {}

        for variable in vnames:
            outarr = np.zeros(len(years), dtype=np.float32)
            for i, year in enumerate(years):
                outarr[i] = fetched_data[i][variable]
            output[f"{variable}_{self.xyname}"] = outarr
        return output


    def _read_annual_metacomm_biomass(self, year: int) -> Dict[str, NDArray[np.float32] | NDArray[np.int32]]:

        years = self._get_years()
        assert year in years, f"Year {year} not available"
        fetched_data = self._fetch_metacommunity_data(year)
        communities = fetched_data["communities"]

        # Collect arrays in lists
        id_list = []
        cleaf_list = []
        croot_list = []
        cwood_list = []

        for community in communities.values():
            if not community["masked"]:
                id_list.append(community["id"])
                cleaf_list.append(community["vp_cleaf"])
                croot_list.append(community["vp_croot"])
                cwood_list.append(community["vp_cwood"])

        # Concatenate once
        pls_id = np.concatenate(id_list) if id_list else np.zeros(0, dtype=np.int32)
        pls_cleaf = np.concatenate(cleaf_list) if cleaf_list else np.zeros(0, dtype=np.float32)
        pls_croot = np.concatenate(croot_list) if croot_list else np.zeros(0, dtype=np.float32)
        pls_cwood = np.concatenate(cwood_list) if cwood_list else np.zeros(0, dtype=np.float32)

        return {
            "pls_id": pls_id,
            "vp_cleaf": pls_cleaf,
            "vp_croot": pls_croot,
            "vp_cwood": pls_cwood
        }


    def _read_daily_output(self,
                           period: Union[int, Tuple[int, int], None] = None,
                           ) -> Union[Tuple, List[Any], Dict]:

        """Read the daily output for this gridcell.

        Warning: This method assumes that the ouptut files are time-ordered
        The argument spinup, if true, will cause the function to return the data
        for for one specified period. The period argument must be provided in this case.
        """
        assert len(self.outputs) > 0, "No output data available. Run the model first"

        if isinstance(period, int):
            assert period > 0, "Period must be positive"
            assert period <= self.run_counter, "Period must be less than the number of spins"
            with ThreadPoolExecutor(max_workers=1) as executor:
                return [executor.submit(self.__fetch_spin_data, period)]

        elif isinstance(period, tuple):
            assert period[1] <= self.run_counter, "Period must be less than the number of spins" # type: ignore
            assert period[0] < period[1], "Period must be a tuple with the start and end spins" # type: ignore
            spins = range(period[0], period[1] + 1) # type: ignore
            files = tuple((f"spin{x}" for x in spins))
        
        elif period is None:
            files:Tuple[str, ...] = tuple(path_str for path_str in self.outputs.values())
            spins = range(1, len(files) + 1)
        
        else:
            raise ValueError("Invalid period argument, period must be an integer, tuple of integers, or None")

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
        """ Get the daily data for the gridcell.

        Args:
            variable (Union[str, Collection[str]], optional): variable name or names. Defaults to "npp".
            spin_slice (Optional[Tuple[int, int]], optional): Slice of spins. Defaults to None.
            pp (bool, optional): Print available variable names in the output data and exits. Defaults to False.
            return_time (bool, optional): Return a collection of time objects with the days of simulation. Defaults to False.
            return_array (bool, optional): Returns one array or a tuple of arrays. Defaults to False.


        Returns:
            Union[List, NDArray, Tuple[NDArray, NDArray], Tuple[Dict[str, NDArray], NDArray], List[NDArray]
        TODO: Add all return types

        """

        if isinstance(variable, str):
            variable = [variable,]
        assert isinstance(variable, Collection), "Variable must be a string or a collection of strings"


        result = []
        f = self._read_daily_output(period=None) if spin_slice is None else self._read_daily_output(period=spin_slice) # type: ignore
        for _read_ in f:
            result.append(_read_.result())

        # GEt start and end dates (by index)
        sind = result[0]["sind"]
        if len(result) == 1:
            eind = result[0]["eind"]
        elif len(result) > 1:
            eind = result[-1]["eind"]

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


    def print_available_periods(self, verbose: bool = False) -> int:
        assert len(self.executed_iterations) > 0, "No output data available. Run the model first"

        for i, period in enumerate(self.executed_iterations):
            if verbose:
                print(f"Period {i + 1}: {period[0]} - {period[1]}")
        return i + 1


if __name__ == '__main__':
    # Short example of how to run the new version of the model. Also used to do some profiling
    skip = False
    try:
        skip = sys.argv[1] == "pass"
    except:
        skip = False

    if skip:
        # Skip all
        pass
    else:
        from metacommunity import pls_table
        from parameters import *
        from region import region
        import polars as pl

        ## Working with gridcells in memory. In the parallel runs of the region, the gridcells are stored in files.
        co2_path = Path("../input/co2/historical_CO2_annual_1765-2024.csv")
        co2_path_ssp370 = Path("../input/co2/ssp370_CO2_annual_2015-2100.csv")
        main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-9999.csv"))
        gridlist = pl.read_csv("../grd/gridlist_test.csv")


        r = region("region_test",
                    "../input/MPI-ESM1-2-HR/historical/caete_input_MPI-ESM1-2-HR_historical.nc",
                    (tsoil, ssoil, hsoil),
                    co2_path,
                    main_table,
                    gridlist)

        # Set gridcells in memory
        r.set_gridcells()

        gridcell = r[0]

        # Profile with cProfile
        try:
            prof = sys.argv[1] == "cprof"
        except:
            prof = False
        if prof:
            import cProfile
            command = "gridcell.run_gridcell('1901-01-01', '1950-12-31', spinup=2, fixed_co2_atm_conc=1901, save=False, nutri_cycle=True, reset_community=True)"
            cProfile.run(command, sort="cumulative", filename="profile.prof")
        else:
        # Run the model for one gridcell to test the functionality
            start = tm.time()
            # test model functionality
            gridcell.run_gridcell("1901-01-01", "1950-12-31", spinup=1, fixed_co2_atm_conc=None,
                                                save=False, nutri_cycle=True, reset_community=True,
                                                env_filter=True)

            gridcell.run_gridcell("1901-01-01", "1950-12-31", spinup=1, fixed_co2_atm_conc=None,
                                    save=True, nutri_cycle=True)

            # test directory update
            r.update_dump_directory("test_new_region")

            # test change input
            r.update_input("../input/MPI-ESM1-2-HR/ssp370/caete_input_MPI-ESM1-2-HR_ssp370.nc", co2 = co2_path_ssp370)

            gridcell = r[0]
            n =gridcell.run_gridcell("2015-01-01", "2030-12-31", spinup=1, fixed_co2_atm_conc=None,
                                                save=True, nutri_cycle=True)
            print("collcted objects:", n)
            end = tm.time()
            print(f"Run time: {end - start:.2f} seconds")

