# -*-coding:utf-8-*-
# "CAETÊ"

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

# This script contains functions to read binary output
# and create gridded and table outputs.
# Author: Joao Paulo Darela Filho
import argparse
import cProfile
import pstats
import time
import tracemalloc
from pstats import SortKey
from functools import wraps

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Collection, Tuple, Dict, List

import numpy as np
import polars as pl
from numpy.typing import NDArray
from joblib import Parallel, delayed
from numba import jit

from caete_jit import pft_area_frac64
from config import fetch_config
from _geos import pan_amazon_region, get_region

if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import grd_mt, get_args, str_or_path
from worker import worker
from region import region


#TODO: implement region configuration
if pan_amazon_region is None:
    raise ValueError("pan_amazon_region is not defined or imported correctly")

# Get the region of interest
ymin, ymax, xmin, xmax = get_region(pan_amazon_region)

config = fetch_config()

def get_spins(r:region, gridcell=0):
    """Prints the available spin slices for a gridcell in the region"""
    return r[gridcell].print_available_periods()

def print_variables(r:region):
    """Prints the available variables for gridcells in the region. Assumes all gridcells have the same variables"""
    r[0]._get_daily_data("DUMMY", 1, pp=True)

def get_var_metadata(var):
    """
    Retrieve metadata for a given variable or list of variables.

    Args:
        var (str or list): Variable name or list of variable names to retrieve metadata for.

    Returns:
        dict: A dictionary where keys are variable names and values are lists containing metadata
              (e.g., long name, units, and standard name). If a variable is not found, its metadata
              defaults to ['unknown', 'unknown', 'unknown'].
    """
    vunits = {'header': ['long_name', 'units', 'standart_name'],
            'rsds': ['Short_wav_rad_down', 'W m-2', 'rsds'],
            'wind': ['Wind_velocity', 'm s-1', 'wind'],
            'ps': ['Sur_pressure', 'Pa', 'ps'],
            'tas': ['Sur_temperature_2m', 'celcius', 'tas'],
            'tsoil': ['Soil_temperature', 'celcius', 'soil_temp'],
            'pr': ['Precipitation', 'Kg m-2 month-1', 'pr'],
            'litter_l': ['Litter C flux - leaf', 'g m-2 day-1', 'll'],
            'cwd': ['Litter C flux - wood', 'g m-2 day-1', 'cwd'],
            'litter_fr': ['Litter C flux fine root', 'g m-2 day-1', 'lr'],
            'litter_n': ['Litter Nitrogen Flux', 'g m-2 day-1', 'ln'],
            'litter_p': ['Litter phosphorus flux', 'g m-2 day-1', 'lp'],
            'sto_c': ['PLant Reserve Carbon', 'g m-2', 'sto_c'],
            'sto_n': ['Pant Reserve Nitrogen', 'g m-2', 'sto_n'],
            'sto_p': ['Plant Reserve Phosphorus', 'g m-2', 'sto_p'],
            'c_cost': ['Carbon costs of Nutrients Uptake', 'g m-2 day-1', 'cc'],
            'wsoil': ['Soil_water_content-wsoil', 'kg m-2', 'mrso'],
            'evapm': ['Evapotranspiration', 'kg m-2 day-1', 'et'],
            'emaxm': ['Potent. evapotrasnpiration', 'kg m-2 day-1', 'etpot'],
            'runom': ['Total_runoff', 'kg m-2 day-1', 'mrro'],
            'aresp': ['Autothrophic respiration', 'kg m-2 year-1', 'ar'],
            'photo': ['Gross primary productivity', 'kg m-2 year-1', 'gpp'],
            'npp': ['Net primary productivity = GPP - AR', 'kg m-2 year-1', 'npp'],
            'rnpp': ['Net primary productivity, C allocation', 'g m-2 day-1', 'npp'],
            'lai': ['Leaf Area Index - LAI', 'm2 m-2', 'lai'],
            'rcm': ['Stomatal resistence', 's m-1', 'rcm'],
            'hresp': ['Soil heterotrophic respiration', 'g m-2 day-1', 'hr'],
            'nupt': ['Nitrogen uptake', 'g m-2 day-1', 'nupt'],
            'pupt': ['Phosphorus uptake', 'g m-2 day-1', 'pupt'],
            'csoil': ['Soil Organic Carbon', 'g m-2', 'csoil'],
            'org_n': ['Soil Organic Nitrogen', 'g m-2', 'org_n'],
            'org_p': ['Soil Organic Phosphorus', 'g m-2', 'org_p'],
            'inorg_n': ['Soil Inorganic Nitrogen', 'g m-2', 'inorg_n'],
            'inorg_p': ['Soil Inorganic Phosphorus', 'g m-2', 'inorg_p'],
            'sorbed_p': ['Soil Sorbed Phosphorus', 'g m-2', 'sorbed_p'],
            'nmin': ['Soil Inorganic Nitrogen (solution)', 'g m-2', 'nmin'],
            'pmin': ['Soil Inorganic Phosphorus (solution)', 'g m-2', 'pmin'],
            'rm': ['Maintenance respiration', 'kg m-2 year-1', 'rm'],
            'rg': ['Growth respiration', 'kg m-2 year-1', 'rg'],
            'wue': ['Water use efficiency', '1', 'wue'],
            'vcmax': ['Maximum RuBisCo activity', 'mol m-2 s-1', 'vcmax'],
            'specific_la': ['Specfic leaf area', 'm2 g-1', 'sla'],
            'cue': ['Carbon use efficiency', '1', 'cue'],
            'cawood': ['C in woody tissues', 'kg m-2', 'cawood'],
            'cfroot': ['C in fine roots', 'kg m-2', 'cfroot'],
            'cleaf': ['C in leaves', 'kg m-2', 'cleaf'],
            'cmass': ['Total Carbon -Biomass', 'kg m-2', 'cmass'],
            'g1': ['G1 param - Stomatal Resistence model', 'hPA', 'g1'],
            'resopfrac': ['Leaf resorpton fraction N & P', '%', 'resopfrac'],
            'tleaf': ['Leaf C residence time', 'years', 'tleaf'],
            'twood': ['Wood C residence time', 'years', 'twood'],
            'troot': ['Fine root C residence time', 'years', 'troot'],
            'aleaf': ['Allocation coefficients for leaf', '1', 'aleaf'],
            'awood': ['Allocation coefficients for wood', '1', 'awood'],
            'aroot': ['Allocation coefficients for root', '1', 'aroot'],
            'c4': ['C4 photosynthesis pathway', '1', 'c4'],
            'leaf_n2c': ['Leaf N:C', 'g g-1', 'leaf_n2c'],
            'awood_n2c': ['Wood tissues N:C', 'g g-1', 'awood_n2c'],
            'froot_n2c': ['Fine root N:C', 'g g-1', 'froot_n2c'],
            'leaf_p2c': ['Leaf P:C', 'g g-1', 'leaf_p2c'],
            'awood_p2c': ['Wood tissues P:C', 'g g-1', 'awood_p2c'],
            'froot_p2c': ['Fine root P:C', 'g g-1', 'froot_p2c'],
            'amp': ['Percentage of fine root colonized by AM', '%', 'amp'],
            'pdia': ['NPP alocated to N fixers', 'fraction_of_npp', 'pdia'],
            'ls': ['Living Plant Life Strategies', '1', 'ls']
        }
    out = {}
    for v in var:
        out[v] = vunits.get(v, ['unknown', 'unknown', 'unknown'])
    return out

def write_metadata_to_csv(variable_names:Tuple[str,...], output_path:Path):
    metadata = get_var_metadata(("header", ) + variable_names)
    header = metadata.pop("header")

    # Convert nested dict to pl.DataFrame
    data = []
    for var_name, values in metadata.items():
        data.append([var_name] + values)

    # Use orient="row" to avoid the warning
    df = pl.DataFrame(data, schema=["variable_name"] + header, orient="row")
    df.write_csv(output_path / "output_metadata.csv")
    return df


#=========================================
# Functions dealing with gridded outputs
#=========================================
class gridded_data:
    """This class contains methods to read and process gridded data from the model outputs.
    """
     # Daily data --------------------------------
    @staticmethod
    def read_grd(grd:grd_mt,
                 variables: Union[str, Collection[str]],
                 spin_slice: Union[int, Tuple[int, int], None]
                 ) -> Tuple[NDArray, Union[Dict, NDArray, List, Tuple], Union[int, float], Union[int, float]]:
        """ Reads the data from a gridcell and returns a tuple with the following keys:
        time, coord, data holding data to be transformed.

        Args:
            grd (grd_mt): A gridcell object
            variables (Collection[str]): which variables to read from the gridcell
            spin_slice (Union[int, Tuple[int, int], None]): which spin slice to read

        Returns:
            Tuple[NDArray, Union[Dict, NDArray, List, Tuple], Union[int, float], Union[int, float]]:
              a tuple with the following keys: time, coord, data holding data to be transformed

        """
        # returns a tuple with data and time Tuple[NDArray, NDArray]
        data = grd._get_daily_data(get_args(variables), spin_slice, return_time=True) # type: ignore
        time = data[-1]
        data = data[0]
        return time, data, grd.y, grd.x


    @staticmethod
    def aggregate_region_data(r: region,
                variables: Union[str, Collection[str]],
                spin_slice: Union[int, Tuple[int, int], None] = None
                )-> Dict[str, NDArray]:
        """_summary_

        Args:
            r (region): a region object

            variables (Union[str, Collection[str]]): variable names to read

            spin_slice (Union[int, Tuple[int, int], None], optional): which spin slice to read.
            Defaults to None, read all available data. Consumes a lot of memory.

        Returns:
            dict: a dict with the following keys: time, coord, data holding data to be transformed
            necessary to create masked arrays and subsequent netCDF files.
        """

        output = []
        # nproc = min(len(r), 56)
        nproc = config.multiprocessing.nprocs # type_ignore
        nproc = max(1, nproc) # Ensure at least one thread is used
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(gridded_data.read_grd, grd, variables, spin_slice) for grd in r]
            for future in futures:
                output.append(future.result())

        # Finalize the data object
        raw_data = np.array(output, dtype=object)
        # Reoeganize resources
        time = raw_data[:,0][0] # We assume all slices have the same time, thus we get the first one
        coord = raw_data[:,2:4][:].astype(np.int64) # 2D matrix of coordinates (y(lat), x(lon))}
        data = raw_data[:,1][:]  # array of dicts, each dict has the variables as keys and the time series as values

        if isinstance(variables, str):
            dim_names = ["time", "coord", variables]
        else:
            dim_names = ["time", "coord", "data"]

        return dict(zip(dim_names, (time, coord, data)))


    @staticmethod
    def create_masked_arrays(data: dict):
        """ Reads a dict generated by aggregate_region_data and reorganize the data
        as masked_arrays with shape=(time, lat, lon) for each variable

        Args:
            data (dict): a dict generated by aggregate_region_data

        Returns:
            _type_: a tuple with a list of masked_arrays (for each variable),
            the time array, and the array names.
        """
        time = data["time"]
        coords = data["coord"]

        assert "data" in data.keys(), "The input dict must contain the 'data' keyword"
        assert isinstance(data["data"][0], dict), "Data must be a dict"
        variables = list(data["data"][0].keys())  # holds variable names being processed

        arrays_dict = data["data"][:]

        # Read dtypes
        dtypes = []
        for var in variables:
            dtypes.append(arrays_dict[0][var].dtype)

        # Calculate region dimensions first
        region_height = ymax - ymin
        region_width = xmax - xmin

        # Allocate the arrays - only for the region of interest (not the whole globe)
        arrays = []
        array_names = []
        nx, ny = -1, -1
        for i, var in enumerate(variables):
            dim = arrays_dict[0][var].shape
            if len(dim) == 1:
                arrays.append(np.ma.masked_all(shape=(dim[0], region_height, region_width), dtype=dtypes[i]))
                array_names.append(var)
            elif len(dim) == 2:
                ny, nx = dim
                for k in range(ny):
                    arrays.append(np.ma.masked_all(shape=(nx, region_height, region_width), dtype=dtypes[i]))
                    array_names.append(f"{var}_{k + 1}")

        # Fill the arrays - adjust coordinates to be relative to the region
        array_index = 0
        for i, var in enumerate(variables):
            for j in range(len(coords)):
                # Calculate coordinates relative to the region bounding box
                y_rel = coords[j][0] - ymin
                x_rel = coords[j][1] - xmin

                # Only process points that fall within our region
                if 0 <= y_rel < region_height and 0 <= x_rel < region_width:
                    if len(arrays_dict[j][var].shape) == 1:
                        arrays[array_index][:, y_rel, x_rel] = arrays_dict[j][var]
                    elif len(arrays_dict[j][var].shape) == 2:
                        ny, nx = arrays_dict[j][var].shape
                        for k in range(ny):
                            arrays[array_index + k][:, y_rel, x_rel] = arrays_dict[j][var][k, :]
            array_index += ny if len(arrays_dict[j][var].shape) == 2 else 1 # type ignore

        return arrays, time, array_names


    @staticmethod
    def save_netcdf(data: dict, output_path: Path, file_name: str):
        pass


# ======================================
# Functions dealing with table outputs
# ======================================

def process_arrays(keys, arrays, shapes):
    """Process multiple arrays efficiently.

    Args:
        keys: List of array key names
        arrays: List of numpy arrays (both 1D and 2D)
        shapes: List of shape information tuples for each array

    Returns:
        Dictionary with processed arrays
    """
    # Create a standard dictionary for results
    result_dict = {}

    for i in range(len(keys)):
        # Skip if index is out of bounds
        if i >= len(arrays) or i >= len(shapes):
            continue

        key = keys[i]
        array = arrays[i]
        shape = shapes[i]

        # Check if shape is valid
        if not isinstance(shape, tuple):
            continue

        # Check dimensionality based on shape information
        if len(shape) == 1:
            # Simply add 1D array to result
            result_dict[key] = array
        elif len(shape) == 2:
            ny, nx = shape
            # Use numpy's optimized sum along axis
            _sum = np.sum(array, axis=0)

            # Add sum to result
            result_dict[key + "_sum"] = _sum

            # Add individual rows
            for j in range(ny):
                result_dict[key + "_" + str(j+1)] = array[j, :].copy()

    return result_dict

# Standalone numba function for optimized calculations
@jit(nopython=True, cache=True)
def calculate_cveg_and_ocp(cleaf, croot, cwood):
    """Numba-optimized function to calculate total vegetation carbon and area fraction.

    Args:
        cleaf: Numpy array of leaf carbon values
        croot: Numpy array of root carbon values
        cwood: Numpy array of wood carbon values

    Returns:
        Tuple of (cveg, ocp) arrays
    """
    # Calculate total vegetation carbon
    cveg = cleaf + croot + cwood

    # Calculate area fraction using the existing function
    # Note: pft_area_frac64 must be compatible with numba
    ocp = pft_area_frac64(cleaf, croot, cwood)

    return cveg, ocp


class table_data:
    """
    Extracts and processes table data from the model outputs.
    This class contains methods to create daily dataframes for each grid cell in a region
    and save them as CSV files. It also includes methods to write yearly metacommunity biomass output
    to CSV files.

    Note:
    Do not use this class directly. It is designed to be used with the output_manager class.
    It is optimized for performance and memory usage, but may not handle very large datasets efficiently.

    May not be suitable for very large datasets due to memory constraints.
    Use with caution for large regions or many variables.
    """


    @staticmethod
    def make_daily_dataframe(r:region,
                variables: Union[str, Collection[str]],
                spin_slice: Union[int, Tuple[int, int], None] = None):
        """
        Create daily dataframes for each grid cell in a region and save them as CSV files.

        Args:
            r (region): The region object containing grid cells.
            variables (Union[str, Collection[str]]): Variable names to retrieve data for.
            spin_slice (Union[int, Tuple[int, int], None], optional): The spin slice to read. Defaults to None.

        Returns:
            None
        """
        for grd in r:
            d = grd._get_daily_data(get_args(variables), spin_slice, return_time=True) #type: ignore

            time = [t.strftime("%Y-%m-%d") for t in d[1]] # type: ignore
            data = d[0] # type: ignore

            # Prepare data for numba processing
            keys = []
            arrays = []
            shapes = []

            # Collect data for numba processing
            for k, v in data.items(): # type: ignore
                keys.append(k)
                # Make a copy to ensure array is writable for numba
                v_copy = v.copy()
                arrays.append(v_copy)
                # Ensure shape is always a valid tuple for numba
                if not hasattr(v_copy, 'shape'):
                    continue
                if isinstance(v_copy.shape, tuple):
                    shapes.append(v_copy.shape)
                elif hasattr(v_copy, 'shape') and isinstance(v_copy.shape, int):
                    # Handle case where shape might be an integer (1D array)
                    shapes.append((v_copy.shape,))
                else:
                    # Skip invalid shapes
                    continue

            # Process arrays using our efficient processing function
            new_data = process_arrays(keys, arrays, shapes)

            # Add time information
            new_data['day'] = time

            fname = f"grd_{grd.y}_{grd.x}_{time[0]}_{time[-1]}.csv"

            # Create DataFrame with polars and write to CSV efficiently
            df = pl.DataFrame(new_data)
            df.write_csv(grd.out_dir / fname)


    @staticmethod
    def write_daily_data(r: region, variables: Union[str, Collection[str]]):
        """Writes the daily data to csv files"""
        periods = r[0].print_available_periods() # assumes all gridcells have the same periods
        write_metadata_to_csv(variables, r.output_path)  # type: ignore

        # Determine optimal number of workers based on CPU count and data size
        max_workers = min(os.cpu_count() or 4, periods)

        def worker(i):
            table_data.make_daily_dataframe(r, variables=variables, spin_slice=i+1)

        # Use ThreadPoolExecutor with optimized worker count
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(worker, range(periods))


    @staticmethod
    def write_metacomm_output(grd:grd_mt) -> None:
        """Writes the metacommunity biomass output (C in vegetation and abundance) to a csv file

        Args:
            grd (grd_mt): gridcell object
        """
        all_df_list = []
        years = grd._get_years()

        # Process each year in parallel when there are multiple years
        if len(years) > 1:
            with ThreadPoolExecutor(max_workers=min(len(years), os.cpu_count() or 4)) as executor:
                futures = []

                for year in years:
                    futures.append(executor.submit(table_data._process_year_data, grd, year))

                # Collect results as they complete
                for future in futures:
                    result_df = future.result()
                    if result_df is not None:
                        all_df_list.append(result_df)
        else:
            # Process single year directly
            for year in years:
                result_df = table_data._process_year_data(grd, year)
                if result_df is not None:
                    all_df_list.append(result_df)

        # Concatenate all dataframes and write to CSV
        if all_df_list:
            pl.concat(all_df_list).write_csv(
                grd.out_dir / f"metacomunity_biomass_{grd.xyname}.csv"
            )


    @staticmethod
    def _process_year_data(grd:grd_mt, year):
        """Process data for a single year - helper function for write_metacomm_output.

        Args:
            grd: The gridcell object
            year: The year to process

        Returns:
            A polars DataFrame with the processed data
        """
        # Get data and convert to DataFrame once
        data_dict = grd._read_annual_metacomm_biomass(year)

        # Skip empty data
        if not data_dict:
            return None

        # Create DataFrame with polars
        data = pl.DataFrame(data_dict)

        # Count occurrences of each pls_id
        count_df = (data.group_by("pls_id")
                    .agg(pl.len().alias("count"))
                    .sort("pls_id"))

        # Group by pls_id and calculate mean
        df = (data.group_by("pls_id")
              .mean()
              .join(count_df, on="pls_id"))

        # Select needed columns
        df = df.select(["pls_id", "vp_cleaf", "vp_croot", "vp_cwood", "count"])

        # Convert to numpy arrays for the pft_area_frac64 calculation
        # Make copies of arrays to ensure they're writeable (not readonly)
        pls_id_values = df.get_column("pls_id").to_numpy().copy()
        cleaf = df.get_column("vp_cleaf").to_numpy().copy()
        croot = df.get_column("vp_croot").to_numpy().copy()
        cwood = df.get_column("vp_cwood").to_numpy().copy()

        # Use the numba optimized function to calculate cveg and ocp
        try:
            cveg, ocp = calculate_cveg_and_ocp(cleaf, croot, cwood)
        except Exception:
            # Fallback to non-numba calculation if there's an error
            ocp = pft_area_frac64(cleaf, croot, cwood)
            cveg = cleaf + croot + cwood

        # Create new DataFrame with all columns
        result_df = pl.DataFrame({
            "pls_id": pls_id_values,
            "vp_cleaf": cleaf,
            "vp_croot": croot,
            "vp_cwood": cwood,
            "count": df.get_column("count").to_numpy(),
            "cveg": cveg,
            "ocp": ocp,
            "year": [year] * len(pls_id_values)
        })

        return result_df

#=======================================
# Interface for output management
#=======================================

class output_manager:

    @staticmethod
    def table_output_per_grd(result:Union[Path, str], variables:Union[str, Collection[str]]):
        """
        Process table outputs for a grid and save them as CSV files.

        Args:
            result (Union[Path, str]): Path to the serialized state file containing the region data.
            variables (Union[str, Collection[str]]): Variable names to process and save.

        Returns:
            None
        """
        # Load state data
        reg:region = worker.load_state_zstd(str_or_path(result))

        # Process daily data first
        table_data.write_daily_data(r=reg, variables=variables)

        # Determine optimal number of processes for parallelism
        available_cpus = os.cpu_count() or 4
        nprocs = min(len(reg), available_cpus)

        # Define grid cell processing function
        def process_gridcell(grd):
            table_data.write_metacomm_output(grd)

        # Use joblib's Parallel for efficient multiprocessing
        # Set verbose=1 to show progress during longer operations
        Parallel(n_jobs=nprocs, verbose=1)(delayed(process_gridcell)(grd) for grd in reg)

    @staticmethod
    def cities_output():
        """
        Process and save outputs for predefined city scenarios.

        This method processes outputs for historical, piControl, ssp370, and ssp585 scenarios
        and saves the results as CSV files.

        Returns:
            None
        """
        results = (Path("./cities_MPI-ESM1-2-HR_hist_output.psz"),
                   Path("./cities_MPI-ESM1-2-HR-ssp370_output.psz"),
                   Path("./cities_MPI-ESM1-2-HR-ssp585_output.psz"))

        variables = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp",
                    "photo", "npp", "evapm", "lai", "f5", "wsoil",
                    "pupt", "nupt", "ls", "c_cost", "rcm", "storage_pool","inorg_n",
                    "inorg_p", "snc", "vcmax", 'specific_la',"cdef")

        # Determine optimal number of processes based on system resources
        available_cpus = os.cpu_count() or 4
        max_jobs = min(len(results), available_cpus)

        # Process with progress reporting
        print(f"Processing {len(results)} scenario outputs using {max_jobs} parallel jobs")
        Parallel(n_jobs=max_jobs, verbose=3)(
            delayed(output_manager.table_output_per_grd)(r, variables) for r in results
        )
        print("All scenarios processed successfully")

        return None



if __name__ == "__main__":
    """

    Direcly from the script:
    python dataframes.py --profile cities

    Dedicated profiling script
    python profile_dataframes.py --test write_daily_data --size medium --visualize

    Using the decorator:

    from dataframes import profile_function

    @profile_function("my_optimization_test")
    def my_test_function():
        # Your code here
        pass

    """
    from profiling import proftools
    ProfilerManager = proftools.ProfilerManager
    profile_function = proftools.profile_function

    # Define profiling options with argparse
    parser = argparse.ArgumentParser(description='Run CAETE-DVM with profiling options')
    parser.add_argument('--profile', choices=['none', 'cities', 'table', 'both'],
                      default='none', help='Profiling mode')
    parser.add_argument('--method', choices=['table_output', 'metacomm_output', 'make_daily_df'],
                      default=None, help='Specific method to profile')
    parser.add_argument('--trace-memory', action='store_true',
                      help='Enable memory tracing')

    # Parse command line arguments when run directly
    if Path(__file__).name in sys.argv[0]:
        args = parser.parse_args()
    else:
        # Default values for module import
        class DefaultArgs:
            def __init__(self):
                self.profile = 'none'
                self.method = None
                self.trace_memory = False
        args = DefaultArgs()

    # Profiling test cases
    def profile_cities_output():
        """Profile the cities_output method"""
        profiler = ProfilerManager("cities_output_profile")
        profiler.start()
        output_manager.cities_output()
        profiler.stop()

    def profile_table_data_method(method_name, *args, **kwargs):
        """Profile a specific method in table_data class"""
        method = getattr(table_data, method_name, None)
        if method is None:
            print(f"Method {method_name} not found in table_data class")
            return

        decorated = profile_function(f"table_data_{method_name}_profile")(method)
        return decorated(*args, **kwargs)

    # Execute based on arguments

    ## We dont have profiling for the gridded data tools yet
    if args.profile == 'none':
        # Regular execution without profiling
        output_file = Path("./pan_amazon_hist_result.psz")
        reg:region = worker.load_state_zstd(output_file)
        variables_to_read = ("npp","photo")
        a = gridded_data.create_masked_arrays(gridded_data.aggregate_region_data(reg, variables_to_read, (1,2)))

    elif args.profile == 'cities':
        # Profile cities_output method
        profile_cities_output()

    elif args.profile == 'table' and args.method:
        # Profile specific table_data method
        if args.method == 'table_output':
            results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
            variables = ("cue", "wue", "csoil", "hresp")
            profile_table_data_method('write_daily_data',
                                     worker.load_state_zstd(results), variables)

        elif args.method == 'metacomm_output':
            results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
            reg = worker.load_state_zstd(results)
            profile_table_data_method('write_metacomm_output', reg[0])

        elif args.method == 'make_daily_df':
            results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
            reg = worker.load_state_zstd(results)
            variables = ("cue", "wue", "csoil", "hresp")
            profile_table_data_method('make_daily_dataframe', reg, variables, None)

    elif args.profile == 'both':
        # Profile both cities_output and table_data
        print("Profiling cities_output...")
        profile_cities_output()

        print("\nProfilering table_data.write_daily_data...")
        results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
        reg = worker.load_state_zstd(results)
        variables = ("cue", "wue", "csoil", "hresp")
        profile_table_data_method('write_daily_data', reg, variables)

    print("\nDone.")
