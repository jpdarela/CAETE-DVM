# -*-coding:utf-8-*-
# "CAETÊ"

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""
# Author: Joao Paulo Darela Filho

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
# and create gridded and table outputs. The 

import argparse
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Collection, Dict, List, Tuple, Union

import numpy as np
import polars as pl
import pyproj

from datetime import datetime
from joblib import Parallel, delayed
from netCDF4 import Dataset
from numba import jit
from numpy.typing import NDArray

from _geos import find_coordinates_xy, get_region, pan_amazon_region
from caete_jit import pft_area_frac64, pft_area_frac
from config import fetch_config

if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import get_args, grd_mt, str_or_path
from region import region
from worker import worker

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

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
    vunits = {'header': ['long_name', 'units', 'standart_name', 'significant_digits'],
            'rsds': ['Short_wav_rad_down', 'W m-2', 'rsds', '2'],
            'wind': ['Wind_velocity', 'm s-1', 'wind', '2'],
            'ps': ['Sur_pressure', 'Pa', 'ps', '2'],
            'tas': ['Sur_temperature_2m', 'celcius', 'tas', '2'],
            'tsoil': ['Soil_temperature', 'celcius', 'soil_temp', '2'],
            'pr': ['Precipitation', 'Kg m-2 month-1', 'pr', '2'],
            'litter_l': ['Litter C flux - leaf', 'g m-2 day-1', 'll', '5'],
            'cwd': ['Litter C flux - wood', 'g m-2 day-1', 'cwd', '5'],
            'litter_fr': ['Litter C flux fine root', 'g m-2 day-1', 'lr', '5'],
            'litter_n': ['Litter Nitrogen Flux', 'g m-2 day-1', 'ln', '6'],
            'litter_p': ['Litter phosphorus flux', 'g m-2 day-1', 'lp', '6'],
            'sto_c': ['PLant Reserve Carbon', 'g m-2', 'sto_c', '4'],
            'sto_n': ['Pant Reserve Nitrogen', 'g m-2', 'sto_n', '4'],
            'sto_p': ['Plant Reserve Phosphorus', 'g m-2', 'sto_p', '6'],
            'c_cost': ['Carbon costs of Nutrients Uptake', 'g m-2 day-1', 'cc', '4'],
            'wsoil': ['Soil_water_content-wsoil', 'kg m-2', 'mrso', '2'],
            'f5': ['empirical factor of GPP penalization by soil water', '1', 'f5', '6'],
            'evapm': ['Evapotranspiration', 'kg m-2 day-1', 'et', '3'],
            'emaxm': ['Potentential evapotranspiration', 'kg m-2 day-1', 'etpot', '2'],
            'runom': ['Total_runoff', 'kg m-2 day-1', 'mrro', '4'],
            'photo': ['Gross primary productivity', 'kg m-2 year-1', 'gpp', '3'],
            'npp': ['Net primary productivity = GPP - AR', 'kg m-2 year-1', 'npp', '3'],
            'rnpp': ['Net primary productivity, C allocation', 'g m-2 day-1', 'npp', '3'],
            'lai': ['Leaf Area Index - LAI', 'm2 m-2', 'lai', '2'],
            'rcm': ['Stomatal resistence', 's m-1', 'rcm', '2'],
            'hresp': ['Soil heterotrophic respiration', 'g m-2 day-1', 'hr', '2'],
            'nupt': ['Nitrogen uptake', 'g m-2 day-1', 'nupt', '6'],
            'pupt': ['Phosphorus uptake', 'g m-2 day-1', 'pupt', '6'],
            'csoil': ['Soil Organic Carbon', 'g m-2', 'csoil', '2'],
            'csoil_1': ['Soil Organic Carbon - Litter 1', 'g m-2', 'csoil', '2'],
            'csoil_2': ['Soil Organic Carbon - Litter 2', 'g m-2', 'csoil', '2'],
            'csoil_3': ['Soil Organic Carbon - Soil 1', 'g m-2', 'csoil', '2'],
            'csoil_4': ['Soil Organic Carbon - Soil 2', 'g m-2', 'csoil', '2'],
            'org_n': ['Soil Organic Nitrogen', 'g m-2', 'org_n', '2'],
            'org_p': ['Soil Organic Phosphorus', 'g m-2', 'org_p', '2'],
            'inorg_n': ['Soil Inorganic Nitrogen', 'g m-2', 'inorg_n', '2'],
            'inorg_p': ['Soil Inorganic Phosphorus', 'g m-2', 'inorg_p', '2'],
            'sorbed_p': ['Soil Sorbed Phosphorus', 'g m-2', 'sorbed_p', '2'],
            'nmin': ['Soil Inorganic Nitrogen (solution)', 'g m-2', 'nmin', '2'],
            'pmin': ['Soil Inorganic Phosphorus (solution)', 'g m-2', 'pmin', '2'],
            'aresp': ['Autotrophic respiration', 'kg m-2 year-1', 'aresp', '3'],
            'rm': ['Maintenance respiration', 'kg m-2 year-1', 'rm', '3'],
            'rg': ['Growth respiration', 'kg m-2 year-1', 'rg', '3'],
            'wue': ['Water use efficiency', '1', 'wue', '6'],
            'vcmax': ['Maximum RuBisCo activity', 'mol m-2 s-1', 'vcmax','6'],
            'specific_la': ['Specfic leaf area', 'm2 g-1', 'sla', '7'],
            'cue': ['Carbon use efficiency', '1', 'cue', '4'],
            'cawood': ['C in woody tissues', 'kg m-2', 'cawood', '2'],
            'cfroot': ['C in fine roots', 'kg m-2', 'cfroot', '2'],
            'cleaf': ['C in leaves', 'kg m-2', 'cleaf', '2'],
            'cmass': ['Total Carbon -Biomass', 'kg m-2', 'cmass', '2'],
            'g1': ['G1 param - Stomatal Resistence model', 'hPA', 'g1', '2'],
            'resopfrac': ['Leaf resorpton fraction N & P', '%', 'resopfrac', '2'],
            'tleaf': ['Leaf C residence time', 'years', 'tleaf', '3'],
            'twood': ['Wood C residence time', 'years', 'twood', '3'],
            'troot': ['Fine root C residence time', 'years', 'troot', '3'],
            'aleaf': ['Allocation coefficients for leaf', '1', 'aleaf', '3'],
            'awood': ['Allocation coefficients for wood', '1', 'awood', '3'],
            'aroot': ['Allocation coefficients for root', '1', 'aroot', '3'],
            'c4': ['C4 photosynthesis pathway', '1', 'c4', '0'],
            'leaf_n2c': ['Leaf N:C', 'g g-1', 'leaf_n2c', '6'],
            'awood_n2c': ['Wood tissues N:C', 'g g-1', 'awood_n2c', '6'],
            'froot_n2c': ['Fine root N:C', 'g g-1', 'froot_n2c', '6'],
            'leaf_p2c': ['Leaf P:C', 'g g-1', 'leaf_p2c', '6'],
            'awood_p2c': ['Wood tissues P:C', 'g g-1', 'awood_p2c', '6'],
            'froot_p2c': ['Fine root P:C', 'g g-1', 'froot_p2c', '6'],
            'amp': ['Percentage of fine root colonized by AM', '%', 'amp', '2'],
            'pdia': ['NPP alocated to N fixers', 'fraction_of_npp', 'pdia', '4'],
            'ls': ['Living Plant Life Strategies', '1', 'ls', '0']
        }
    out = {}
    for v in var:
        out[v] = vunits.get(v, ['unknown', 'unknown', 'unknown', '5'])
    return out

def write_metadata_to_csv(variable_names:Tuple[str,...], output_path:Path | str) -> pl.DataFrame:
    """Writes metadata for given variable names to a CSV file."""

    if isinstance(variable_names, list):
        variable_names = tuple(variable_names)
    metadata = get_var_metadata(("header", ) + variable_names)
    header = metadata.pop("header")

    # Convert nested dict to pl.DataFrame
    data = []
    for var_name, values in metadata.items():
        data.append([var_name] + values)

    df = pl.DataFrame(data, schema=["variable_name"] + header, orient="row")
    df.write_csv(str_or_path(output_path) / "output_metadata.csv")
    return df


#=========================================
# Functions dealing with gridded outputs
#=========================================
class gridded_data:
    """This class contains methods to read and process gridded data from the model outputs.
    """
     # Daily data --------------------------------
    @staticmethod
    def read_grd(grd: grd_mt,
                 variables: Union[str, Collection[str]],
                 spin_slice: Union[int, Tuple[int, int], None]
                 ) -> Tuple[NDArray, Union[Dict, NDArray, List, Tuple], Union[int, float], Union[int, float]]:
        """ Reads the data from a gridcell and returns a tuple with the following keys:
        time, coord, data, holding data to be transformed.

        Args:
            grd (grd_mt): A gridcell object
            variables (Collection[str]): which variables to read from the gridcell
            spin_slice (Union[int, Tuple[int, int], None]): which spin slice to read

        Returns:
            Tuple[NDArray, Union[Dict, NDArray, List, Tuple], Union[int, float], Union[int, float]]:
              a tuple with the following keys: time, coord, data holding data to be transformed

        """
        # Get the raw data - this can return different formats depending on input
        raw_data = grd._get_daily_data(get_args(variables), spin_slice, return_time=True)

        time = raw_data[-1]  # Time is always the last element
        data_part = raw_data[0]  # Data is always the first element

        # Handle different return formats from _get_daily_data
        if isinstance(variables, str):
            # Single variable case - _get_daily_data returns just the array
            # We need to convert it to a dictionary format
            data = {variables: data_part}
        else:
            # Multiple variables case - _get_daily_data returns a dictionary
            data = data_part

        return time, data, grd.y, grd.x


    @staticmethod
    def aggregate_region_data(r: region,
                              variables: Union[str, Collection[str]],
                              spin_slice: Union[int, Tuple[int, int], None] = None,
                              temp_dir: Path = None,
                              batch_size: int = 580,
                            ) -> Dict[str, NDArray]:
        """Fully memory-mapped version for large regions"""

        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "caete_memmap"
        temp_dir.mkdir(exist_ok=True)

        n_gridcells = len(r)

        # Get sample data to determine dimensions
        sample_data = gridded_data.read_grd(r[0], variables, spin_slice)
        time_data = sample_data[0]
        sample_vars = sample_data[1]

        # Determine variable shapes and dtypes
        var_info = {}
        if isinstance(variables, str):
            var_list = [variables]
        else:
            var_list = list(variables)

        for var_name in var_list:
            var_data = sample_vars[var_name]
            var_info[var_name] = {
                'shape': var_data.shape,
                'dtype': var_data.dtype
            }

        # Create memory-mapped arrays for coordinates
        coord_memmap_file = temp_dir / f"coords_{os.getpid()}.dat"
        coord_memmap = np.memmap(
            coord_memmap_file,
            dtype=np.int64,
            mode='w+',
            shape=(n_gridcells, 2)
        )

        # Create memory-mapped arrays for EACH VARIABLE
        var_memmaps = {}
        var_files = {}

        for var_name, info in var_info.items():
            var_file = temp_dir / f"{var_name}_{os.getpid()}.dat"
            var_files[var_name] = var_file

            # Create memory-mapped array for this variable
            if len(info['shape']) == 1:
                # 1D variable (time series)
                full_shape = (n_gridcells, info['shape'][0])
            elif len(info['shape']) == 2:
                # 2D variable (e.g., multiple PFTs over time)
                full_shape = (n_gridcells, info['shape'][0], info['shape'][1])
            else:
                raise ValueError(f"Unsupported variable shape: {info['shape']}")

            var_memmaps[var_name] = np.memmap(
                var_file,
                dtype=info['dtype'],
                mode='w+',
                shape=full_shape
            )

        # Process batches and store directly in memory-mapped arrays
        def read_batch_parallel(batch_indices, batch_gridcells):
            """Read batch and return structured results"""
            batch_results = []

            with ThreadPoolExecutor(max_workers=config.multiprocessing.nprocs) as batch_executor:
                future_to_idx = {}
                for local_idx, (global_idx, grd) in enumerate(zip(batch_indices, batch_gridcells)):
                    future = batch_executor.submit(gridded_data.read_grd, grd, variables, spin_slice)
                    future_to_idx[future] = global_idx

                for future in as_completed(future_to_idx):
                    global_idx = future_to_idx[future]
                    try:
                        time, data, y, x = future.result()
                        batch_results.append((global_idx, data, y, x))
                    except Exception as e:
                        print(f"Error reading gridcell: {e}")

            return batch_results

        # Process in batches
        processed_count = 0
        for batch_start in range(0, n_gridcells, batch_size):
            batch_end = min(batch_start + batch_size, n_gridcells)
            batch_indices = list(range(batch_start, batch_end))
            batch_gridcells = [r[i] for i in batch_indices]

            # Read batch
            batch_results = read_batch_parallel(batch_indices, batch_gridcells)

            # Store results directly in memory-mapped arrays
            for global_idx, data, y, x in batch_results:
                # Store coordinates
                coord_memmap[global_idx] = [y, x]

                # Store variable data in memory-mapped arrays
                for var_name in var_list:
                    if var_name in data:
                        var_data = data[var_name]
                        if len(var_data.shape) == 1:
                            var_memmaps[var_name][global_idx, :] = var_data
                        elif len(var_data.shape) == 2:
                            var_memmaps[var_name][global_idx, :, :] = var_data

            processed_count += len(batch_results)

            # Force flush to disk
            for var_memmap in var_memmaps.values():
                var_memmap.flush()
            coord_memmap.flush()

            # Garbage collection
            if batch_start % (batch_size * 4) == 0:
                gc.collect()

        # Convert memory-mapped arrays to regular arrays for return
        coord_array = np.array(coord_memmap)

        # CONSISTENT RETURN FORMAT: Always return as list of dicts
        variable_data = []

        # Convert each gridcell's data to the expected dict format
        for i in range(processed_count):
            gridcell_data = {}
            for var_name in var_list:
                gridcell_data[var_name] = var_memmaps[var_name][i]
            variable_data.append(gridcell_data)

        # Clean up memory-mapped files
        try:
            coord_memmap_file.unlink(missing_ok=True)
            for var_file in var_files.values():
                var_file.unlink(missing_ok=True)
        except:
            pass

        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except:
            pass

        # CONSISTENT RETURN FORMAT: Always return with "data" key containing list of dicts
        return {
            "time": time_data,
            "coord": coord_array[:processed_count],  # Trim to actual processed data
            "data": variable_data  # List of dicts, one per gridcell - CONSISTENT for both single and multiple variables
        }


    @staticmethod
    def read_grd_annual(grd: grd_mt,
                       data_type: str = "biomass",
                       years: Union[int, List[int], None] = None
                       ) -> Tuple[NDArray, Dict, Union[int, float], Union[int, float]]:
        """Read annual data from a gridcell for gridded output

        Args:
            grd: Gridcell object
            data_type: Type of annual data ("biomass", "productivity", etc.)
            years: Specific years to read (None for all available)

        Returns:
            Tuple of (years_array, data_dict, y, x)
        """
        available_years = grd._get_years()

        if years is None:
            target_years = available_years
        elif isinstance(years, int):
            target_years = [years] if years in available_years else []
        else:
            target_years = [y for y in years if y in available_years]

        if not target_years:
            raise ValueError(f"No valid years found. Available: {available_years}")

        if data_type == "biomass":
            # Process biomass data for all years
            all_data = {}

            for year in target_years:
                year_data = grd._read_annual_metacomm_biomass(year)
                if year_data:
                    # Convert to DataFrame and process like in table_data
                    df = pl.DataFrame(year_data)

                    # Calculate aggregated metrics
                    grouped = df.group_by("pls_id").agg([
                        pl.col("vp_cleaf").mean().alias("cleaf"),
                        pl.col("vp_croot").mean().alias("croot"),
                        pl.col("vp_cwood").mean().alias("cwood"),
                        pl.len().alias("count")
                    ])

                    # Calculate cveg and ocp using the optimized function
                    # Ensure float64 for Numba compatibility
                    cleaf = grouped.get_column("cleaf").to_numpy().astype(np.float64)
                    croot = grouped.get_column("croot").to_numpy().astype(np.float64)
                    cwood = grouped.get_column("cwood").to_numpy().astype(np.float64)

                    cveg, ocp = calculate_cveg_and_ocp(cleaf, croot, cwood)

                    # Store aggregated annual data
                    all_data[year] = {
                        "pls_id": grouped.get_column("pls_id").to_numpy(),
                        "cleaf": cleaf,
                        "croot": croot,
                        "cwood": cwood,
                        "cveg": cveg,
                        "ocp": ocp,
                        "count": grouped.get_column("count").to_numpy()
                    }

            # Organize data by variable across years
            if all_data:
                variables = ["cleaf", "croot", "cwood", "cveg", "ocp", "count"]
                organized_data = {}
                years_array = np.array(sorted(all_data.keys()))

                # Get maximum number of PFTs across all years for consistent array size
                max_pfts = max(len(all_data[year]["pls_id"]) for year in all_data.keys())

                for var in variables:
                    # Create (n_years, n_pfts) array
                    var_array = np.full((len(years_array), max_pfts), np.nan)

                    for i, year in enumerate(years_array):
                        year_data = all_data[year][var]
                        var_array[i, :len(year_data)] = year_data

                    organized_data[var] = var_array

                # Also store PLS IDs (use the year with most PFTs as reference)
                reference_year = max(all_data.keys(), key=lambda y: len(all_data[y]["pls_id"]))
                pls_ids = np.full(max_pfts, -1)
                ref_ids = all_data[reference_year]["pls_id"]
                pls_ids[:len(ref_ids)] = ref_ids
                organized_data["pls_id"] = pls_ids

                return years_array, organized_data, grd.y, grd.x

        return np.array([]), {}, grd.y, grd.x


    @staticmethod
    def aggregate_region_annual_data(r: region,
                                    data_type: str = "biomass",
                                    years: Union[int, List[int], None] = None,
                                    temp_dir: Path = None,
                                    batch_size: int = 580
                                    ) -> Dict[str, NDArray]:
        """Aggregate annual data across a region for gridded output"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if temp_dir is None:
            # Use a simpler temp directory path for Windows compatibility
            temp_dir = Path.cwd() / "temp_annual_data"

        temp_dir.mkdir(exist_ok=True)

        n_gridcells = len(r)

        # Get sample data to determine structure
        sample_data = gridded_data.read_grd_annual(r[0], data_type, years)
        years_array = sample_data[0]
        sample_vars = sample_data[1]

        if len(years_array) == 0:
            raise ValueError("No valid annual data found in sample gridcell")

        # Determine data structure
        var_info = {}
        for var_name, var_data in sample_vars.items():
            var_info[var_name] = {
                'shape': var_data.shape,
                'dtype': var_data.dtype
            }

        # Create memory-mapped arrays with Windows-safe file names
        coord_memmap_file = temp_dir / f"coords.dat"
        coord_memmap = np.memmap(
            coord_memmap_file,
            dtype=np.int64,
            mode='w+',
            shape=(n_gridcells, 2)
        )

        var_memmaps = {}
        var_files = {}

        for var_name, info in var_info.items():
            # Use simple file names for Windows compatibility
            var_file = temp_dir / f"{var_name}.dat"
            var_files[var_name] = var_file

            # Annual data has shape (n_gridcells, n_years, n_pfts) for 2D vars
            if len(info['shape']) == 1:
                full_shape = (n_gridcells, info['shape'][0])
            elif len(info['shape']) == 2:
                full_shape = (n_gridcells, info['shape'][0], info['shape'][1])
            else:
                raise ValueError(f"Unsupported annual data shape: {info['shape']}")

            var_memmaps[var_name] = np.memmap(
                var_file,
                dtype=info['dtype'],
                mode='w+',
                shape=full_shape
            )

            # Initialize with appropriate fill values based on dtype
            if np.issubdtype(info['dtype'], np.integer):
                if var_name == "pls_id":
                    var_memmaps[var_name][:] = -1
                else:
                    var_memmaps[var_name][:] = 0
            else:
                var_memmaps[var_name][:] = np.nan

        # Process in batches (rest of the function remains the same)
        def read_annual_batch_parallel(batch_indices, batch_gridcells):
            batch_results = []

            with ThreadPoolExecutor(max_workers=config.multiprocessing.nprocs) as executor:
                future_to_idx = {}
                for global_idx, grd in zip(batch_indices, batch_gridcells):
                    future = executor.submit(gridded_data.read_grd_annual, grd, data_type, years)
                    future_to_idx[future] = global_idx

                for future in as_completed(future_to_idx):
                    global_idx = future_to_idx[future]
                    try:
                        years_data, data, y, x = future.result()
                        if len(years_data) > 0:
                            batch_results.append((global_idx, data, y, x))
                    except Exception as e:
                        print(f"Error reading annual data for gridcell: {e}")

            return batch_results

        processed_count = 0
        for batch_start in range(0, n_gridcells, batch_size):
            batch_end = min(batch_start + batch_size, n_gridcells)
            batch_indices = list(range(batch_start, batch_end))
            batch_gridcells = [r[i] for i in batch_indices]

            batch_results = read_annual_batch_parallel(batch_indices, batch_gridcells)

            # Store results in memory-mapped arrays
            for global_idx, data, y, x in batch_results:
                coord_memmap[global_idx] = [y, x]

                for var_name, var_data in data.items():
                    if var_name in var_memmaps:
                        # Handle different dimensionalities
                        if len(var_data.shape) == 1:
                            # Handle 1D arrays by only copying what fits
                            dest_length = var_memmaps[var_name][global_idx].shape[0]
                            # Copy only up to the minimum length
                            length_to_copy = min(var_data.shape[0], dest_length)
                            var_memmaps[var_name][global_idx, :length_to_copy] = var_data[:length_to_copy]
                        elif len(var_data.shape) == 2:
                            # Handle varying PFT counts by only copying what fits
                            dest_shape = var_memmaps[var_name][global_idx].shape
                            # Copy only up to the minimum size in each dimension
                            rows_to_copy = min(var_data.shape[0], dest_shape[0])
                            cols_to_copy = min(var_data.shape[1], dest_shape[1])
                            var_memmaps[var_name][global_idx, :rows_to_copy, :cols_to_copy] = var_data[:rows_to_copy, :cols_to_copy]

            processed_count += len(batch_results)

            # Flush and cleanup
            for var_memmap in var_memmaps.values():
                var_memmap.flush()
            coord_memmap.flush()

            if batch_start % (batch_size * 4) == 0:
                gc.collect()

        # Convert to regular arrays and format return
        coord_array = np.array(coord_memmap)

        # Convert each gridcell's data to expected format
        variable_data = []
        var_names = list(var_info.keys())

        for i in range(processed_count):
            gridcell_data = {}
            for var_name in var_names:
                gridcell_data[var_name] = var_memmaps[var_name][i]
            variable_data.append(gridcell_data)

        # Cleanup
        try:
            coord_memmap_file.unlink(missing_ok=True)
            for var_file in var_files.values():
                var_file.unlink(missing_ok=True)
            temp_dir.rmdir()
        except:
            pass

        return {
            "years": years_array,
            "coord": coord_array[:processed_count],
            "data": variable_data
        }


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
        NODATA = 1.0e20

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
                new_array = np.ma.masked_all(shape=(dim[0], region_height, region_width), dtype=dtypes[i])
                # np.ma.set_fill_value(new_array, NODATA)
                arrays.append(new_array)
                array_names.append(var)
            elif len(dim) == 2:
                ny, nx = dim
                for k in range(ny):
                    new_array = np.ma.masked_all(shape=(nx, region_height, region_width), dtype=dtypes[i])
                    # np.ma.set_fill_value(new_array, NODATA)
                    arrays.append(new_array)
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

    ## TODO Annual gridded data ------> The creation of masked arrays for annual data is
    # not mature (to be saved in netCDF files). We only output annual biomass data in table format for now.

    @staticmethod
    def create_annual_masked_arrays(data: dict, data_type: str = "biomass"):
        """Create masked arrays from annual aggregated data"""
        years = data["years"]
        coords = data["coord"]
        arrays_dict = data["data"]

        if not arrays_dict:
            return [], years, []

        # Get variable names
        variables = list(arrays_dict[0].keys())

        # Get region bounds from coordinates
        ymin_coord, ymax_coord = coords[:, 0].min(), coords[:, 0].max()
        xmin_coord, xmax_coord = coords[:, 1].min(), coords[:, 1].max()

        # Calculate region dimensions
        region_height = ymax_coord - ymin_coord + 1
        region_width = xmax_coord - xmin_coord + 1

        print(f"Annual data bounds: Y={ymin_coord}-{ymax_coord}, X={xmin_coord}-{xmax_coord}")
        print(f"Annual array shape: {region_height} × {region_width}")

        arrays = []
        array_names = []

        # For biomass data, create CWM arrays
        if data_type == "biomass":
            # Create arrays for Community Weighted Mean biomass variables
            biomass_vars = ["cleaf", "croot", "cwood", "cveg"]

            for var in biomass_vars:
                if var in variables:
                    # Create arrays for each year (CWM values)
                    for year_idx, year in enumerate(years):
                        arrays.append(np.ma.masked_all(shape=(region_height, region_width), dtype=np.float64))
                        array_names.append(f"{var}_cwm_{year}")

                    # Create mean across years
                    arrays.append(np.ma.masked_all(shape=(region_height, region_width), dtype=np.float64))
                    array_names.append(f"{var}_cwm_mean")

            # Create PFT richness array
            if "pls_id" in variables:
                arrays.append(np.ma.masked_all(shape=(region_height, region_width), dtype=np.int32))
                array_names.append("pft_richness")

            # Create total occupation array (sum of all ocp values)
            if "ocp" in variables:
                for year_idx, year in enumerate(years):
                    arrays.append(np.ma.masked_all(shape=(region_height, region_width), dtype=np.float64))
                    array_names.append(f"total_ocp_{year}")

                arrays.append(np.ma.masked_all(shape=(region_height, region_width), dtype=np.float64))
                array_names.append("total_ocp_mean")

        # Fill the arrays
        array_index = 0

        if data_type == "biomass":
            biomass_vars = ["cleaf", "croot", "cwood", "cveg"]

            # Process biomass variables with CWM calculation
            for var in biomass_vars:
                if var in variables:
                    for j in range(len(coords)):
                        y_rel = coords[j][0] - ymin_coord
                        x_rel = coords[j][1] - xmin_coord

                        if 0 <= y_rel < region_height and 0 <= x_rel < region_width:
                            var_data = arrays_dict[j][var]  # Shape: (n_years, n_pfts)
                            ocp_data = arrays_dict[j]["ocp"]  # Shape: (n_years, n_pfts)

                            if len(var_data.shape) == 2 and len(ocp_data.shape) == 2:
                                n_years, n_pfts = var_data.shape

                                # Calculate CWM for each year
                                cwm_values = []
                                for year_idx in range(n_years):
                                    # Community Weighted Mean: sum(biomass * occupation)
                                    valid_mask = ~np.isnan(var_data[year_idx, :]) & ~np.isnan(ocp_data[year_idx, :])
                                    if np.any(valid_mask):
                                        cwm = np.sum(var_data[year_idx, valid_mask] * ocp_data[year_idx, valid_mask])
                                        if array_index + year_idx < len(arrays):
                                            arrays[array_index + year_idx][y_rel, x_rel] = cwm
                                        cwm_values.append(cwm)

                                # Calculate mean CWM across years
                                if cwm_values and array_index + n_years < len(arrays):
                                    mean_cwm = np.mean(cwm_values)
                                    arrays[array_index + n_years][y_rel, x_rel] = mean_cwm

                    array_index += len(years) + 1  # +1 for mean

            # Process PFT richness - FIXED FOR 2D ARRAYS
            if "pls_id" in variables:
                print("Processing PFT richness...")
                richness_count = 0

                for j in range(len(coords)):
                    y_rel = coords[j][0] - ymin_coord
                    x_rel = coords[j][1] - xmin_coord

                    if 0 <= y_rel < region_height and 0 <= x_rel < region_width:
                        pls_data = arrays_dict[j]["pls_id"]

                        # Debug: Check the structure of pls_data
                        if j == 0:  # Only print for first gridcell
                            print(f"First gridcell pls_data shape: {pls_data.shape}")

                        if len(pls_data.shape) == 2:
                            # 2D array: (n_years, n_pfts) - CORRECT HANDLING
                            # Count unique valid PFTs across all years
                            valid_pfts = pls_data[pls_data >= 0]
                            unique_pfts = len(np.unique(valid_pfts)) if len(valid_pfts) > 0 else 0

                            if unique_pfts > 0:
                                richness_count += 1
                                if array_index < len(arrays):
                                    arrays[array_index][y_rel, x_rel] = unique_pfts

                        elif len(pls_data.shape) == 1:
                            # 1D array: just count valid PFTs (fallback for old format)
                            valid_pfts = pls_data[pls_data >= 0]
                            unique_pfts = len(np.unique(valid_pfts)) if len(valid_pfts) > 0 else 0

                            if unique_pfts > 0:
                                richness_count += 1
                                if array_index < len(arrays):
                                    arrays[array_index][y_rel, x_rel] = unique_pfts

                print(f"Processed PFT richness for {richness_count} gridcells")
                array_index += 1

            # Process total occupation (same logic as biomass vars)
            if "ocp" in variables:
                for j in range(len(coords)):
                    y_rel = coords[j][0] - ymin_coord
                    x_rel = coords[j][1] - xmin_coord

                    if 0 <= y_rel < region_height and 0 <= x_rel < region_width:
                        ocp_data = arrays_dict[j]["ocp"]

                        if len(ocp_data.shape) == 2:
                            n_years, n_pfts = ocp_data.shape

                            # Calculate total occupation for each year
                            total_ocp_values = []
                            for year_idx in range(n_years):
                                valid_mask = ~np.isnan(ocp_data[year_idx, :])
                                if np.any(valid_mask):
                                    total_ocp = np.sum(ocp_data[year_idx, valid_mask])
                                    if array_index + year_idx < len(arrays):
                                        arrays[array_index + year_idx][y_rel, x_rel] = total_ocp
                                    total_ocp_values.append(total_ocp)

                            # Calculate mean total occupation
                            if total_ocp_values and array_index + n_years < len(arrays):
                                mean_total_ocp = np.mean(total_ocp_values)
                                arrays[array_index + n_years][y_rel, x_rel] = mean_total_ocp

        return arrays, years, array_names


    @staticmethod
    def process_annual_gridded_data(r: region,
                                   data_type: str = "biomass",
                                   years: Union[int, List[int], None] = None,
                                   output_path: Path = None,
                                   file_name: str = None
                                   ) -> Tuple[List, NDArray, List[str]]:
        """Complete workflow for processing annual data into gridded format

        Args:
            r: Region object
            data_type: Type of annual data ("biomass", "productivity")
            years: Years to process
            output_path: Path to save NetCDF file (optional)
            file_name: Name for output file (optional)

        Returns:
            Tuple of (masked_arrays, years, array_names)
        """
        print(f"Aggregating annual {data_type} data...")
        aggregated_data = gridded_data.aggregate_region_annual_data(r, data_type, years)

        print("Creating masked arrays...")
        arrays, years_array, array_names = gridded_data.create_annual_masked_arrays(aggregated_data, data_type)

        if output_path and file_name:
            print(f"Saving to NetCDF: {output_path / file_name}")
            # TODO: Implement save_annual_netcdf method
            # gridded_data.save_annual_netcdf(arrays, years_array, array_names, output_path, file_name)

        return arrays, years_array, array_names
    # End of annual gridded data methods ------

    @staticmethod
    def save_netcdf_daily(data: List, run_name="caete_run"):
        """Saves gridded data to a NetCDF file. Only implemented for daily data so far.

        Args:
            data (List): List of masked arrays to save.
            output_path (Path | None): Path to save the NetCDF file.

        Returns:
            None
        """
        vnames = data[-1]
        arr = data[0]
        time = data[1]
        output_path = Path(config.output.output_dir)

        var_metadata = get_var_metadata(vnames)

        # Time metadata
        calendar = time[0].calendar
        time_units = f"days since {time[0].isoformat()}"
        time_vals = np.arange(len(time), dtype=np.float64)

        # Geographic coordinates
        lat_south, lon_east = find_coordinates_xy(ymax, xmax)
        lat_north, lon_west = find_coordinates_xy(ymin, xmin)
        # lats = np.arange(lat_north, lat_south, config.crs.res * (-1) , dtype=np.float32)
        # lons = np.arange(lon_west, lon_east, config.crs.res * (-1), dtype=np.float32)
        # FIXED: Latitude decreases from north to south (positive to negative)
        lats = np.arange(lat_north, lat_south, -config.crs.res, dtype=np.float32)

        # FIXED: Longitude INCREASES from west to east (negative to positive for Americas)
        lons = np.arange(lon_west, lon_east, config.crs.res, dtype=np.float32)

        # CRS information
        crs_metadata = pyproj.CRS(config.crs.datum).to_cf()

        # Loop through variables and save each to its own NetCDF file
        for i, var in enumerate(vnames):
            print(f"Variable {i+1}/{len(vnames)}: {var}", end=' -> ')
            var_data = arr[i]


            fill_value = var_data.get_fill_value()
            var_dtype = var_data.dtype

            # # Update mask to include NaN values
            var_data.mask = np.logical_or(var_data.mask, np.isnan(var_data))

            # Ensure all masked values use the correct fill_value
            np.ma.set_fill_value(var_data, fill_value)
            # large_value_mask = var_data.data > 1e17
            # var_data.data[large_value_mask] = fill_value
            # var_data.mask = np.logical_or(var_data.mask, large_value_mask)

            nc_filename = output_path / f"{var}_{run_name}_{time[0].strftime('%Y%m%d')}_{time[-1].strftime('%Y%m%d')}.nc"
            print(nc_filename)

            rootgrp = Dataset(nc_filename, "w", format="NETCDF4")

            sig_dig = int(var_metadata[var][3])

            # Create dimensions
            rootgrp.createDimension("time", None)
            rootgrp.createDimension("lat", len(lats))
            rootgrp.createDimension("lon", len(lons))

            # Create variables
            times = rootgrp.createVariable("time", "f8", ("time",))
            latitudes = rootgrp.createVariable("lat", "f4", ("lat",))
            longitudes = rootgrp.createVariable("lon", "f4", ("lon",))
            crs = rootgrp.createVariable("crs", "i4")
            var_nc = rootgrp.createVariable(var,
                                            var_dtype,
                                            ("time", "lat", "lon"),
                                            fill_value=fill_value,
                                            significant_digits=sig_dig,
                                            compression="zlib",
                                            complevel=9,
                                            fletcher32=True)

            # Assign data to variables
            # TIme
            times[:] = time_vals
            times.units = time_units
            times.calendar = calendar
            times.axis = "T"
            times.standard_name = "time"
            times.long_name = "Time"

            # Spatial coordinates
            #Lat
            latitudes[:] = lats
            latitudes.units = "degrees_north"
            latitudes.axis = "Y"
            latitudes.standard_name = "latitude"
            latitudes.long_name = "Latitude"
            latitudes.valid_min = lats.min()
            latitudes.valid_max = lats.max()
            latitudes.delta = -config.crs.res  # Latitude decreases as index increases
            latitudes.spacing = -config.crs.res # Latitude decreases as index increases

            #Lon
            longitudes[:] = lons
            longitudes.units = "degrees_east"
            longitudes.axis = "X"
            longitudes.standard_name = "longitude"
            longitudes.long_name = "Longitude"
            longitudes.valid_min = lons.min()
            longitudes.valid_max = lons.max()
            longitudes.delta = config.crs.res
            longitudes.spacing = config.crs.res

            # CRS
            for k, v in crs_metadata.items():
                setattr(crs, k, v)


            # Variable
            var_nc[:, :, :] = var_data
            var_nc.units = var_metadata[var][1]
            var_nc.standard_name = var_metadata[var][2]
            var_nc.long_name = var_metadata[var][0]
            var_nc.description = f"{var_metadata[var][0]} from CAETE model run {run_name}"
            var_nc.history = f"Created: {datetime.now().isoformat()}"
            var_nc.source = "CAETE model"
            var_nc.references = ""
            var_nc.comment = ""
            rootgrp.close()

# ======================================
# Functions dealing with table outputs
# ======================================
# Standalone numba function for optimized calculations
@jit(nopython=True, cache=True)
def calculate_cveg_and_ocp(cleaf, croot, cwood):
    """Numba-optimized function to calculate total vegetation carbon and area fraction.

    Expects float64 arrays as input.

    Args:
        cleaf: Numpy array of leaf carbon values (float64)
        croot: Numpy array of root carbon values (float64)
        cwood: Numpy array of wood carbon values (float64)

    Returns:
        Tuple of (cveg, ocp) arrays
    """
    # Calculate total vegetation carbon
    cveg = cleaf + croot + cwood

    # Calculate area fraction using the existing function
    ocp = pft_area_frac64(cleaf, croot, cwood)

    return cveg, ocp


class table_data:
    """
    Extracts and processes table data from the model outputs.
    This class contains methods to create daily dataframes for each grid cell in a region
    and save them as CSV files. It also includes methods to write yearly metacommunity biomass output
    to CSV files. Use this class to handle outputs from a small number of gridcells efficiently.
    The output file sizes can be quite large if it contains many variables for large regions.

    Note:
    Do not use this class directly. It is designed to be used with the output_manager class.
    It is optimized for performance and memory usage, but may not handle very large datasets efficiently.

    May not be suitable for very large datasets due to memory constraints.
    Use with caution for large regions or many variables.
    """
    @staticmethod
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
        # Load all the gridcells in the region to accelerate processing. Consumes a lot of memory if the region is large.
        if not r.gridcells:
            r.load_gridcells()
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
            new_data = table_data.process_arrays(keys, arrays, shapes)

            # Add time information
            new_data['day'] = time

            fname = f"grd_{grd.y}_{grd.x}_{time[0]}_{time[-1]}.csv"

            # Create DataFrame with polars and write to CSV efficiently
            df = pl.DataFrame(new_data)
            df.write_csv(grd.out_dir / fname)

    @staticmethod
    def write_daily_data(r: region, variables: Union[str, Collection[str]]):
        """Writes the daily data to csv files"""

        # Load all the gridcells in the region to accelerate processing. Consumes a lot of memory if the region is large.
        if not r.gridcells:
            r.load_gridcells()

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
    def write_metacomm_output(grd:grd_mt, year=None) -> None:
        """Writes the metacommunity biomass output (C in vegetation and abundance) to a csv file

        Args:
            grd (grd_mt): gridcell object
        """
        all_df_list = []
        if year is None:
            years = grd._get_years() # get all available years
        else:
            years = get_args(year)

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
    def table_output_per_grd(result:Union[Path, str], variables:Union[str, Collection[str]]):
        """
        Process table outputs for a grid (region) and save them as gridcell-individual CSV files.

        Args:
            result (Union[Path, str]): Path to the serialized state file containing the region data.
            variables (Union[str, Collection[str]]): Variable names to process and save.

        Returns:
            None
        """
        # Load state data
        reg:region = worker.load_state_zstd(str_or_path(result))
        # Load all the gridcells in the region to accelerate processing. Consumes a lot of memory if the region is large.
        if not reg.gridcells:
            reg.load_gridcells()

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
        Parallel(n_jobs=nprocs, verbose=1, prefer="threads")(delayed(process_gridcell)(grd) for grd in reg)

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

        # Make copies of arrays to ensure they're writeable (not readonly)
        pls_id_values = df.get_column("pls_id").to_numpy().copy()
        cleaf = df.get_column("vp_cleaf").to_numpy().copy()
        croot = df.get_column("vp_croot").to_numpy().copy()
        cwood = df.get_column("vp_cwood").to_numpy().copy()

        # Use the numba optimized functions to calculate cveg and ocp
        try:
            cveg, ocp = calculate_cveg_and_ocp(cleaf, croot, cwood)
        except Exception:
            # Fallback to 32bit calculation if there's an error
            # print("FALLING_BACK TO 32BIT OCP CALCULATION")
            ocp = pft_area_frac(cleaf, croot, cwood)
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

    @staticmethod
    def _extract_coordinates_from_filename(csv_path: Path) -> Tuple[int, int]:
        """Extract y, x coordinates from grd_y_x_startdate_enddate.csv filename"""
        parts = csv_path.stem.split('_')
        if len(parts) >= 3:
            try:
                y, x = int(parts[1]), int(parts[2])
                return y, x
            except (ValueError, IndexError):
                pass
        return -1, -1

    @staticmethod
    def _indices_to_latlon(y: int, x: int, res_y: float = 0.5, res_x: float = 0.5) -> Tuple[float, float]:
        """Convert grid indices to latitude/longitude using _geos functionality"""

        return find_coordinates_xy(y, x, res_y, res_x)

    @staticmethod
    def _process_single_csv_for_consolidation(csv_path: Path) -> pl.DataFrame:
        """Process a single CSV file and add coordinate columns"""
        try:
            # Read CSV with polars
            df = pl.read_csv(csv_path)

            # Extract coordinates from filename
            y, x = table_data._extract_coordinates_from_filename(csv_path)

            if y == -1 or x == -1:
                print(f"Warning: Could not extract coordinates from {csv_path.name}")
                return pl.DataFrame()

            # Convert indices to lat/lon
            lat, lon = table_data._indices_to_latlon(y, x)

            # Add coordinate columns
            df = df.with_columns([
                pl.lit(y).alias("grid_y"),
                pl.lit(x).alias("grid_x"),
                pl.lit(lat).alias("latitude"),
                pl.lit(lon).alias("longitude")
            ])

            return df

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return pl.DataFrame()

    @staticmethod
    def consolidate_daily_outputs(experiment_dir: Path,
                                output_format: str = "parquet",
                                chunk_size: int = 500) -> None:
        """
        Consolidate daily CSV outputs from multiple gridcells into a single file.

        Args:
            experiment_dir (Path): Directory containing gridcell folders with CSV files
            output_format (str): Output format - "parquet", "feather", or "hdf5"
            chunk_size (int): Number of CSV files to process in each chunk
        """
        print(f"Consolidating daily outputs in {experiment_dir.name}")

        # Find all CSV files with the expected pattern
        csv_files = list(experiment_dir.rglob("grd_*.csv"))

        if not csv_files:
            print(f"No CSV files found in {experiment_dir}")
            return

        print(f"Found {len(csv_files)} CSV files")

        # Process files in chunks to manage memory
        n_workers = min(os.cpu_count() or 4, 8)  # Cap workers to avoid I/O saturation
        all_dfs = []

        for i in range(0, len(csv_files), chunk_size):
            chunk = csv_files[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(csv_files)-1)//chunk_size + 1}")

            # Process chunk in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                chunk_dfs = list(executor.map(table_data._process_single_csv_for_consolidation, chunk))

            # Filter out empty DataFrames and concatenate
            valid_dfs = [df for df in chunk_dfs if len(df) > 0]
            if valid_dfs:
                chunk_combined = pl.concat(valid_dfs, rechunk=True)
                all_dfs.append(chunk_combined)

            # Force garbage collection between chunks

            gc.collect()

        # Combine all chunks
        if not all_dfs:
            print(f"No valid data found in {experiment_dir}")
            return

        print("Combining all chunks...")
        final_df = pl.concat(all_dfs, rechunk=True)

        # Sort by coordinates and date for better organization
        final_df = final_df.sort(["grid_y", "grid_x", "day"])

        # Write consolidated file based on format
        output_file = experiment_dir / f"{experiment_dir.name}"

        if output_format.lower() == "parquet":
            table_data._write_parquet_consolidated(final_df, output_file.with_suffix('.parquet'))
        elif output_format.lower() == "feather":
            table_data._write_feather_consolidated(final_df, output_file.with_suffix('.feather'))
        elif output_format.lower() == "hdf5":
            table_data._write_hdf5_consolidated(final_df, output_file.with_suffix('.h5'))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"Successfully consolidated {len(csv_files)} files")

    @staticmethod
    def _write_parquet_consolidated(df: pl.DataFrame, output_path: Path) -> None:
        """Write consolidated data to Parquet format with partitioning"""
        print(f"Writing Parquet file: {output_path}")

        # Add partition columns for better query performance
        df_partitioned = df.with_columns([
            pl.col("day").str.slice(0, 4).alias("year"),
            pl.col("day").str.slice(5, 2).alias("month")
        ])

        # Write as partitioned dataset for better performance
        df_partitioned.write_parquet(
            output_path,
            compression="snappy",
            row_group_size=50000,
            use_pyarrow=True
        )

    @staticmethod
    def _write_feather_consolidated(df: pl.DataFrame, output_path: Path) -> None:
        """Write consolidated data to Feather format"""
        print(f"Writing Feather file: {output_path}")

        # Feather doesn't support partitioning, but is very fast for reading
        df.write_ipc(output_path, compression="zstd")

    @staticmethod
    def _write_hdf5_consolidated(df: pl.DataFrame, output_path: Path) -> None:
        """Write consolidated data to HDF5 format with hierarchical structure"""
        print(f"Writing HDF5 file: {output_path}")

        try:
            import h5py
        except ImportError:
            print("h5py not available. Please install it with: pip install h5py")
            return

        # Convert to pandas for HDF5 compatibility
        df_pandas = df.to_pandas()

        with h5py.File(output_path, 'w') as f:
            # Create groups for metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['description'] = 'Consolidated daily outputs from CAETE-DVM'

            # Use standard datetime for creation date
            from datetime import datetime
            meta_group.attrs['creation_date'] = str(datetime.now())
            meta_group.attrs['total_gridcells'] = len(df.select("grid_y", "grid_x").unique())
            meta_group.attrs['date_range'] = f"{df['day'].min()} to {df['day'].max()}"

            # Store coordinate information
            coords_group = f.create_group('coordinates')
            unique_coords = df.select("grid_y", "grid_x", "latitude", "longitude").unique().to_pandas()
            coords_group.create_dataset('grid_y', data=unique_coords['grid_y'].values, compression='gzip')
            coords_group.create_dataset('grid_x', data=unique_coords['grid_x'].values, compression='gzip')
            coords_group.create_dataset('latitude', data=unique_coords['latitude'].values, compression='gzip')
            coords_group.create_dataset('longitude', data=unique_coords['longitude'].values, compression='gzip')

            # Store main data
            data_group = f.create_group('data')

            # Get all variable columns (exclude coordinate and date columns)
            coord_cols = {"grid_y", "grid_x", "latitude", "longitude", "day"}
            variable_cols = [col for col in df.columns if col not in coord_cols]

            # Store each variable as a separate dataset
            for var in variable_cols:
                var_data = df_pandas[var].values
                data_group.create_dataset(var, data=var_data, compression='gzip', compression_opts=6)

            # Store time information
            time_group = f.create_group('time')
            days = df_pandas['day'].values.astype('S10')  # Convert to bytes for HDF5
            time_group.create_dataset('day', data=days, compression='gzip')

            # Store grid indices for reconstruction
            grid_group = f.create_group('grid_indices')
            grid_group.create_dataset('grid_y', data=df_pandas['grid_y'].values, compression='gzip')
            grid_group.create_dataset('grid_x', data=df_pandas['grid_x'].values, compression='gzip')

    @staticmethod
    def consolidate_daily_outputs_partitioned(experiment_dir: Path,
                                            output_format: str = "parquet",
                                            partition_by: str = "time") -> None:
        """
        Consolidate daily outputs with smart partitioning for better query performance.

        Args:
            experiment_dir (Path): Directory containing gridcell folders with CSV files
            output_format (str): Output format - "parquet" only for partitioned output
            partition_by (str): Partition strategy - "time" or "space"
        """
        if output_format.lower() != "parquet":
            print("Partitioned output only supported for Parquet format")
            return

        print(f"Consolidating daily outputs with {partition_by} partitioning")

        # Find all CSV files
        csv_files = list(experiment_dir.rglob("grd_*.csv"))

        if not csv_files:
            print(f"No CSV files found in {experiment_dir}")
            return

        print(f"Found {len(csv_files)} CSV files")

        # Process all files
        n_workers = min(os.cpu_count() or 4, 8)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            all_dfs = list(executor.map(table_data._process_single_csv_for_consolidation, csv_files))

        # Filter valid DataFrames
        valid_dfs = [df for df in all_dfs if len(df) > 0]

        if not valid_dfs:
            print("No valid data found")
            return

        # Combine all data
        final_df = pl.concat(valid_dfs, rechunk=True)

        # Add partitioning columns
        if partition_by == "time":
            final_df = final_df.with_columns([
                pl.col("day").str.slice(0, 4).alias("year"),
                pl.col("day").str.slice(5, 2).alias("month")
            ])
            partition_cols = ["year", "month"]
        elif partition_by == "space":
            # Partition by latitude bands
            final_df = final_df.with_columns([
                (pl.col("latitude") // 10 * 10).alias("lat_band")
            ])
            partition_cols = ["lat_band"]
        else:
            raise ValueError(f"Unsupported partition strategy: {partition_by}")

        # Write partitioned dataset
        output_dir = experiment_dir / f"{experiment_dir.name}_partitioned"
        output_dir.mkdir(exist_ok=True)

        print(f"Writing partitioned dataset to {output_dir}")

        # Use PyArrow for partitioned writing
        try:
            import pyarrow.parquet as pq

            # Convert to PyArrow table
            table = final_df.to_arrow()

            # Write partitioned dataset
            pq.write_to_dataset(
                table,
                root_path=str(output_dir),
                partition_cols=partition_cols,
                compression='snappy',
                use_legacy_dataset=False
            )

            print(f"Successfully created partitioned dataset with {len(csv_files)} files")

        except ImportError:
            print("PyArrow not available for partitioned writing. Using standard Parquet.")
            final_df.write_parquet(
                output_dir / "output.parquet",
                compression="snappy"
            )

    @staticmethod
    def consolidate_annual_biomass(experiment_dir: Path,
                                 output_format: str = "parquet",
                                 chunk_size: int = 100,
                                 name_aux:str = "") -> None:
        """
        Consolidate annual biomass CSV outputs from multiple gridcells into a single file.

        Args:
            experiment_dir (Path): Directory containing gridcell folders with biomass CSV files
            output_format (str): Output format - "parquet", "feather", "hdf5", or "csv"
            chunk_size (int): Number of CSV files to process in each chunk
        """
        print(f"Consolidating annual biomass outputs in {experiment_dir.name}")

        # Find all biomass CSV files with the pattern metacomunity_biomass_*.csv
        biomass_files = list(experiment_dir.rglob("metacomunity_biomass_*.csv"))

        if not biomass_files:
            print(f"No biomass CSV files found in {experiment_dir}")
            return

        print(f"Found {len(biomass_files)} biomass CSV files")

        # Process files in chunks to manage memory
        n_workers = min(os.cpu_count() or 4, 8)
        all_dfs = []

        for i in range(0, len(biomass_files), chunk_size):
            chunk = biomass_files[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(biomass_files)-1)//chunk_size + 1}")

            # Process chunk in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                chunk_dfs = list(executor.map(table_data._process_single_biomass_csv, chunk))

            # Filter out empty DataFrames and concatenate
            valid_dfs = [df for df in chunk_dfs if len(df) > 0]
            if valid_dfs:
                chunk_combined = pl.concat(valid_dfs, rechunk=True)
                all_dfs.append(chunk_combined)

            # Force garbage collection between chunks
            gc.collect()

        # Combine all chunks
        if not all_dfs:
            print(f"No valid biomass data found in {experiment_dir}")
            return

        print("Combining all chunks...")
        final_df = pl.concat(all_dfs, rechunk=True)

        # Sort by coordinates and year for better organization
        final_df = final_df.sort(["grid_y", "grid_x", "year", "pls_id"])

        # Write consolidated file based on format
        output_file = experiment_dir.parent / f"{experiment_dir.name}_biomass{name_aux}"

        if output_format.lower() == "parquet":
            final_df.write_parquet(
                output_file.with_suffix('.parquet'),
                compression="snappy"
            )
        elif output_format.lower() == "feather":
            final_df.write_ipc(
                output_file.with_suffix('.feather'),
                compression="zstd"
            )
        elif output_format.lower() == "hdf5":
            table_data._write_hdf5_biomass_consolidated(final_df, output_file.with_suffix('.h5'))
        elif output_format.lower() == "csv":
            final_df.write_csv(output_file.with_suffix('.csv'))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"Successfully consolidated {len(biomass_files)} biomass files into {output_file}.{output_format.lower()}")


    @staticmethod
    def consolidate_annual_biomass_new(experiment_dir: Path,
                                output_format: str = "parquet",
                                chunk_size: int = 100,
                                name_aux: str = "") -> None:
        """
        Consolidate annual biomass CSV outputs from multiple gridcells into a single file.
        """
        print(f"Consolidating annual biomass outputs in {experiment_dir.name}")

        # Find all biomass CSV files
        biomass_files = list(experiment_dir.rglob("metacomunity_biomass_*.csv"))

        if not biomass_files:
            print(f"No biomass CSV files found in {experiment_dir}")
            return

        print(f"Found {len(biomass_files)} biomass CSV files")

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid conflicts
        # Limit workers to prevent resource exhaustion
        max_workers = min(os.cpu_count() or 4, 16)  # Cap at 16 workers
        all_dfs = []

        for i in range(0, len(biomass_files), chunk_size):
            chunk = biomass_files[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(biomass_files)-1)//chunk_size + 1}")

            # Use ThreadPoolExecutor instead of ProcessPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_dfs = list(executor.map(table_data._process_single_biomass_csv, chunk))

            # Filter out empty DataFrames and concatenate
            valid_dfs = [df for df in chunk_dfs if len(df) > 0]
            if valid_dfs:
                chunk_combined = pl.concat(valid_dfs, rechunk=True)
                all_dfs.append(chunk_combined)

            # Force garbage collection between chunks
            gc.collect()

        # Rest of the method remains the same...
        if not all_dfs:
            print(f"No valid biomass data found in {experiment_dir}")
            return

        print("Combining all chunks...")
        final_df = pl.concat(all_dfs, rechunk=True)
        final_df = final_df.sort(["grid_y", "grid_x", "year", "pls_id"])

        output_file = experiment_dir.parent / f"{experiment_dir.name}_biomass{name_aux}"

        if output_format.lower() == "parquet":
            final_df.write_parquet(
                output_file.with_suffix('.parquet'),
                compression="snappy"
            )
        elif output_format.lower() == "feather":
            final_df.write_ipc(
                output_file.with_suffix('.feather'),
                compression="zstd"
            )
        elif output_format.lower() == "hdf5":
            table_data._write_hdf5_biomass_consolidated(final_df, output_file.with_suffix('.h5'))
        elif output_format.lower() == "csv":
            final_df.write_csv(output_file.with_suffix('.csv'))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"Successfully consolidated {len(biomass_files)} biomass files into {output_file}.{output_format.lower()}")



    @staticmethod
    def _extract_coordinates_from_biomass_filename(csv_path: Path) -> Tuple[int, int]:
        """Extract y, x coordinates from metacomunity_biomass_y-x.csv filename"""
        parts = csv_path.stem.split('_')
        if len(parts) >= 3:
            try:
                # Handle format like metacomunity_biomass_182-263.csv
                coords = parts[-1].split('-')
                if len(coords) == 2:
                    y, x = int(coords[0]), int(coords[1])
                    return y, x
            except (ValueError, IndexError):
                pass
        return -1, -1

    @staticmethod
    def _process_single_biomass_csv(csv_path: Path) -> pl.DataFrame:
        """Process a single biomass CSV file and add coordinate columns"""
        try:
            # Read CSV with polars
            df = pl.read_csv(csv_path)

            # Extract coordinates from filename
            y, x = table_data._extract_coordinates_from_biomass_filename(csv_path)

            if y == -1 or x == -1:
                print(f"Warning: Could not extract coordinates from {csv_path.name}")
                return pl.DataFrame()

            # Convert indices to lat/lon
            lat, lon = table_data._indices_to_latlon(y, x)

            # Add coordinate columns
            df = df.with_columns([
                pl.lit(y).alias("grid_y"),
                pl.lit(x).alias("grid_x"),
                pl.lit(lat).alias("latitude"),
                pl.lit(lon).alias("longitude")
            ])

            return df

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return pl.DataFrame()

    @staticmethod
    def _write_hdf5_biomass_consolidated(df: pl.DataFrame, output_path: Path) -> None:
        """Write consolidated biomass data to HDF5 format"""
        print(f"Writing HDF5 biomass file: {output_path}")

        try:
            import h5py
        except ImportError:
            print("h5py not available. Please install it with: pip install h5py")
            return

        # Convert to pandas for HDF5 compatibility
        df_pandas = df.to_pandas()

        with h5py.File(output_path, 'w') as f:
            # Create metadata group
            meta_group = f.create_group('metadata')
            meta_group.attrs['description'] = 'Consolidated annual biomass outputs from CAETE-DVM'
            meta_group.attrs['data_type'] = 'annual_biomass'

            from datetime import datetime
            meta_group.attrs['creation_date'] = str(datetime.now())
            meta_group.attrs['total_gridcells'] = len(df.select("grid_y", "grid_x").unique())
            meta_group.attrs['year_range'] = f"{df['year'].min()} to {df['year'].max()}"
            meta_group.attrs['variables'] = ['pls_id', 'vp_cleaf', 'vp_croot', 'vp_cwood', 'count', 'cveg', 'ocp']

            # Store coordinate information
            coords_group = f.create_group('coordinates')
            unique_coords = df.select("grid_y", "grid_x", "latitude", "longitude").unique().to_pandas()
            coords_group.create_dataset('grid_y', data=unique_coords['grid_y'].values, compression='gzip')
            coords_group.create_dataset('grid_x', data=unique_coords['grid_x'].values, compression='gzip')
            coords_group.create_dataset('latitude', data=unique_coords['latitude'].values, compression='gzip')
            coords_group.create_dataset('longitude', data=unique_coords['longitude'].values, compression='gzip')

            # Store main biomass data
            biomass_group = f.create_group('biomass')
            biomass_vars = ['pls_id', 'vp_cleaf', 'vp_croot', 'vp_cwood', 'count', 'cveg', 'ocp']

            for var in biomass_vars:
                var_data = df_pandas[var].values
                biomass_group.create_dataset(var, data=var_data, compression='gzip', compression_opts=6)

            # Store temporal information
            time_group = f.create_group('time')
            time_group.create_dataset('year', data=df_pandas['year'].values, compression='gzip')

            # Store spatial indices for reconstruction
            spatial_group = f.create_group('spatial')
            spatial_group.create_dataset('grid_y', data=df_pandas['grid_y'].values, compression='gzip')
            spatial_group.create_dataset('grid_x', data=df_pandas['grid_x'].values, compression='gzip')

            print(f"Successfully wrote HDF5 biomass file with {len(df)} records")

    @staticmethod
    def consolidate_all_annual_outputs(experiment_dir: Path,
                                     output_types: List[str] = None,
                                     output_format: str = "parquet",
                                     name_aux:str = "") -> None:
        """
        Consolidate all types of annual outputs (biomass, productivity, etc.)

        Args:
            experiment_dir (Path): Directory containing gridcell folders
            output_types (List[str]): Types of outputs to consolidate (default: ['biomass'])
            output_format (str): Output format for consolidated files
        """
        if output_types is None:
            output_types = ['biomass']

        print(f"Consolidating annual outputs for {experiment_dir.name}")

        for output_type in output_types:
            if output_type == 'biomass':
                table_data.consolidate_annual_biomass_new(experiment_dir, output_format, name_aux=name_aux)
            else:
                print(f"Output type '{output_type}' not yet implemented")

#==============================================
# Output Manager Class
#==============================================
class output_manager:
    """
    Manager class for organizing and processing different types of model outputs.
    This class coordinates both daily and annual output processing workflows.
    """

    @staticmethod
    def cities_output():
        """
        Process and save outputs for predefined city scenarios.

        This method processes outputs for historical, ssp370, and ssp585 scenarios
        and saves the results as CSV files and the consolidated outputs to feather format.

        Returns:
            None
        """
        results = (config.output.output_dir / Path("./cities_MPI-ESM1-2-HR_hist_output.psz"),
                   config.output.output_dir / Path("./cities_MPI-ESM1-2-HR-ssp370_output.psz"),
                   config.output.output_dir / Path("./cities_MPI-ESM1-2-HR-ssp585_output.psz"),
                   config.output.output_dir / Path("./cities_MPI-ESM1-2-HR-piControl_output.psz"))

        variables = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp",
                    "photo", "npp", "evapm", "lai", "f5", "wsoil",
                    "pupt", "nupt", "ls", "c_cost", "rcm", "storage_pool","inorg_n",
                    "inorg_p", "snc", "vcmax", 'specific_la',"cdef")

        # Determine optimal number of processes based on system resources
        available_cpus = os.cpu_count() or 4
        max_jobs = min(len(results), available_cpus)

        # Process with progress reporting
        print(f"Processing {len(results)} scenario outputs using {max_jobs} parallel jobs")
        Parallel(n_jobs=max_jobs, backend="loky", verbose=3)(
            delayed(table_data.table_output_per_grd)(r, variables) for r in results
        )
        print("All scenarios processed successfully")

        print("Consolidating daily outputs for cities scenarios...")
        experiments = [config.output.output_dir / "cities_MPI-ESM1-2-HR_hist",
                       config.output.output_dir / "cities_MPI-ESM1-2-HR-ssp370",
                       config.output.output_dir / "cities_MPI-ESM1-2-HR-ssp585",
                       config.output.output_dir / "cities_MPI-ESM1-2-HR-piControl"]

        for experiment_dir in experiments:
            res = Path(experiment_dir).resolve()
            if not res.exists():
                print(f"Warning: Experiment directory {res} does not exist")
                continue

            table_data.consolidate_daily_outputs(
                res,
                output_format="feather",
                chunk_size=500
            )

            table_data.consolidate_all_annual_outputs(
                res,
                output_types=['biomass'],
                output_format="feather"
            )

        return None


    @staticmethod
    def test_output():
        """
        Process and save outputs for predefined city scenarios.

        This method processes outputs for historical, ssp370, and ssp585 scenarios
        and saves the results as CSV files and the consolidated outputs to feather format.

        Returns:
            None
        """
        results = (Path(config.output.output_dir / "pan_amazon_hist_result.psz"),)

        variables = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp",
                    "photo", "npp", "evapm", "lai", "f5", "wsoil",
                    "pupt", "nupt", "ls", "c_cost", "rcm", "storage_pool","inorg_n",
                    "inorg_p", "snc", "vcmax", 'specific_la',"cdef")

        # Determine optimal number of processes based on system resources
        available_cpus = os.cpu_count() or 4
        max_jobs = min(len(results), available_cpus)

        # Process with progress reporting
        print(f"Processing {len(results)} scenario outputs using {max_jobs} parallel jobs")
        Parallel(n_jobs=max_jobs, backend="loky", verbose=3)(
            delayed(table_data.table_output_per_grd)(r, variables) for r in results
        )
        print("All scenarios processed successfully")

        print("Consolidating daily outputs for cities scenarios...")
        experiments = [config.output.output_dir / "pan_amazon_hist"]

        for experiment_dir in experiments:
            res = Path(experiment_dir).resolve()
            if not res.exists():
                print(f"Warning: Experiment directory {res} does not exist")
                continue

            table_data.consolidate_daily_outputs(
                res,
                output_format="feather",
                chunk_size=500
            )

            table_data.consolidate_all_annual_outputs(
                res,
                output_types=['biomass'],
                output_format="feather"
            )

        return None


    @staticmethod
    def table_outputs(filename:Union[Path, str], year:None | int = None):
        """
        Process biomass files and output them as a parquet table.

        Args:
            result (Union[Path, str]): Path to the state file with model results.
        
        Raises:
            ValueError: If the year argument is neither an integer nor None.
        """
        results = get_args(filename)
        available_cpus = 64

        #TODO: Add checks for year validity and file existence

        # Define grid cell processing function
        def process_gridcell(grd, year_arg=None):
            table_data.write_metacomm_output(grd, year_arg)

        # Use joblib's Parallel for efficient multiprocessing
        # Set verbose=1 to show progress during longer operations

        for fname in results:
            reg = worker.load_state_zstd(fname)
            nprocs = min(len(reg), available_cpus)
            if year is not None:
                Parallel(n_jobs=nprocs, verbose=3, prefer="threads")(delayed(process_gridcell)(grd, year) for grd in reg)
            elif year is None:
                Parallel(n_jobs=nprocs, verbose=3, prefer="threads")(delayed(process_gridcell)(grd) for grd in reg)
            else:
                raise ValueError("Year argument must be an integer or None")

            res = reg.output_path
            table_data.consolidate_all_annual_outputs(
                res,
                output_types=['biomass'],
                output_format="parquet",
                name_aux=f"_{year}" if year is not None else ""
            )


    @staticmethod
    def table_outputs_new(filename: Union[Path, str], year: None | int = None):
        """Process biomass files and output them as a parquet table."""
        results = get_args(filename)
        # Limit workers to prevent conflicts
        available_cpus = min(os.cpu_count() or 4, 32)  # Cap at 32 workers

        def process_gridcell(grd, year_arg=None):
            table_data.write_metacomm_output(grd, year_arg)

        for fname in results:
            reg = worker.load_state_zstd(fname)
            nprocs = min(len(reg), available_cpus)
            
            if year is not None:
                # Use "threads" backend instead of "processes" for better compatibility
                Parallel(n_jobs=nprocs, verbose=1, prefer="threads")(
                    delayed(process_gridcell)(grd, year) for grd in reg
                )
            elif year is None:
                Parallel(n_jobs=nprocs, verbose=1, prefer="threads")(
                    delayed(process_gridcell)(grd) for grd in reg
                )
            else:
                raise ValueError("Year argument must be an integer or None")

            res = reg.output_path
            table_data.consolidate_all_annual_outputs(
                res,
                output_types=['biomass'],
                output_format="parquet",
                name_aux=f"_{year}" if year is not None else ""
            )


    @staticmethod
    def pan_amazon_output():
        """Function to process Pan-Amazon historical output and save as netCDF daily files and parquet biomass files (per year)."""
        
        # Load region result file
        output_file = Path("../outputs/pan_amazon_hist_result.psz")
        reg:region = worker.load_state_zstd(output_file)

        # Select the variables to be written
        # Daily outputs
        variables_to_read = ("npp", "rnpp", "photo", "evapm", "wsoil", "csoil", "hresp", "aresp", "lai")
        
        # Years to output biomass tables
        years_to_output = [1901, 1961, 1971, 1981, 1991, 2001, 2011, 2021, 2024]

        ## NetCDF daily outputs
        from time import perf_counter
        start = perf_counter()
        a = gridded_data.create_masked_arrays(gridded_data.aggregate_region_data(reg, variables_to_read, (3,5)))
        gridded_data.save_netcdf_daily(a, "pan_amazon_hist_da")
        end = perf_counter()
        print(f"Elapsed time: {end - start:.2f} seconds")

        # Biomass outputs per year
        for year in years_to_output:
            output_manager.table_outputs_new(output_file, year=year)


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    sys.exit(1)
    pass

    # Some old code for ddevelopment and profiling features
    # """
    # Usage examples:

    # # Basic daily output consolidation
    # python dataframes.py --consolidate /path/to/experiment --format parquet

    # # Consolidate with partitioning
    # python dataframes.py --consolidate /path/to/experiment --format parquet --partition time

    # # Profile cities output
    # python dataframes.py --profile cities

    # # Dedicated profiling script
    # python profile_dataframes.py --test write_daily_data --size medium --visualize

    # Using the decorator:

    # from dataframes import profile_function

    # @profile_function("my_optimization_test")
    # def my_test_function():
    #     # Your code here
    #     pass

    # """

    # # Define profiling options with argparse
    # parser = argparse.ArgumentParser(description='Run CAETE-DVM with profiling options')
    # parser.add_argument('--profile', choices=['none', 'cities', 'table', 'both'],
    #                   default='none', help='Profiling mode')
    # parser.add_argument('--method', choices=['table_output', 'metacomm_output', 'make_daily_df'],
    #                   default=None, help='Specific method to profile')
    # parser.add_argument('--trace-memory', action='store_true',
    #                   help='Enable memory tracing')

    # # Add consolidation options
    # parser.add_argument('--consolidate', type=str, metavar='EXPERIMENT_DIR',
    #                   help='Path to experiment directory to consolidate daily outputs')
    # parser.add_argument('--consolidate-annual', type=str, metavar='EXPERIMENT_DIR',
    #                   help='Path to experiment directory to consolidate annual outputs')
    # parser.add_argument('--format', choices=['parquet', 'feather', 'hdf5', 'csv'],
    #                   default='parquet', help='Output format for consolidated data')
    # parser.add_argument('--partition', choices=['time', 'space'],
    #                   help='Partitioning strategy (parquet only)')
    # parser.add_argument('--chunk-size', type=int, default=500,
    #                   help='Number of CSV files to process per chunk')
    # parser.add_argument('--annual-types', nargs='+', choices=['biomass'],
    #                   default=['biomass'], help='Types of annual outputs to consolidate')

    # # Parse command line arguments when run directly
    # if Path(__file__).name in sys.argv[0]:
    #     args = parser.parse_args()
    # else:
    #     # Default values for module import
    #     class DefaultArgs:
    #         def __init__(self):
    #             self.profile = 'none'
    #             self.method = None
    #             self.trace_memory = False
    #             self.consolidate = None
    #             self.format = 'parquet'
    #             self.partition = None
    #             self.chunk_size = 500
    #     args = DefaultArgs()

    # # Handle consolidation requests
    # if args.consolidate:
    #     experiment_dir = Path(args.consolidate)
    #     if not experiment_dir.exists():
    #         print(f"Error: Experiment directory {experiment_dir} does not exist")
    #         sys.exit(1)

    #     print(f"Consolidating daily outputs from {experiment_dir}")

    #     if args.partition:
    #         # Use partitioned consolidation
    #         table_data.consolidate_daily_outputs_partitioned(
    #             experiment_dir,
    #             output_format=args.format,
    #             partition_by=args.partition
    #         )
    #     else:
    #         # Use regular consolidation
    #         table_data.consolidate_daily_outputs(
    #             experiment_dir,
    #             output_format=args.format,
    #             chunk_size=args.chunk_size
    #         )

    #     print("Consolidation completed!")
    #     sys.exit(0)

    # # Handle annual consolidation requests
    # if args.consolidate_annual:
    #     experiment_dir = Path(args.consolidate_annual)
    #     if not experiment_dir.exists():
    #         print(f"Error: Experiment directory {experiment_dir} does not exist")
    #         sys.exit(1)

    #     print(f"Consolidating annual outputs from {experiment_dir}")
    #     table_data.consolidate_all_annual_outputs(
    #         experiment_dir,
    #         output_types=args.annual_types,
    #         output_format=args.format
    #     )

    #     print("Annual consolidation completed!")
    #     sys.exit(0)

    # # Profiling imports and setup
    # from profiling import proftools
    # ProfilerManager = proftools.ProfilerManager
    # profile_function = proftools.profile_function

    # # Profiling test cases
    # def profile_cities_output():
    #     """Profile the cities_output method"""
    #     profiler = ProfilerManager("cities_output_profile")
    #     profiler.start()
    #     output_manager.cities_output()
    #     profiler.stop()

    # def profile_table_data_method(method_name, *args, **kwargs):
    #     """Profile a specific method in table_data class"""
    #     method = getattr(table_data, method_name, None)
    #     if method is None:
    #         print(f"Method {method_name} not found in table_data class")
    #         return

    #     decorated = profile_function(f"table_data_{method_name}_profile")(method)
    #     return decorated(*args, **kwargs)

    # # Execute based on arguments

    # ## We dont have profiling for the gridded data tools yet
    # if args.profile == 'none':
    #     # Regular execution without profiling
    #     output_file = Path("./pan_amazon_hist_result.psz")
    #     reg:region = worker.load_state_zstd(output_file)
    #     variables_to_read = ("npp","photo")
    #     a = gridded_data.create_masked_arrays(gridded_data.aggregate_region_data(reg, variables_to_read, (1,2)))

    # elif args.profile == 'cities':
    #     # Pro        from pathlib import Path
    #     from dataframes import table_data

    #     # Consolidate daily outputs from an experiment
    #     experiment_dir = Path("outputs/cities_MPI-ESM1-2-HR_hist")
    #     table_data.consolidate_daily_outputs(
    #         experiment_dir,
    #         output_format="parquet",
    #         chunk_size=500
    #     )
    #     profile_cities_output()

    # elif args.profile == 'table' and args.method:
    #     # Profile specific table_data method
    #     if args.method == 'table_output':
    #         results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    #         variables = ("cue", "wue", "csoil", "hresp")
    #         profile_table_data_method('write_daily_data',
    #                                  worker.load_state_zstd(results), variables)

    #     elif args.method == 'metacomm_output':
    #         results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    #         reg = worker.load_state_zstd(results)
    #         profile_table_data_method('write_metacomm_output', reg[0])

    #     elif args.method == 'make_daily_df':
    #         results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    #         reg = worker.load_state_zstd(results)
    #         variables = ("cue", "wue", "csoil", "hresp")
    #         profile_table_data_method('make_daily_dataframe', reg, variables, None)

    # elif args.profile == 'both':
    #     # Profile both cities_output and table_data
    #     print("Profiling cities_output...")
    #     profile_cities_output()

    #     print("\nProfilering table_data.write_daily_data...")
    #     results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    #     reg = worker.load_state_zstd(results)
    #     variables = ("cue", "wue", "csoil", "hresp")
    #     profile_table_data_method('write_daily_data', reg, variables)

    # print("\nDone.")


##SANDBOX - Old code that was excluded but might be useful some day. Probably not.
## Old version of the aggregate_region_data with threading. It was faster but memory intensive.

    # @staticmethod
    # def aggregate_region_data(r: region,
    #             variables: Union[str, Collection[str]],
    #             spin_slice: Union[int, Tuple[int, int], None] = None
    #             )-> Dict[str, NDArray]:
    #     """_summary_

    #     Args:
    #         r (region): a region object

    #         variables (Union[str, Collection[str]]): variable names to read

    #         spin_slice (Union[int, Tuple[int, int], None], optional): which spin slice to read.
    #         Defaults to None, read all available data. Consumes a lot of memory.

    #     Returns:
    #         dict: a dict with the following keys: time, coord, data holding data to be transformed
    #         necessary to create masked arrays and subsequent netCDF files.
    #     """

    #     output = []
    #     # nproc = min(len(r), 56)
    #     nproc = config.multiprocessing.nprocs # type_ignore
    #     nproc = max(1, 56) # Ensure at least one thread is used
    #     with ThreadPoolExecutor(max_workers=nproc) as executor:
    #         futures = [executor.submit(gridded_data.read_grd, grd, variables, spin_slice) for grd in r]
    #         for future in futures:
    #             output.append(future.result())

    #     # Finalize the data object
    #     raw_data = np.array(output, dtype=object)
    #     # Reoeganize resources
    #     time = raw_data[:,0][0] # We assume all slices have the same time, thus we get the first one
    #     coord = raw_data[:,2:4][:].astype(np.int64) # 2D matrix of coordinates (y(lat), x(lon))}
    #     data = raw_data[:,1][:]  # array of dicts, each dict has the variables as keys and the time series as values

    #     if isinstance(variables, str):
    #         dim_names = ["time", "coord", variables]
    #     else:
    #         dim_names = ["time", "coord", "data"]

    #     return dict(zip(dim_names, (time, coord, data)))


    # Vibe-coded version with memory-mapping and parallel reading for large regions

    # @staticmethod
    # def aggregate_region_data(r: region,
    #                           variables: Union[str, Collection[str]],
    #                           spin_slice: Union[int, Tuple[int, int], None] = None,
    #                           temp_dir: Path = None,
    #                           batch_size: int = 58,
    #                           max_workers: int = None
    #                         ) -> Dict[str, NDArray]:
    #     """Memory-mapped version with parallel reading for large regions

    #     Args:
    #         r (region): Region object with gridcells
    #         variables: Variable names to read
    #         spin_slice: Which spin slice to read
    #         temp_dir: Directory for temporary memory-mapped files
    #         batch_size: Number of gridcells to process per batch
    #         max_workers: Maximum number of parallel workers

    #     Returns:
    #         Dictionary with aggregated data
    #     """
    #     import tempfile
    #     from concurrent.futures import ThreadPoolExecutor, as_completed

    #     if temp_dir is None:
    #         temp_dir = Path(tempfile.gettempdir()) / "caete_memmap"
    #     temp_dir.mkdir(exist_ok=True)

    #     n_gridcells = len(r)

    #     # Pre-allocate memory-mapped arrays for coordinates
    #     coord_memmap_file = temp_dir / f"coords_{os.getpid()}.dat"
    #     coord_memmap = np.memmap(
    #         coord_memmap_file,
    #         dtype=np.int64,
    #         mode='w+',
    #         shape=(n_gridcells, 2)
    #     )

    #     # Storage for data - will be filled in parallel
    #     all_data = [None] * n_gridcells
    #     time_data = None
    #     completed_count = 0

    #     def read_batch_parallel(batch_indices, batch_gridcells):
    #         """Read a batch of gridcells in parallel"""
    #         nonlocal time_data, completed_count

    #         batch_results = []
    #         batch_coords = []

    #         # Use ThreadPoolExecutor for this batch
    #         with ThreadPoolExecutor(max_workers=config.multipricessing.nprocs) as batch_executor:
    #             # Submit all gridcell reads for this batch
    #             future_to_idx = {}
    #             for local_idx, (global_idx, grd) in enumerate(zip(batch_indices, batch_gridcells)):
    #                 future = batch_executor.submit(gridded_data.read_grd, grd, variables, spin_slice)
    #                 future_to_idx[future] = (global_idx, local_idx, grd)

    #             # Collect results as they complete
    #             for future in as_completed(future_to_idx):
    #                 global_idx, local_idx, grd = future_to_idx[future]
    #                 try:
    #                     time, data, y, x = future.result()

    #                     # Store time data once
    #                     if time_data is None:
    #                         time_data = time

    #                     # Store results
    #                     batch_results.append((global_idx, data))
    #                     batch_coords.append((global_idx, y, x))

    #                 except Exception as e:
    #                     print(f"Error reading gridcell {grd.y}-{grd.x}: {e}")

    #         return batch_results, batch_coords

    #     # Process gridcells in batches with parallel reading
    #     # print(f"Processing {n_gridcells} gridcells in batches of {batch_size} with {max_workers} workers")

    #     for batch_start in range(0, n_gridcells, batch_size):
    #         batch_end = min(batch_start + batch_size, n_gridcells)
    #         batch_indices = list(range(batch_start, batch_end))
    #         batch_gridcells = [r[i] for i in batch_indices]

    #         # print(f"Processing batch {batch_start//batch_size + 1}/{(n_gridcells-1)//batch_size + 1}")

    #         # Read batch in parallel
    #         batch_results, batch_coords = read_batch_parallel(batch_indices, batch_gridcells)

    #         # Store results in memory-mapped arrays
    #         for global_idx, data in batch_results:
    #             all_data[global_idx] = data

    #         for global_idx, y, x in batch_coords:
    #             coord_memmap[global_idx] = [y, x]

    #         completed_count += len(batch_results)
    #         # print(f"Completed {completed_count}/{n_gridcells} gridcells")

    #         # Periodic garbage collection
    #         if batch_start % (batch_size * 4) == 0:
    #             gc.collect()

    #     # Convert memory-mapped coordinates to regular array
    #     coord_array = np.array(coord_memmap)

    #     # Filter out None values from failed reads
    #     valid_data = [data for data in all_data if data is not None]

    #     # Clean up memory-mapped files
    #     try:
    #         coord_memmap_file.unlink(missing_ok=True)
    #     except:
    #         pass

    #     # Clean up temp directory if empty
    #     try:
    #         temp_dir.rmdir()
    #     except:
    #         pass

        # # Return results
        # if isinstance(variables, str):
        #     return {
        #         "time": time_data,
        #         "coord": coord_array[:len(valid_data)],  # Trim to valid data size
        #         variables: valid_data
        #     }
        # else:
        #     return {
        #         "time": time_data,
        #         "coord": coord_array[:len(valid_data)],
        #         "data": valid_data
        #     }

    # @staticmethod
    # def aggregate_region_data_streaming(r: region,
    #                                 variables: Union[str, Collection[str]],
    #                                 spin_slice: Union[int, Tuple[int, int], None] = None,
    #                                 output_file: Path = None
    #                                 ) -> Path:
    #     """Stream directly to disk without accumulating in memory"""

    #     import h5py
    #     import tempfile

    #     if output_file is None:
    #         output_file = Path(tempfile.gettempdir()) / f"region_data_{os.getpid()}.h5"

    #     n_gridcells = len(r)

    #     # Get sample data for structure
    #     sample_data = gridded_data.read_grd(r[0], variables, 1)
    #     time_data = sample_data[0]
    #     sample_vars = sample_data[1]

    #     with h5py.File(output_file, 'w') as f:
    #         # Create datasets with known dimensions
    #         coord_dset = f.create_dataset('coordinates', (n_gridcells, 2), dtype=np.int64)
    #         time_dset = f.create_dataset('time', data=time_data)

    #         # Create datasets for each variable
    #         var_datasets = {}
    #         if isinstance(variables, str):
    #             var_list = [variables]
    #         else:
    #             var_list = list(variables)

    #         for var_name in var_list:
    #             var_data = sample_vars[var_name]
    #             if len(var_data.shape) == 1:
    #                 shape = (n_gridcells, var_data.shape[0])
    #             elif len(var_data.shape) == 2:
    #                 shape = (n_gridcells, var_data.shape[0], var_data.shape[1])

    #             var_datasets[var_name] = f.create_dataset(
    #                 var_name,
    #                 shape,
    #                 dtype=var_data.dtype,
    #                 compression='gzip',
    #                 compression_opts=6
    #             )

    #         # Process and stream data
    #         batch_size = 50  # Smaller batches for streaming

    #         for batch_start in range(0, n_gridcells, batch_size):
    #             batch_end = min(batch_start + batch_size, n_gridcells)
    #             batch_gridcells = r[batch_start:batch_end]

    #             # Process batch
    #             with ThreadPoolExecutor(max_workers=min(len(batch_gridcells), 4)) as executor:
    #                 futures = [executor.submit(gridded_data.read_grd, grd, variables, spin_slice)
    #                         for grd in batch_gridcells]

    #                 for i, future in enumerate(futures):
    #                     try:
    #                         time, data, y, x = future.result()
    #                         global_idx = batch_start + i

    #                         # Write directly to HDF5 (no memory accumulation)
    #                         coord_dset[global_idx] = [y, x]

    #                         for var_name in var_list:
    #                             var_datasets[var_name][global_idx] = data[var_name]

    #                     except Exception as e:
    #                         print(f"Error processing gridcell: {e}")

    #             # Force write to disk
    #             f.flush()

    #             # Garbage collection
    #             gc.collect()

    #     return output_file
