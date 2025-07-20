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
import os
import gc
import sys
from concurrent.futures import ProcessPoolExecutor
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
from _geos import pan_amazon_region, get_region, find_coordinates_xy
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
        time, coord, data, holding data to be transformed.

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
        nproc = max(1, 56) # Ensure at least one thread is used
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
                                 chunk_size: int = 100) -> None:
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
        output_file = experiment_dir / f"{experiment_dir.name}_biomass"

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
                                     output_format: str = "parquet") -> None:
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
                table_data.consolidate_annual_biomass(experiment_dir, output_format)
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
        results = (Path("./cities_MPI-ESM1-2-HR_hist_output.psz"),
                   Path("./cities_MPI-ESM1-2-HR-ssp370_output.psz"),
                   Path("./cities_MPI-ESM1-2-HR-ssp585_output.psz"),
                   Path("./cities_MPI-ESM1-2-HR-piControl_output.psz"))

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
            delayed(table_data.table_output_per_grd)(r, variables) for r in results
        )
        print("All scenarios processed successfully")

        print("Consolidating daily outputs for cities scenarios...")
        experiments = ["../outputs/cities_MPI-ESM1-2-HR_hist",
                       "../outputs/cities_MPI-ESM1-2-HR-ssp370",
                       "../outputs/cities_MPI-ESM1-2-HR-ssp585",
                       "../outputs/cities_MPI-ESM1-2-HR-piControl"]

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


if __name__ == "__main__":
    """
    Usage examples:

    # Basic daily output consolidation
    python dataframes.py --consolidate /path/to/experiment --format parquet

    # Consolidate with partitioning
    python dataframes.py --consolidate /path/to/experiment --format parquet --partition time

    # Profile cities output
    python dataframes.py --profile cities

    # Dedicated profiling script
    python profile_dataframes.py --test write_daily_data --size medium --visualize

    Using the decorator:

    from dataframes import profile_function

    @profile_function("my_optimization_test")
    def my_test_function():
        # Your code here
        pass

    """

    # Define profiling options with argparse
    parser = argparse.ArgumentParser(description='Run CAETE-DVM with profiling options')
    parser.add_argument('--profile', choices=['none', 'cities', 'table', 'both'],
                      default='none', help='Profiling mode')
    parser.add_argument('--method', choices=['table_output', 'metacomm_output', 'make_daily_df'],
                      default=None, help='Specific method to profile')
    parser.add_argument('--trace-memory', action='store_true',
                      help='Enable memory tracing')

    # Add consolidation options
    parser.add_argument('--consolidate', type=str, metavar='EXPERIMENT_DIR',
                      help='Path to experiment directory to consolidate daily outputs')
    parser.add_argument('--consolidate-annual', type=str, metavar='EXPERIMENT_DIR',
                      help='Path to experiment directory to consolidate annual outputs')
    parser.add_argument('--format', choices=['parquet', 'feather', 'hdf5', 'csv'],
                      default='parquet', help='Output format for consolidated data')
    parser.add_argument('--partition', choices=['time', 'space'],
                      help='Partitioning strategy (parquet only)')
    parser.add_argument('--chunk-size', type=int, default=500,
                      help='Number of CSV files to process per chunk')
    parser.add_argument('--annual-types', nargs='+', choices=['biomass'],
                      default=['biomass'], help='Types of annual outputs to consolidate')

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
                self.consolidate = None
                self.format = 'parquet'
                self.partition = None
                self.chunk_size = 500
        args = DefaultArgs()

    # Handle consolidation requests
    if args.consolidate:
        experiment_dir = Path(args.consolidate)
        if not experiment_dir.exists():
            print(f"Error: Experiment directory {experiment_dir} does not exist")
            sys.exit(1)

        print(f"Consolidating daily outputs from {experiment_dir}")

        if args.partition:
            # Use partitioned consolidation
            table_data.consolidate_daily_outputs_partitioned(
                experiment_dir,
                output_format=args.format,
                partition_by=args.partition
            )
        else:
            # Use regular consolidation
            table_data.consolidate_daily_outputs(
                experiment_dir,
                output_format=args.format,
                chunk_size=args.chunk_size
            )

        print("Consolidation completed!")
        sys.exit(0)

    # Handle annual consolidation requests
    if args.consolidate_annual:
        experiment_dir = Path(args.consolidate_annual)
        if not experiment_dir.exists():
            print(f"Error: Experiment directory {experiment_dir} does not exist")
            sys.exit(1)

        print(f"Consolidating annual outputs from {experiment_dir}")
        table_data.consolidate_all_annual_outputs(
            experiment_dir,
            output_types=args.annual_types,
            output_format=args.format
        )

        print("Annual consolidation completed!")
        sys.exit(0)

    # Profiling imports and setup
    from profiling import proftools
    ProfilerManager = proftools.ProfilerManager
    profile_function = proftools.profile_function

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
        # Pro        from pathlib import Path
        from dataframes import table_data

        # Consolidate daily outputs from an experiment
        experiment_dir = Path("outputs/cities_MPI-ESM1-2-HR_hist")
        table_data.consolidate_daily_outputs(
            experiment_dir,
            output_format="parquet",
            chunk_size=500
        )
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
