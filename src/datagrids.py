import os
import gc
import sys
from pathlib import Path
from typing import Union, Collection, Tuple, Dict, List

import numpy as np
from numpy.typing import NDArray

from config import fetch_config
from _geos import pan_amazon_region, get_region
if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import grd_mt, get_args
from region import region

#TODO: implement region configuration
if pan_amazon_region is None:
    raise ValueError("pan_amazon_region is not defined or imported correctly")

# Get the region of interest
ymin, ymax, xmin, xmax = get_region(pan_amazon_region)

config = fetch_config()


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
                              batch_size: int = 58,
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
