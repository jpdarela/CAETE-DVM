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

from concurrent.futures import ThreadPoolExecutor

import copy
# import gc
import multiprocessing as mp
import os
from shutil import copy2
import sys

from pathlib import Path
from typing import Callable, Dict, List,Tuple, Union, Optional
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray
from joblib import dump, load
from joblib import Parallel, delayed

from config import Config, fetch_config
from parameters import hsoil, ssoil, tsoil
import metacommunity as mc

if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import str_or_path, get_co2_concentration, read_bz2_file, print_progress, grd_mt

# Standard output path for the model
# from parameters import output_path
from input_handler import input_handler

class region:
    """Region class containing the gridcells for a given region and methods to manage them.
    """


    def __init__(self,
                name:str,
                clim_data:Union[str,Path],
                soil_data:Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
                co2:Union[str, Path],
                pls_table:NDArray,
                gridlist:Optional[pl.DataFrame] = None)->None:
        """ Region class constructor

        Args:
            name (str): this will be the name of the region and the name of the output folder
            clim_data (Union[str,Path]): Path for the climate data
            soil_data (Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]]): _description_
            output_folder (Union[str, Path]): _description_
            co2 (Union[str, Path]): _description_
            pls_table (np.ndarray): _description_
        """
        self.config: Config = fetch_config()
        self.compression = tuple(self.config.compression.model_dump().values()) # type: ignore
        self.gridlist = gridlist
        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # IO
        self.climate_files = [] # Climate files for the region. Used only during the set_gridcells method.
        self.input_data = str_or_path(clim_data)
        self.input_type = self.config.input_handler.input_type
        self.input_method = self.config.input_handler.input_method
        self.nchunks = self.config.multiprocessing.max_processes

        if self.input_method == "ih":
            if gridlist is None:
                raise ValueError("Gridlist must be provided when using input_handler")
            self.input_handler = input_handler(self.input_data,
                                               self.gridlist,
                                               batch_size=self.nchunks)
        else:
            self.input_handler = None

        self.soil_data = copy.deepcopy(soil_data)
        self.pls_table = mc.pls_table(pls_table)
        self.file_objects = []

        # calculate_matrix size from grid resolution
        self.nx = len(np.arange(0, 180, self.config.crs.xres/2)) # type: ignore
        self.ny = len(np.arange(0,  90, self.config.crs.yres/2)) # type: ignore

        self.epsg = f"EPSG:{self.config.crs.epsg_id}" # type: ignore
        self.datum = self.config.crs.datum # type: ignore

        # Grid mask of the region
        self.grid_mask = np.ones((self.ny, self.nx), dtype=bool)

        # Number of PLS in the main table (global table)
        self.npls_main_table = self.pls_table.npls

        # Read the metadata from the climate files
        if self.input_method == "legacy":
            # Legacy input method
            try:
                metadata_file = list(self.input_data.glob("METADATA.pbz2"))[0]
            except:
                raise FileNotFoundError("Metadata file not found in the input data folder")

            try:
                mtd = str_or_path(metadata_file, check_is_file=True)
            except:
                raise AssertionError("Metadata file path could not be resolved. Cannot proceed without metadata")

            # Read metadata from climate files
            self.metadata = read_bz2_file(mtd)
            self.stime = copy.deepcopy(self.metadata[0])

        elif self.config.input_handler.input_method == "ih":
            self.metadata = self.input_handler.get_metadata
            self.stime = copy.deepcopy(self.metadata[0])
        else:
            raise ValueError(f"Unknown input method: {self.config.input_handler.input_method}. "
                             "Please use 'legacy' or 'ih' as input method.")

        if self.config.input_handler.input_method == "legacy":
            for file_path in sorted(list(self.input_data.glob("input_data_*-*.pbz2"))):
                self.climate_files.append(file_path)
        elif self.config.input_handler.input_method == "ih" and self.input_handler.input_type == "bz2":
            self.climate_files = self.input_handler._handler.input_files
        elif self.config.input_handler.input_method == "ih" and self.input_handler.input_type == "netcdf":
            st_names = self.input_handler._handler.gridlist.get_column("station_name").to_list()
            self.climate_files = list(map(Path, st_names))

        self.region_size = len(self.climate_files)

        # Define grid structure
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
        # output_path is a global defined in parameters.py. The region object will
        # create the internal output folder structure into this directory
        os.makedirs(self.config.output.output_dir, exist_ok=True)

        # This is the output path for this region
        self.output_root = self.config.output.output_dir
        self.output_path = self.output_root / self.name

        # Path to the state files for this region
        self.state_path:Path | None = self.output_path / f"state_files_{uuid4().hex}"
        # Mapping of gridcells to their file objects
        self.gridcell_address_file_objects = {}

        # Create the output and state folders for this region
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.state_path, exist_ok=True)

        # A list to store this region's gridcells
        # Some magic methods are defined to deal with this list
        self.gridcells:List[grd_mt] = []


    def load_gridcell(self, idx:int):
        """Load a gridcell by its index from the intermediate files

        Args:
            idx (int): Index of the gridcell to load (zero-based index)

        Returns:
            grd_mt: The gridcell object
        """
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")
        if self.region_size <= 0:
            raise ValueError("Region size is not set. Cannot load gridcells")

        if idx < 0 or idx >= self.region_size:
            raise IndexError("Gridcell index out of range")

        # Get the file object and index for the gridcell
        k, f = self.gridcell_address_file_objects[idx]
        with open(f, 'rb') as fh:
            gridcells:List[grd_mt] = load(fh)

        return gridcells[k]


    def load_gridcells(self):
        """Load the gridcells from the intermediate files in parallel using threads, keeping order."""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

        if self.gridcells:
            return

        def load_file(f):
            try:
                with open(f, 'rb') as fh:
                    return load(fh)
            except Exception as e:
                print(f"Error loading file {f}: {e}")
                raise

        # Submit all jobs and keep their order
        with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
            results = list(executor.map(load_file, self.file_objects))
        # Flatten the list of lists, preserving order
        for gridcell_list in results:
            self.gridcells.extend(gridcell_list)


    def unload_gridcells(self):
        """Force the gridcells writing to the intermediate files"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")
        if not self.gridcells:
            raise ValueError("No gridcells found. Cannot save intermediate files")
        # Divide the gridcelss in self.gridcells into chunks and save them to the intermediate files
        chunks = [ self.gridcells[i:i + self.config.multiprocessing.max_processes] \
                  for i in range(0, len(self.gridcells), self.config.multiprocessing.max_processes)]
        for f, chunk in zip(self.file_objects, chunks):
            # Save the gridcells to the intermediate files
            with open(f, 'wb') as fh:
                dump(chunk, fh, compress=self.compression) # type: ignore
        self._build_gridcell_address_mapping()


    def set_new_state(self):
        # print("Creating a copy of the state files in a new folder")
        new_id = uuid4().hex
        new_state_folder = self.output_path / f"state_files_{new_id}"
        os.makedirs(new_state_folder, exist_ok=True)

        # Use ThreadPoolExecutor for I/O-bound operations like file copying
        with ThreadPoolExecutor(max_workers=min(64, len(self.file_objects))) as executor:
            # Create a list to store the new file paths
            new_file_objects = []

            # Create a mapping function that copies files and returns the new path
            def copy_file(f):
                dest_path = new_state_folder / f.name
                copy2(f, dest_path)
                return dest_path

            # Submit all copy tasks
            futures = [executor.submit(copy_file, f) for f in self.file_objects]

            # Handle exceptions if any
            for f, future in zip(self.file_objects, futures):
                try:
                    new_file_objects.append(future.result())  # Get the new file path
                except Exception as e:
                    print(f"Error copying file {f}: {e}")
                    raise e

        # Update file objects with the new paths
        self.file_objects = new_file_objects
        self.state_path = new_state_folder


    def delete_state(self):
        """Delete the state files of the region using parallel processing"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot delete state files")

        # Use ThreadPoolExecutor for I/O-bound operations like file deletion
        with ThreadPoolExecutor(max_workers=min(32, len(self.file_objects))) as executor:
            # Submit all delete tasks
            futures = [executor.submit(os.remove, f) for f in self.file_objects]

            # Handle exceptions if any
            for f, future in zip(self.file_objects, futures):
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
                    raise e

        self.file_objects = []
        os.rmdir(str(self.state_path))
        self.state_path = None


    def save_state(self, state_file:Union[str, Path], new_state = False):
        """Save the state of the region to a file

        Args:
            state_file (Union[str, Path]): Path to the file where the state will be saved
        """
        from worker import worker
        worker.save_state_zstd(self, self.output_root / state_file)
        if new_state:
            self.set_new_state()


    def update_dump_directory(self, new_name:str="copy"):
        """Update the output folder for the region

        Args:
            new_name (Union[str, Path]): name of the new folder where outputs of the region should be saved. Defaults to "copy"
        """
        self.name = Path(f"{new_name}")
        self.output_path = self.output_root / self.name # Update region output folder path

        os.makedirs(self.output_path, exist_ok=True)

        if not self.file_objects and not self.gridcells:
            raise ValueError("No file objects found. Cannot read intermediate files. No gridcells found. Cannot update output directory")

        if self.file_objects:
            def process_file(f):
                gridcells = load(f)
                for gridcell in gridcells:
                    # Update the output folder for each gridcell
                    gridcell.run_counter = 0
                    gridcell.outputs = {}
                    gridcell.metacomm_output = {}
                    gridcell.executed_iterations = []
                    gridcell.out_dir = self.output_path / Path(f"grd_{gridcell.xyname}")
                    os.makedirs(gridcell.out_dir, exist_ok=True)

                with open(f, 'wb') as fh:
                    dump(gridcells, fh, compress=self.compression) # type: ignore
                    fh.flush()  # Explicitly flush before closing
                # gc.collect()

            with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
                executor.map(process_file, self.file_objects)

        elif self.gridcells:
            for gridcell in self.gridcells:
                # Update the output folder for each gridcell
                gridcell.run_counter = 0
                gridcell.outputs = {}
                gridcell.metacomm_output = {}
                gridcell.executed_iterations = []
                gridcell.out_dir = self.output_path / Path(f"grd_{gridcell.xyname}")
                os.makedirs(gridcell.out_dir, exist_ok=True)


    def update_input(self, input_data=None, co2=None):
        """Update the input data for the region

        Args:
            input_data (str | Path, optional): Folder with input data or netCDF file to be used. Defaults to None.
            co2 (str | Path, optional): Text file (tsv/csv) with annual co2 concentration. Defaults to None.
            Attributes are updated if valid values are provided

        Raises:
            FileNotFoundError: _description_
            AssertionError: _description_
        """

        is_netcdf_input = self.input_type == "netcdf" and self.input_method == "ih"

        if co2 is not None:
            self.co2_path = str_or_path(co2)
            self.co2_data = get_co2_concentration(self.co2_path)

        if input_data is not None:
            # Read the climate data
            self.input_data = str_or_path(input_data)
            if self.input_method == "legacy":
                try:
                    metadata_file = list(self.input_data.glob("METADATA.pbz2"))[0]
                except:
                    raise FileNotFoundError("Metadata file not found in the input data folder")

                try:
                    mtd = str_or_path(metadata_file, check_is_file=True)
                except:
                    raise AssertionError("Metadata file path could not be resolved. Cannot proceed without metadata")

                # Read metadata from climate files
                self.metadata = read_bz2_file(mtd)
            else:
                self.input_handler = input_handler(self.input_data,
                                                    self.gridlist,
                                                    batch_size=self.nchunks)
                self.metadata = self.input_handler.get_metadata

            self.stime = copy.deepcopy(self.metadata[0])
        else:
            if co2 is None:
                raise ValueError("Input data must be provided to update the region input")

        if not self.file_objects and not self.gridcells:
            raise ValueError("No file objects found. Cannot read intermediate files. No gridcells found. Cannot update input data")
        if self.input_method == "ih":
            all_data = self.input_handler.get_data
        if is_netcdf_input:
            self.input_handler.close()
            self.input_handler = None
        # Update the input data for each gridcell
        if self.file_objects:
            def process_file(f):
                try:
                    with open(f, 'rb') as file:
                        gridcells:List[grd_mt] = load(file)
                    if co2 is not None:
                        for gridcell in gridcells:
                            if self.input_method == "ih":
                                cell_data = all_data[gridcell.station_name]
                                gridcell.change_input(cell_data, self.stime, self.co2_data)
                            else:
                                gridcell.change_input(self.input_data, self.stime, self.co2_data)
                    else:
                        for gridcell in gridcells:
                            if self.input_method == "ih":
                                cell_data = all_data[gridcell.station_name]
                                gridcell.change_input(cell_data, self.stime, None)
                            else:
                                gridcell.change_input(self.input_data, self.stime, None)
                    with open(f, 'wb') as file:
                        dump(gridcells, file, compress=self.compression) # type: ignore
                        file.flush()
                    # gc.collect()
                except Exception as e:
                    raise e


            with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
                # Submit all tasks
                futures = [executor.submit(process_file, f) for f in self.file_objects]

                # Process results and update progress
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        raise e

        elif self.gridcells:
            for gridcell in self.gridcells:
                if co2 is not None:
                    if self.input_method == "ih":
                        cell_data = all_data[gridcell.station_name]
                        gridcell.change_input(cell_data, self.stime, self.co2_data)
                    else:
                        gridcell.change_input(self.input_data, self.stime, self.co2_data)
                else:
                    if self.input_method == "ih":
                        cell_data = all_data[gridcell.station_name]
                        gridcell.change_input(cell_data, self.stime, None)
                    else:
                        gridcell.change_input(self.input_data, self.stime, None)


    def update_input_mp(self, input_data=None, co2=None):
        """Update the input data for the region

        Args:
            input_file (str | Path, optional): Folder with input data to be used. Defaults to None.
            co2 (str | Path, optional): Text file (tsv/csv) with annual co2 concentration. Defaults to None.
            Attributes are updated if valid values are provided

        Raises:
            FileNotFoundError: _description_
            AssertionError: _description_
        """

        is_netcdf_input = self.input_type == "netcdf" and self.input_method == "ih"

        if co2 is not None:
            self.co2_path = str_or_path(co2)
            self.co2_data = get_co2_concentration(self.co2_path)

        if input_data is not None:
            # Read the climate data
            self.input_data = str_or_path(input_data)
            if self.input_method == "legacy":
                try:
                    metadata_file = list(self.input_data.glob("METADATA.pbz2"))[0]
                except:
                    raise FileNotFoundError("Metadata file not found in the input data folder")

                try:
                    mtd = str_or_path(metadata_file, check_is_file=True)
                except:
                    raise AssertionError("Metadata file path could not be resolved. Cannot proceed without metadata")

                # Read metadata from climate files
                self.metadata = read_bz2_file(mtd)
            else:
                self.input_handler = input_handler(self.input_data,
                                                    self.gridlist,
                                                    batch_size=self.nchunks)
                self.metadata = self.input_handler.get_metadata

            self.stime = copy.deepcopy(self.metadata[0])
        else:
            if co2 is None:
                raise ValueError("Input data must be provided to update the region input")

        if not self.file_objects and not self.gridcells:
            raise ValueError("No file objects found. Cannot read intermediate files. No gridcells found. Cannot update input data")
        if self.input_method == "ih":
            all_data = self.input_handler.get_data
        if is_netcdf_input:
            self.input_handler.close()
            self.input_handler = None
         # Update the input data for each gridcell
        if self.file_objects:
            def process_file(f):
                try:
                    with open(f, 'rb') as file:
                        gridcells:List[grd_mt] = load(file)
                    if co2 is not None:
                        for gridcell in gridcells:
                            if self.input_method == "ih":
                                cell_data = all_data[gridcell.station_name]
                                gridcell.change_input(cell_data, self.stime, self.co2_data)
                            else:
                                gridcell.change_input(self.input_data, self.stime, self.co2_data)
                    else:
                        for gridcell in gridcells:
                            if self.input_method == "ih":
                                cell_data = all_data[gridcell.station_name]
                                gridcell.change_input(cell_data, self.stime, None)
                            else:
                                gridcell.change_input(self.input_data, self.stime, None)
                    with open(f, 'wb') as file:
                        dump(gridcells, file, compress=self.compression)
                        file.flush()
                    # gc.collect()
                except Exception as e:
                    raise e

            # Replace ThreadPoolExecutor with joblib.Parallel
            Parallel(n_jobs=min(len(self.file_objects), self.config.multiprocessing.max_processes),
                    backend='multiprocessing',
                    verbose=1)(delayed(process_file)(f) for f in self.file_objects)


        elif self.gridcells:
            for gridcell in self.gridcells:
                if co2 is not None:
                    if self.input_method == "ih":
                        cell_data = all_data[gridcell.station_name]
                        gridcell.change_input(cell_data, self.stime, self.co2_data)
                    else:
                        gridcell.change_input(self.input_data, self.stime, self.co2_data)
                else:
                    if self.input_method == "ih":
                        cell_data = all_data[gridcell.station_name]
                        gridcell.change_input(cell_data, self.stime, None)
                    else:
                        gridcell.change_input(self.input_data, self.stime, None)


    def get_from_main_table(self, comm_npls) -> Tuple[Union[int, NDArray[np.intp]], NDArray[np.float32]]:
        """Returns a number of IDs (in the main table) and the respective
        functional identities (PLS table) to set or reset a community

        This method is passed as an argument for the gridcell class. It is used to read
        the main table and the PLS table to set or reset a community.

        Args:
        comm_npls: (int) Number of PLS in the output table (must match npls_max (see caete.toml))"""

        assert comm_npls > 0, "Number of PLS must be greater than 0"

        if comm_npls == 1:
            idx = np.random.choice(self.npls_main_table, 1, replace=False)[0]
            return idx, self.pls_table.table[:, idx]

        assert comm_npls <= self.npls_main_table, "Number of PLS must be less than the number of PLS in the main table"

        idx = np.random.choice(self.npls_main_table, comm_npls, replace=False)
        return idx, self.pls_table.table[:, idx]


    def set_gridcells(self):
        """Method to set gridcells for a region. Only used in testing contexts.
        The constructions of gridcells for parallel runs is managed by the run_region_map method.
        """
        print("Starting gridcells")
        if input_handler is not None:
            all_data = self.input_handler.get_data
        i = 0
        print_progress(i, len(self.yx_indices), prefix='Progress:', suffix='Complete')
        for f,pos in zip(self.climate_files, self.yx_indices):
            y, x = pos
            gridcell_dump_directory = self.output_path/Path(f"grd_{y}-{x}") # The gridcell folder
            grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
            if self.input_method == "ih":
                grd_cell.set_gridcell(all_data[grd_cell.station_name], stime_i=self.stime, co2=self.co2_data,
                                    tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
            else:
                grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                        tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
            self.gridcells.append(grd_cell)
            self.lats[i] = grd_cell.lat
            self.lons[i] = grd_cell.lon
            print_progress(i+1, len(self.yx_indices), prefix='Progress:', suffix='Complete')
            i += 1
            # Print data about the model execution: number of metacommunities, number of gridcells, etc.
        print(f"Number of gridcells: {len(self.gridcells)}")
        print(f"Number of metacommunities: {self.config.metacomm.n}") # type: ignore
        print(f"Maximum number of PLS per community: {self.config.metacomm.npls_max}") # type: ignore


    def run_region_map(self, func: Callable):
        """Run a function across all gridcells using multiprocessing.Pool with async I/O"""

        result = []

        if self.file_objects:
            j = 0
            i = 0
            pending_writes = []  # Track pending write operations

            print_progress(j, len(self.file_objects), prefix='Progress:', suffix='Complete')
            self.gridcell_address_file_objects = {}
            # Create a dedicated executor for I/O operations
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="FileWriter") as io_executor:

                for f in self.file_objects:
                    # Load and process current chunk
                    with open(f, 'rb') as fh:
                        new_chunk = load(fh)

                    for k, _ in enumerate(new_chunk):
                        self.gridcell_address_file_objects[i] = (k, f)
                        i += 1

                    # Process chunk with multiprocessing
                    num_workers = min(self.config.multiprocessing.max_processes, len(new_chunk))
                    with mp.Pool(processes=num_workers) as pool:
                        try:
                            result = pool.map(func, new_chunk)
                        except Exception as e:
                            print(f"Error during multiprocessing: {e}")
                            pool.terminate()
                            raise
                        finally:
                            pool.close()
                            pool.join()

                    # Submit async write operation
                    def write_chunk_to_file(filename, data):
                        """Write chunk data to file with compression."""
                        with open(filename, 'wb') as fh:
                            dump(data, fh, compress=self.compression) # type: ignore
                            fh.flush()
                        # gc.collect()
                        return filename

                    # Submit write task to I/O executor (non-blocking)
                    write_future = io_executor.submit(write_chunk_to_file, f, result)
                    pending_writes.append(write_future)

                    # Limit concurrent writes to prevent memory buildup
                    if len(pending_writes) > 3:  # Keep max 3 pending writes
                        oldest_write = pending_writes.pop(0)
                        try:
                            oldest_write.result()  # Wait for the oldest write to complete
                        except Exception as e:
                            print(f"Error during async write: {e}")
                            raise

                    result = []  # Clear result for next iteration
                    print_progress(j+1, len(self.file_objects), prefix='Progress:', suffix='Complete')
                    j += 1

                # Wait for all remaining writes to complete
                for future in pending_writes:
                    try:
                        result_file = future.result()
                        print(f"Write completed: {result_file}")
                    except Exception as e:
                        print(f"Error during final async write: {e}")
                        raise

        else:
            # First run -> no file objects
            is_netcdf_input = self.input_type == "netcdf" and self.input_method == "ih"
            jobs = list(zip(self.climate_files, self.yx_indices))
            self.region_size = len(jobs)
            cpu_count = self.config.multiprocessing.max_processes
            chunks = [jobs[i:i + cpu_count] for i in range(0, len(jobs), cpu_count)]
            i = 0
            j = 0
            pending_writes = []
            self.gridcell_address_file_objects = {}

            print_progress(j, len(chunks), prefix='Progress:', suffix='Complete')

            # Create executor for first run
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="FileWriter") as io_executor:
                current_batch = 0
                for chunk in chunks:
                    if is_netcdf_input:
                        self.input_handler = input_handler(self.input_data,
                                                           self.gridlist,
                                                           batch_size=self.nchunks)
                    # if self.input_method == "ih":
                    #     print(self.input_handler.get_batch_data(current_batch)['gridlist'])
                    #     print_progress(j, len(chunks), prefix='Progress:', suffix='Complete', nl="\n")

                    if self.state_path is None:
                        raise ValueError("State path is not set. Cannot write gridcells to files")

                    fname = self.state_path / Path(f"region_file_{uuid4()}.z")
                    k = 0
                    result = []

                    for f, pos in chunk:
                        y, x = pos
                        gridcell_dump_directory = self.output_path / Path(f"grd_{y}-{x}")
                        grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
                        if self.input_method == "ih":
                            batch_data = self.input_handler.get_batch_data(current_batch)
                            input_data = batch_data['data'][grd_cell.station_name]
                            grd_cell.set_gridcell(input_data, stime_i=self.stime, co2=self.co2_data,
                                                tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
                        else:
                            grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                                tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
                        result.append(grd_cell)
                        self.lats[i] = grd_cell.lat
                        self.lons[i] = grd_cell.lon
                        self.gridcell_address_file_objects[i] = (k, fname)
                        i += 1
                        k += 1
                    current_batch += 1
                    self.file_objects.append(fname)

                    # Process with multiprocessing
                    if is_netcdf_input:
                        self.input_handler = None  # Set to none because netCDF4 Dataset is not pickable
                    num_workers = min(self.config.multiprocessing.max_processes, len(result))
                    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                        try:
                            processed_result = pool.map(func, result, chunksize=1)
                        except Exception as e:
                            print(f"Error during multiprocessing: {e}")
                            pool.terminate()
                            raise
                        finally:
                            pool.close()
                            pool.join()

                    # Submit async write operation
                    def write_chunk_to_file(filename, data):
                        """Write chunk data to file with compression."""
                        with open(filename, 'wb') as f:
                            dump(data, f, compress=self.compression) # type: ignore
                            f.flush()
                        # gc.collect()
                        return filename

                    write_future = io_executor.submit(write_chunk_to_file, fname, processed_result)
                    pending_writes.append(write_future)

                    # Limit concurrent writes
                    if len(pending_writes) > 3:
                        oldest_write = pending_writes.pop(0)
                        try:
                            oldest_write.result()
                        except Exception as e:
                            print(f"Error during async write: {e}")
                            raise

                    result = []
                    print_progress(j+1, len(chunks), prefix='Progress:', suffix='Complete')
                    j += 1

                # Wait for all remaining writes to complete
                for future in pending_writes:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error during final async write: {e}")
                        raise


    def run_region_starmap(self, func: Callable, args):
        """Run a function with arguments across all gridcells using multiprocessing.Pool with async I/O"""

        if not self.file_objects:
            raise ValueError("No file objects found. Cannot run starmap without file objects")

        j = 0
        i = 0
        pending_writes = []  # Track pending write operations

        print_progress(j, len(self.file_objects), prefix='Progress:', suffix='Complete')
        self.gridcell_address_file_objects = {}

        # Create a dedicated executor for I/O operations
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="FileWriter") as io_executor:

            for f in self.file_objects:
                # Load current chunk
                with open(f, 'rb') as fh:
                    new_chunk = load(fh)

                for k, _ in enumerate(new_chunk):
                    self.gridcell_address_file_objects[i] = (k, f)
                    i += 1

                # Process chunk with multiprocessing starmap
                num_workers = min(self.config.multiprocessing.max_processes, len(new_chunk))
                with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                    try:
                        result = pool.starmap(func, [(gc, args) for gc in new_chunk], chunksize=1)
                    except Exception as e:
                        print(f"Error during multiprocessing: {e}")
                        pool.terminate()
                        raise
                    finally:
                        pool.close()
                        pool.join()

                # Submit async write operation
                def write_chunk_to_file(filename, data):
                    """Write chunk data to file with compression."""
                    with open(filename, 'wb') as fh:
                        dump(data, fh, compress=self.compression) # type: ignore
                        fh.flush()
                    # gc.collect()
                    return filename

                # Submit write task to I/O executor (non-blocking)
                write_future = io_executor.submit(write_chunk_to_file, f, result)
                pending_writes.append(write_future)

                # Optional: limit concurrent writes to prevent memory buildup
                if len(pending_writes) > 3:  # Keep max 3 pending writes
                    oldest_write = pending_writes.pop(0)
                    try:
                        oldest_write.result()  # Wait for the oldest write to complete
                    except Exception as e:
                        print(f"Error during async write: {e}")
                        raise

                print_progress(j+1, len(self.file_objects), prefix='Progress:', suffix='Complete')
                j += 1

            # Wait for all remaining writes to complete
            for future in pending_writes:
                try:
                    result_file = future.result()
                    # Optionally log completion: print(f"Write completed: {result_file}")
                except Exception as e:
                    print(f"Error during final async write: {e}")
                    raise


    def clean_model_state(self):
        """
        Clean state of the region by removing unnecessary attributes from gridcells.
        This method iterates over all gridcells in the region and removes attributes
        that are not in the predefined set of attributes to keep. This is useful for
        reducing memory usage and ensuring that only essential attributes are retained
        in the gridcells after a model run. Only attributes that are necessary for model
        output post processing are kept, while all other attributes are removed.
        It renders the current model state unusable for further model runs.
        Raises:
            ValueError: If no file objects are found.

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
                              'metacomm_output',
                              'run_counter',
                              'x',
                              'xres',
                              'xyname',
                              'y',
                              'yres',
                              'grid_filename'
                              }

        if not self.file_objects:
            raise ValueError("No file objects found. Cannot clean model state.")

        def process_file(f):
            with open(f, 'rb') as file:
                gridcells = load(file)

            # Clean each gridcell in this file
            for gridcell in gridcells:
                all_attributes = set(gridcell.__dict__.keys())
                # Delete attributes that are not in the subset
                for attr in all_attributes - attributes_to_keep:
                    delattr(gridcell, attr)
                # gc.collect()
            # Save the cleaned gridcells back to the file this renders the current model state unusable for model run
            with open(f, 'wb') as file:
                dump(gridcells, file, compress=self.compression) # type: ignore
                file.flush()

        with ThreadPoolExecutor(max_workers=min(32, len(self.file_objects))) as executor:
            list(executor.map(process_file, self.file_objects))


    def clean_model_state_fast(self):
        """
        Clean state of the region by removing unnecessary attributes from gridcells.
        Optimized version with better parallelization and memory management.
        """


        if not self.file_objects:
            raise ValueError("No file objects found. Cannot clean model state.")

        def process_file_mp(f):
            """Process file - clean gridcells sequentially within each file"""
            attributes_to_keep = {'calendar', 'time_unit', 'cell_area', 'config',
                                  'executed_iterations', 'lat', 'lon', 'ncomms',
                                  'out_dir', 'outputs', 'metacomm_output', 'run_counter',
                                  'x', 'xres', 'xyname', 'y', 'yres', 'grid_filename'}
            # Load gridcells
            with open(f, 'rb') as file:
                gridcells = load(file)

            for gridcell in gridcells:
                all_attributes = set(gridcell.__dict__.keys())
                for attr in all_attributes - attributes_to_keep:
                    delattr(gridcell, attr)

            # Save cleaned gridcells
            with open(f, 'wb') as file:
                dump(gridcells, file, compress=self.compression)
                file.flush()

        # Process files in parallel using joblib
        # Use multiprocessing backend for CPU-bound attribute deletion
        Parallel(n_jobs=min(len(self.file_objects), mp.cpu_count()),
                backend='loky',
                verbose=10)(delayed(process_file_mp)(f) for f in self.file_objects)


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


    def __getitem__(self, idx:int):
        """Get a gridcell by its index using the load_gridcell method"""
        if self.gridcells:
            # If gridcells are already loaded in memory, return the requested gridcell
            if idx < 0 or idx >= len(self.gridcells):
                raise IndexError(f"Index {idx} out of range. Region has {self.__len__()} gridcells")
            return self.gridcells[idx]
        else:
            # If gridcells are not loaded, use the load_gridcell method to fetch it
            if not self.file_objects:
                raise ValueError("No file objects found. Cannot read intermediate files")
            if idx < 0 or idx >= self.region_size:
                raise IndexError(f"Index {idx} out of range. Region has {self.__len__()} gridcells")
            return self.load_gridcell(idx)


    def __len__(self):
        if self.gridcells:
            return len(self.gridcells)
        return self.region_size


    def __iter__(self):
        """Iterator that yields gridcells one at a time using load_gridcell for memory efficiency."""
        if self.gridcells:
            # If gridcells are already loaded in memory, yield them directly
            yield from self.gridcells
        else:
            # Use memory-conservative approach: load gridcells one at a time
            if not self.file_objects:
                raise ValueError("No file objects found. Cannot iterate over gridcells")

            # Build the gridcell address mapping if it doesn't exist
            if not self.gridcell_address_file_objects:
                self._build_gridcell_address_mapping()

            # Group gridcells by file to minimize file I/O
            for f in self.file_objects:
                # Open each file only once and yield all gridcells from it
                with open(f, 'rb') as fh:
                    gridcells_chunk = load(fh)
                    # Yield each gridcell in the chunk sequentially
                    yield from gridcells_chunk


    def _build_gridcell_address_mapping(self):
        """Build the mapping of gridcell indices to file objects and positions."""
        if self.gridcell_address_file_objects:
            return  # Already built

        self.gridcell_address_file_objects = {}
        global_idx = 0

        for f in self.file_objects:
            with open(f, 'rb') as fh:
                gridcells_chunk = load(fh)
                for local_idx in range(len(gridcells_chunk)):
                    self.gridcell_address_file_objects[global_idx] = (local_idx, f)
                    global_idx += 1


    def __del__(self):
        """Ensure proper cleanup of multiprocessing resources."""
        try:
            if hasattr(self, 'gridcells') and self.gridcells:
                for gridcell in self.gridcells:
                    if hasattr(gridcell, 'outputs'):
                        gridcell.outputs.clear()
                    if hasattr(gridcell, 'metacomm_output'):
                        gridcell.metacomm_output.clear()
        except Exception as e:
            print(f"Error during cleanup: {e}")