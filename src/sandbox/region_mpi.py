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

import copy
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from shutil import copy2

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
from uuid import uuid4
from numpy.typing import NDArray
from joblib import dump, load
import numpy as np

# MPI imports
from mpi4py import MPI

from config import Config, fetch_config
from parameters import hsoil, ssoil, tsoil
import metacommunity as mc

if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import str_or_path, get_co2_concentration, read_bz2_file, print_progress, grd_mt

# Standard output path for the model
from parameters import output_path
from input_handler import input_handler


class region_mpi:
    """
    MPI-enabled Region class containing the gridcells for a given region using MPI for parallelization.
    
    This class replaces the multiprocessing-based region class with MPI (Message Passing Interface)
    for better scalability across multiple nodes and improved resource management.
    
    Key MPI Features:
    - Distributed work allocation across MPI processes
    - Collective operations for data gathering and synchronization
    - Fault-tolerant file I/O with collision avoidance
    - Progress tracking across all ranks
    - Memory-efficient processing for large regions
    
    Usage:
        # Initialize MPI environment first (usually done by mpirun/mpiexec)
        region = region_mpi(name, clim_data, soil_data, co2, pls_table)
        
        # Set up gridcells (distributed across ranks)
        region.set_gridcells()
        
        # Run processing functions (distributed)
        region.run_region_map(processing_function)
        
        # Access results (available on all ranks)
        results = region.get_performance_stats()
    
    Note:
        This class requires mpi4py and should be run with mpirun/mpiexec:
        mpirun -n <num_processes> python your_script.py
    """

    def __init__(self,
                name: str,
                clim_data: Union[str, Path],
                soil_data: Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
                co2: Union[str, Path],
                pls_table: NDArray) -> None:
        """Initialize MPI-enabled region

        Args:
            name (str): this will be the name of the region and the name of the output folder
            clim_data (Union[str,Path]): Path for the climate data
            soil_data (Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]]): soil data
            co2 (Union[str, Path]): CO2 data path
            pls_table (np.ndarray): Plant life strategy table
        """
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.config: Config = fetch_config(os.path.join(os.path.dirname(__file__), 'caete.toml'))

        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # IO
        self.climate_files = []  # Climate files for the region. Used only during the set_gridcells method.
        self.input_data = str_or_path(clim_data)
        self.soil_data = copy.deepcopy(soil_data)
        self.pls_table = mc.pls_table(pls_table)
        self.file_objects = []
        self.region_size = 0

        # calculate_matrix size from grid resolution
        self.nx = len(np.arange(0, 180, self.config.crs.xres/2))  # type: ignore
        self.ny = len(np.arange(0,  90, self.config.crs.yres/2))  # type: ignore

        self.epsg = f"EPSG:{self.config.crs.epsg_id}"  # type: ignore
        self.datum = self.config.crs.datum  # type: ignore

        # Grid mask of the region
        self.grid_mask = np.ones((self.ny, self.nx), dtype=bool)

        # Number of PLS in the main table (global table)
        self.npls_main_table = self.pls_table.npls

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

        # These paths are only used at construction time (this method) and
        # at gridcell creation time (self.set_gridcell). After that,
        # the path to the climate files is update at the gridcell level when needed.
        for file_path in sorted(list(self.input_data.glob("input_data_*-*.pbz2"))):
            self.climate_files.append(file_path)

        # Define grid structure
        self.yx_indices = []
        self.lats = np.zeros(len(self.climate_files), dtype=np.float32)
        self.lons = np.zeros(len(self.climate_files), dtype=np.float32)
        for f in self.climate_files:
            # Warning: This is a very specific way to extract the gridcell indices from the file name
            # Thus, the file name must be in the format input_data_y-x.pbz2
            # Look at the ../input/pre_processing.py file to see how the files are created
            y, x = f.stem.split("_")[-1].split("-")
            self.yx_indices.append((int(y), int(x)))
            # Update the grid mask
            self.grid_mask[int(y), int(x)] = False

        # create the output folder structure
        # This is the output path for the regions, Create it if it does not exist
        # output_path is a global defined in parameters.py. The region object will
        # create the internal output folder structure into this directory
        if self.rank == 0:
            os.makedirs(output_path, exist_ok=True)

        # This is the output path for this region
        self.output_root = output_path
        self.output_path = self.output_root / self.name

        # Path to the state files for this region
        self.state_path = self.output_root / f"state_files_{uuid4().hex}"
        # Mapping of gridcells to their file objects
        self.gridcell_address_file_objects = {}

        # Create the output and state folders for this region (only on rank 0)
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.state_path, exist_ok=True)

        # Synchronize all processes
        self.comm.Barrier()

        # A list to store this region's gridcells
        # Some magic methods are defined to deal with this list
        self.gridcells: List[grd_mt] = []

    def _distribute_work(self, work_items: List) -> List:
        """Distribute work items across MPI processes"""
        # Calculate work distribution
        total_work = len(work_items)
        work_per_process = total_work // self.size
        remainder = total_work % self.size
        
        # Calculate start and end indices for this process
        start_idx = self.rank * work_per_process + min(self.rank, remainder)
        end_idx = start_idx + work_per_process + (1 if self.rank < remainder else 0)
        
        # Return work for this process
        return work_items[start_idx:end_idx]

    def _gather_results(self, local_results: List) -> List:
        """Gather results from all MPI processes"""
        # Gather all results to rank 0
        all_results = self.comm.gather(local_results, root=0)
        
        if self.rank == 0:
            # Flatten the list of lists
            flattened_results = []
            for result_list in all_results:
                flattened_results.extend(result_list)
            return flattened_results
        else:
            return []

    def load_gridcell(self, idx: int):
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
            gridcells: List[grd_mt] = load(fh)

        return gridcells[k]

    def load_gridcells(self):
        """Load the gridcells from the intermediate files using MPI coordination."""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

        if self.gridcells:
            return

        # Distribute file loading across MPI processes
        files_to_load = self._distribute_work(self.file_objects)
        
        local_gridcells = []
        for f in files_to_load:
            try:
                with open(f, 'rb') as fh:
                    gridcells_chunk = load(fh)
                    local_gridcells.extend(gridcells_chunk)
            except Exception as e:
                print(f"Error loading file {f} on rank {self.rank}: {e}")

        # Gather all gridcells to rank 0
        all_gridcells = self.comm.gather(local_gridcells, root=0)
        
        if self.rank == 0:
            self.gridcells = []
            for gridcell_list in all_gridcells:
                self.gridcells.extend(gridcell_list)
        
        # Broadcast the complete list to all processes if needed
        # self.gridcells = self.comm.bcast(getattr(self, 'gridcells', []), root=0)

    def unload_gridcells(self):
        """Force the gridcells writing to the intermediate files using MPI"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot save intermediate files")
        if not self.gridcells:
            raise ValueError("No gridcells found. Cannot save intermediate files")
        
        # Only rank 0 has the complete gridcells list
        if self.rank == 0:
            # Divide the gridcells into chunks for each file
            chunk_size = len(self.gridcells) // len(self.file_objects)
            remainder = len(self.gridcells) % len(self.file_objects)
            
            chunks = []
            start_idx = 0
            for i in range(len(self.file_objects)):
                end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                chunks.append(self.gridcells[start_idx:end_idx])
                start_idx = end_idx
        else:
            chunks = []
        
        # Distribute chunks to processes for parallel writing
        chunk_assignments = self._distribute_work(list(zip(self.file_objects, chunks)) if self.rank == 0 else [])
        
        for f, chunk in chunk_assignments:
            try:
                with open(f, 'wb') as fh:
                    dump(chunk, fh, compress=("lzma", 9))
                    fh.flush()
            except Exception as e:
                print(f"Error saving chunk to {f} on rank {self.rank}: {e}")
        
        # Synchronize and rebuild address mapping
        self.comm.Barrier()
        if self.rank == 0:
            self.gridcells = []  # Clear from memory
        self._build_gridcell_address_mapping_mpi()

    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics for MPI execution"""
        local_stats = {
            'rank': self.rank,
            'total_ranks': self.size,
            'files_processed': len(self._distribute_work(self.file_objects)) if self.file_objects else 0,
            'region_size': self.region_size if hasattr(self, 'region_size') else 0
        }
        
        # Gather stats from all ranks
        all_stats = self.comm.gather(local_stats, root=0)
        
        if self.rank == 0 and all_stats:
            total_files = sum(stats['files_processed'] for stats in all_stats)
            return {
                'total_ranks': self.size,
                'total_files_processed': total_files,
                'region_size': self.region_size if hasattr(self, 'region_size') else 0,
                'files_per_rank': [stats['files_processed'] for stats in all_stats],
                'load_balance': max(stats['files_processed'] for stats in all_stats) / max(1, min(stats['files_processed'] for stats in all_stats)) if all_stats else 1.0
            }
        else:
            return local_stats

    def set_new_state(self):
        """Create a copy of the state files in a new folder"""
        if self.rank == 0:
            new_id = uuid4().hex
            new_state_folder = self.output_root / f"state_files_{new_id}"
            os.makedirs(new_state_folder, exist_ok=True)
        else:
            new_state_folder = None

        # Broadcast new state folder to all processes
        new_state_folder = self.comm.bcast(new_state_folder, root=0)

        # Distribute file copying across MPI processes
        files_to_copy = self._distribute_work(self.file_objects)
        
        new_file_objects = []
        for f in files_to_copy:
            try:
                new_file_path = new_state_folder / Path(f).name
                copy2(f, new_file_path)
                new_file_objects.append(new_file_path)
            except Exception as e:
                if self.rank == 0:
                    print(f"Error copying file {f}: {e}")

        # Gather all new file paths to rank 0
        all_new_files = self.comm.gather(new_file_objects, root=0)
        
        if self.rank == 0:
            # Flatten and update file objects
            self.file_objects = []
            for file_list in all_new_files:
                self.file_objects.extend(file_list)
            self.state_path = new_state_folder
        
        # Broadcast updated file objects to all processes
        self.file_objects = self.comm.bcast(getattr(self, 'file_objects', []), root=0)
        self.state_path = self.comm.bcast(getattr(self, 'state_path', None), root=0)

    def delete_state(self):
        """Delete the state files of the region using MPI parallelization"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot delete state files")

        # Distribute file deletion across MPI processes
        files_to_delete = self._distribute_work(self.file_objects)
        
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception as e:
                if self.rank == 0:
                    print(f"Error deleting file {f}: {e}")

        # Synchronize before removing directory
        self.comm.Barrier()
        
        if self.rank == 0:
            try:
                os.rmdir(str(self.state_path))
            except Exception as e:
                print(f"Error removing state directory: {e}")
        
        self.file_objects = []
        self.state_path = None

    def save_state(self, state_file: Union[str, Path], new_state=False):
        """Save the state of the region to a file

        Args:
            state_file (Union[str, Path]): Path to the file where the state will be saved
        """
        from worker import worker
        if self.rank == 0:
            worker.save_state_zstd(self, state_file)
        
        if new_state:
            self.set_new_state()

    def update_dump_directory(self, new_name: str = "copy"):
        """Update the output folder for the region

        Args:
            new_name (Union[str, Path]): name of the new folder where outputs of the region should be saved. Defaults to "copy"
        """
        self.name = Path(f"{new_name}")
        self.output_path = output_path / self.name  # Update region output folder path
        
        if self.rank == 0:
            os.makedirs(self.output_path, exist_ok=True)

        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

        # Distribute file processing across MPI processes
        files_to_process = self._distribute_work(self.file_objects)

        def process_file(f):
            gridcells = load(f)
            for gridcell in gridcells:
                gridcell.out_dir = self.output_path / Path(f"grd_{gridcell.y}-{gridcell.x}")

            with open(f, 'wb') as fh:
                dump(gridcells, fh, compress=("lzma", 9))
            gc.collect()

        # Process files on this MPI process
        for f in files_to_process:
            process_file(f)
            
        # Synchronize all processes
        self.comm.Barrier()

    def update_input(self, input_folder=None, co2=None):
        """Update the input data for the region

        Args:
            input_file (str | Path, optional): Folder with input data to be used. Defaults to None.
            co2 (str | Path, optional): Text file (tsv/csv) with annual co2 concentration. Defaults to None.
            Attributes are updated if valid values are provided

        Raises:
            FileNotFoundError: _description_
            AssertionError: _description_
        """
        t1 = time.perf_counter()
        if co2 is not None:
            self.co2_path = str_or_path(co2)
            self.co2_data = get_co2_concentration(self.co2_path)

        if input_folder is not None:
            # Read the climate data
            self.input_data = str_or_path(input_folder)
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

        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

        # Distribute file processing across MPI processes
        files_to_process = self._distribute_work(self.file_objects)

        def process_file(f):
            try:
                gridcells = load(f)
                for gridcell in gridcells:
                    gridcell.stime = self.stime
                    gridcell.co2 = self.co2_data
                    if input_folder is not None:
                        gridcell.input_fpath = self.input_data / f"input_data_{gridcell.y}-{gridcell.x}.pbz2"
                
                with open(f, 'wb') as fh:
                    dump(gridcells, fh, compress=("lzma", 9))
                gc.collect()
            except Exception as e:
                print(f"Error processing file {f}: {e}")

        # Process files on this MPI process
        for f in files_to_process:
            process_file(f)
            
        # Synchronize all processes
        self.comm.Barrier()

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
            return idx, self.pls_table.table[:, idx]  # Return 1D array like original

        assert comm_npls <= self.npls_main_table, "Number of PLS must be less than the number of PLS in the main table"

        idx = np.random.choice(self.npls_main_table, comm_npls, replace=False)
        return idx, self.pls_table.table[:, idx]

    def set_gridcells(self):
        """Set up gridcells across MPI processes with improved error handling and progress tracking"""
        if self.rank == 0:
            print("Starting gridcells setup with MPI")
        
        # Distribute work across MPI processes
        jobs = list(zip(self.climate_files, self.yx_indices))
        self.region_size = len(jobs)
        local_jobs = self._distribute_work(jobs)
        
        local_gridcells = []
        local_lats = []
        local_lons = []
        failed_jobs = []
        
        # Process local jobs with progress tracking
        for i, (f, pos) in enumerate(local_jobs):
            if self.rank == 0 and len(local_jobs) > 10:  # Only show progress for larger workloads
                print_progress(i+1, len(local_jobs), prefix=f'Rank {self.rank} Progress:', suffix='Complete')
            
            try:
                y, x = pos
                gridcell_dump_directory = self.output_path / Path(f"grd_{y}-{x}")
                grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
                grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                      tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
                local_gridcells.append(grd_cell)
                local_lats.append(grd_cell.lat)
                local_lons.append(grd_cell.lon)
            except Exception as e:
                print(f"Error setting up gridcell {pos} on rank {self.rank}: {e}")
                failed_jobs.append((f, pos, str(e)))

        # Gather results from all processes
        all_gridcells = self.comm.gather(local_gridcells, root=0)
        all_lats = self.comm.gather(local_lats, root=0)
        all_lons = self.comm.gather(local_lons, root=0)
        all_failed = self.comm.gather(failed_jobs, root=0)
        
        if self.rank == 0:
            # Report any failures
            total_failed = sum(len(failed_list) for failed_list in all_failed)
            if total_failed > 0:
                print(f"Warning: {total_failed} gridcells failed to initialize")
                for failed_list in all_failed:
                    for f, pos, error in failed_list:
                        print(f"  Failed: {pos} - {error}")
            
            # Flatten results
            for gridcell_list in all_gridcells:
                self.gridcells.extend(gridcell_list)
            
            flat_lats = []
            flat_lons = []
            for lat_list, lon_list in zip(all_lats, all_lons):
                flat_lats.extend(lat_list)
                flat_lons.extend(lon_list)
            
            self.lats = np.array(flat_lats, dtype=np.float32)
            self.lons = np.array(flat_lons, dtype=np.float32)
            
            # Print summary statistics
            successful_gridcells = len(self.gridcells)
            print(f"Successfully initialized {successful_gridcells}/{self.region_size} gridcells")
            print(f"Number of metacommunities per gridcell: {self.config.metacomm.n}")
            print(f"Maximum number of PLS per community: {self.config.metacomm.npls_max}")
            print(f"Using {self.size} MPI processes")

        # Broadcast coordinate data to all processes for consistency
        self.lats = self.comm.bcast(getattr(self, 'lats', None), root=0)
        self.lons = self.comm.bcast(getattr(self, 'lons', None), root=0)

    def run_region_map(self, func: Callable):
        """Run a function across all gridcells using MPI with enhanced progress tracking"""
        
        if self.file_objects:
            # Process existing file objects
            files_to_process = self._distribute_work(self.file_objects)
            
            if self.rank == 0:
                print_progress(0, len(self.file_objects), prefix='Progress:', suffix='Complete')
            
            processed_count = 0
            total_files = len(self.file_objects)
            
            for f in files_to_process:
                try:
                    with open(f, 'rb') as fh:
                        gridcells = load(fh)
                    
                    # Process gridcells
                    processed_gridcells = []
                    for gridcell in gridcells:
                        try:
                            result = func(gridcell)
                            processed_gridcells.append(result)
                        except Exception as e:
                            print(f"Error processing gridcell on rank {self.rank}: {e}")
                            processed_gridcells.append(gridcell)
                    
                    # Save processed gridcells back to file
                    with open(f, 'wb') as fh:
                        dump(processed_gridcells, fh, compress=("lzma", 9))
                        fh.flush()
                    
                    processed_count += 1
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing file {f} on rank {self.rank}: {e}")
                    processed_count += 1  # Count even failed files to maintain progress
            
            # Synchronize progress reporting across all ranks
            total_processed = self.comm.allreduce(processed_count, op=MPI.SUM)
            if self.rank == 0:
                print_progress(total_processed, total_files, prefix='Progress:', suffix='Complete')
        
        else:
            # First run -> no file objects
            jobs = list(zip(self.climate_files, self.yx_indices))
            self.region_size = len(jobs)
            
            # Distribute jobs across MPI processes
            local_jobs = self._distribute_work(jobs)
            
            if self.rank == 0:
                print_progress(0, len(jobs), prefix='Progress:', suffix='Complete')
                # Create file objects for storing intermediate results
                # Use a more conservative approach for file creation
                max_files = min(self.size, len(jobs), 16)  # More conservative limit
                self.file_objects = []
                for i in range(max_files):
                    fname = self.state_path / Path(f"region_file_{i}_{uuid4().hex}.z")
                    self.file_objects.append(fname)
                    # Pre-create empty file to avoid race conditions
                    with open(fname, 'wb') as f:
                        dump([], f, compress=("lzma", 9))
            else:
                self.file_objects = []
            
            # Broadcast file objects to all processes
            self.file_objects = self.comm.bcast(getattr(self, 'file_objects', []), root=0)
            
            # Determine which file this rank should write to (round-robin)
            file_idx = self.rank % len(self.file_objects)
            output_file = self.file_objects[file_idx]
            
            local_gridcells = []
            local_lats = []
            local_lons = []
            
            for i, (f, pos) in enumerate(local_jobs):
                y, x = pos
                gridcell_dump_directory = self.output_path / Path(f"grd_{y}-{x}")
                grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
                grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                      tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
                
                try:
                    processed_cell = func(grd_cell)
                    local_gridcells.append(processed_cell)
                except Exception as e:
                    print(f"Error processing gridcell on rank {self.rank}: {e}")
                    local_gridcells.append(grd_cell)
                
                local_lats.append(grd_cell.lat)
                local_lons.append(grd_cell.lon)
            
            # Handle file writing with collision avoidance
            self._write_gridcells_to_file(output_file, local_gridcells)
            
            # Gather latitude and longitude data
            all_lats = self.comm.gather(local_lats, root=0)
            all_lons = self.comm.gather(local_lons, root=0)
            
            # Synchronize before building address mapping
            self.comm.Barrier()
            
            if self.rank == 0:
                # Flatten and store coordinate data
                flat_lats = []
                flat_lons = []
                for lat_list, lon_list in zip(all_lats, all_lons):
                    flat_lats.extend(lat_list)
                    flat_lons.extend(lon_list)
                
                self.lats = np.array(flat_lats, dtype=np.float32)
                self.lons = np.array(flat_lons, dtype=np.float32)
                
                print_progress(len(jobs), len(jobs), prefix='Progress:', suffix='Complete')
            
            # Broadcast coordinate data to all processes
            self.lats = self.comm.bcast(getattr(self, 'lats', None), root=0)
            self.lons = self.comm.bcast(getattr(self, 'lons', None), root=0)
            
            # Build gridcell address mapping
            self._build_gridcell_address_mapping_mpi()
            gc.collect()

    def run_region_starmap(self, func: Callable, args):
        """Run a function with arguments across all gridcells using MPI"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot run starmap without file objects")
        
        # Distribute file processing across MPI processes
        files_to_process = self._distribute_work(self.file_objects)
        
        if self.rank == 0:
            print_progress(0, len(self.file_objects), prefix='Progress:', suffix='Complete')
        
        processed_count = 0
        total_files = len(self.file_objects)
        
        for f in files_to_process:
            try:
                with open(f, 'rb') as fh:
                    gridcells = load(fh)
                
                processed_gridcells = []
                for gridcell in gridcells:
                    try:
                        result = func(gridcell, args)
                        processed_gridcells.append(result)
                    except Exception as e:
                        print(f"Error processing gridcell on rank {self.rank}: {e}")
                        processed_gridcells.append(gridcell)
                
                with open(f, 'wb') as fh:
                    dump(processed_gridcells, fh, compress=("lzma", 9))
                    fh.flush()
                
                processed_count += 1
                gc.collect()
                
            except Exception as e:
                print(f"Error processing file {f} on rank {self.rank}: {e}")
                processed_count += 1
        
        # Synchronize progress reporting
        total_processed = self.comm.allreduce(processed_count, op=MPI.SUM)
        if self.rank == 0:
            print_progress(total_processed, total_files, prefix='Progress:', suffix='Complete')

    def run_region_batch(self, func: Callable, batch_size: int = None):
        """Run a function across gridcells in batches using MPI for memory efficiency"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot run batch processing without file objects")
        
        if batch_size is None:
            batch_size = max(1, len(self.file_objects) // (self.size * 2))
        
        # Distribute file processing across MPI processes
        files_to_process = self._distribute_work(self.file_objects)
        
        if self.rank == 0:
            print_progress(0, len(self.file_objects), prefix='Batch Progress:', suffix='Complete')
        
        processed_count = 0
        
        # Process files in batches
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            
            for f in batch_files:
                try:
                    with open(f, 'rb') as fh:
                        gridcells = load(fh)
                    
                    processed_gridcells = []
                    for gridcell in gridcells:
                        try:
                            result = func(gridcell)
                            processed_gridcells.append(result)
                        except Exception as e:
                            print(f"Error processing gridcell on rank {self.rank}: {e}")
                            processed_gridcells.append(gridcell)
                    
                    with open(f, 'wb') as fh:
                        dump(processed_gridcells, fh, compress=("lzma", 9))
                        fh.flush()
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing file {f} on rank {self.rank}: {e}")
                    processed_count += 1
            
            # Synchronize after each batch
            self.comm.Barrier()
            gc.collect()
        
        # Final progress report
        total_processed = self.comm.allreduce(processed_count, op=MPI.SUM)
        if self.rank == 0:
            print_progress(total_processed, len(self.file_objects), prefix='Batch Progress:', suffix='Complete')

    def clean_model_state(self):
        """
        Clean state of the region by removing unnecessary attributes from gridcells using MPI.
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

        # Distribute file processing across MPI processes
        files_to_process = self._distribute_work(self.file_objects)

        def process_file(f):
            with open(f, 'rb') as file:
                gridcells = load(file)

            # Clean each gridcell in this file
            for gridcell in gridcells:
                all_attributes = set(gridcell.__dict__.keys())
                # Delete attributes that are not in the subset
                for attr in all_attributes - attributes_to_keep:
                    delattr(gridcell, attr)
                gc.collect()
            
            # Save the cleaned gridcells back to the file
            with open(f, 'wb') as file:
                dump(gridcells, file, compress=("lzma", 9))

        # Process files on this MPI process
        for f in files_to_process:
            process_file(f)
        
        # Synchronize all processes
        self.comm.Barrier()

    def get_mask(self) -> np.ndarray:
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
            'proj4': f'{self.config.crs.proj4}',  # type: ignore
            'lats': self.lats,
            'lons': self.lons,
            'epsg': self.epsg,
            'datum': self.datum,
            'lat_units': self.config.crs.lat_units,  # type: ignore
            'lon_units': self.config.crs.lon_units,  # type: ignore
            'lat_zero': self.config.crs.lat_zero,  # type: ignore
            'lon_zero': self.config.crs.lon_zero,  # type: ignore
        }

    def __getitem__(self, idx: int):
        """Get a gridcell by its index using the load_gridcell method"""
        if self.gridcells:
            # If gridcells are already loaded in memory, return the requested gridcell
            if idx < 0 or idx >= len(self.gridcells):
                raise IndexError("Gridcell index out of range")
            return self.gridcells[idx]
        else:
            # If gridcells are not loaded, use the load_gridcell method to fetch it
            if not self.file_objects:
                raise ValueError("No file objects found. Cannot load gridcells")
            if idx < 0 or idx >= self.region_size:
                raise IndexError("Gridcell index out of range")
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
                with open(f, 'rb') as fh:
                    gridcells = load(fh)
                    yield from gridcells

    def _build_gridcell_address_mapping(self):
        """Build the mapping of gridcell indices to file objects and positions."""
        if self.gridcell_address_file_objects:
            return  # Already built
        
        self.gridcell_address_file_objects = {}
        global_idx = 0
        
        for f in self.file_objects:
            try:
                with open(f, 'rb') as fh:
                    gridcells = load(fh)
                    for k, _ in enumerate(gridcells):
                        self.gridcell_address_file_objects[global_idx] = (k, f)
                        global_idx += 1
            except Exception as e:
                if self.rank == 0:
                    print(f"Warning: Could not read file {f} for address mapping: {e}")

    def _build_gridcell_address_mapping_mpi(self):
        """Build the mapping of gridcell indices to file objects using MPI coordination."""
        if self.gridcell_address_file_objects:
            return  # Already built
        
        # Only rank 0 builds the mapping initially
        if self.rank == 0:
            self.gridcell_address_file_objects = {}
            global_idx = 0
            
            for f in self.file_objects:
                if not f.exists():
                    print(f"Warning: File {f} does not exist, skipping for address mapping")
                    continue
                    
                try:
                    with open(f, 'rb') as fh:
                        gridcells = load(fh)
                        for k, _ in enumerate(gridcells):
                            self.gridcell_address_file_objects[global_idx] = (k, f)
                            global_idx += 1
                except Exception as e:
                    print(f"Warning: Could not read file {f} for address mapping: {e}")
        else:
            self.gridcell_address_file_objects = {}
        
        # Broadcast the mapping to all processes
        self.gridcell_address_file_objects = self.comm.bcast(self.gridcell_address_file_objects, root=0)

    def _write_gridcells_to_file(self, output_file: Path, gridcells: List):
        """Write gridcells to file with proper synchronization to avoid collisions."""
        if not gridcells:
            return
        
        # Ensure output file exists
        if not output_file.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                dump([], f, compress=("lzma", 9))
        
        # Use rank-specific temporary file first, then append to main file
        temp_file = output_file.with_suffix(f'.tmp_rank_{self.rank}')
        
        try:
            with open(temp_file, 'wb') as f:
                dump(gridcells, f, compress=("lzma", 9))
                f.flush()
            
            # Synchronize before combining files
            self.comm.Barrier()
            
            # Only one rank per file should combine the temp files
            # Use lexicographically smallest rank for each file
            file_idx = self.file_objects.index(output_file) if output_file in self.file_objects else 0
            responsible_rank = file_idx % self.size
            
            if self.rank == responsible_rank:
                # Load existing data from main file
                try:
                    with open(output_file, 'rb') as f:
                        existing_gridcells = load(f)
                except:
                    existing_gridcells = []
                
                # Collect and combine all temp files for this output file
                all_temp_files = []
                for r in range(self.size):
                    temp_f = output_file.with_suffix(f'.tmp_rank_{r}')
                    if temp_f.exists():
                        all_temp_files.append(temp_f)
                
                combined_gridcells = existing_gridcells[:]
                for temp_f in all_temp_files:
                    try:
                        with open(temp_f, 'rb') as f:
                            temp_gridcells = load(f)
                            combined_gridcells.extend(temp_gridcells)
                        # Clean up temp file
                        temp_f.unlink()
                    except Exception as e:
                        print(f"Error processing temp file {temp_f}: {e}")
                
                # Write combined results to final file
                with open(output_file, 'wb') as f:
                    dump(combined_gridcells, f, compress=("lzma", 9))
                    f.flush()
            
            # Final synchronization
            self.comm.Barrier()
            
        except Exception as e:
            print(f"Error writing gridcells on rank {self.rank}: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    def __del__(self):
        """Ensure proper cleanup of MPI resources."""
        try:
            # MPI cleanup is usually handled automatically
            # but we can add explicit cleanup if needed
            pass
        except Exception as e:
            pass

# Convenience functions for MPI usage
def mpi_print(message, rank=None):
    """Print a message only from a specific rank (default: rank 0)"""
    comm = MPI.COMM_WORLD
    if rank is None:
        rank = 0
    if comm.Get_rank() == rank:
        print(message)

def mpi_barrier():
    """Synchronize all MPI processes"""
    comm = MPI.COMM_WORLD
    comm.Barrier()

def get_mpi_info():
    """Get MPI rank and size information"""
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()

def check_mpi_environment():
    """Check if MPI environment is properly set up and return diagnostics"""
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Test basic MPI functionality
        test_data = rank * 10
        gathered_data = comm.gather(test_data, root=0)
        
        if rank == 0:
            print(f"MPI Environment Check:")
            print(f"  Total processes: {size}")
            print(f"  Data gathering test: {'PASSED' if gathered_data == [i*10 for i in range(size)] else 'FAILED'}")
            print(f"  MPI Version: {MPI.Get_version()}")
            
        return True, rank, size
        
    except Exception as e:
        print(f"MPI Environment Error: {e}")
        return False, 0, 1

# Utility function for running MPI jobs with proper error handling
def run_mpi_region(region_setup_func, processing_func, *args, **kwargs):
    """
    Utility function to run MPI region processing with proper error handling and cleanup.
    
    Args:
        region_setup_func: Function that returns a configured region_mpi instance
        processing_func: Function to apply to each gridcell
        *args, **kwargs: Arguments to pass to processing_func
    
    Returns:
        region_mpi instance (on rank 0) or None (on other ranks)
    """
    rank, size = get_mpi_info()
    
    try:
        # Set up region
        region = region_setup_func()
        
        if rank == 0:
            print(f"Starting MPI region processing with {size} processes")
        
        # Set up gridcells if not already done
        if not hasattr(region, 'gridcells') or not region.gridcells:
            region.set_gridcells()
        
        # Run processing
        if args or kwargs:
            region.run_region_starmap(processing_func, (args, kwargs))
        else:
            region.run_region_map(processing_func)
        
        if rank == 0:
            stats = region.get_performance_stats()
            print(f"MPI processing completed. Stats: {stats}")
            return region
        
        return None
        
    except Exception as e:
        print(f"Error in MPI region processing on rank {rank}: {e}")
        # Ensure all ranks exit together
        MPI.COMM_WORLD.Abort(1)
        return None
