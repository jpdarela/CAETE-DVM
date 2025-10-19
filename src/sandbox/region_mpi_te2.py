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
from mpi4py import MPI

import copy
import os
from shutil import copy2
import sys

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray
from joblib import dump, load


from config import Config, fetch_config
from parameters import hsoil, ssoil, tsoil
import metacommunity as mc

if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import str_or_path, get_co2_concentration, read_bz2_file, print_progress, grd_mt

# Standard output path for the model
from input_handler import input_handler

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_root = rank == 0

# MPI tags for different message types
TAG_WORK = 1
TAG_RESULT = 2
TAG_TERMINATE = 3
TAG_STATUS = 4

class region:
    """Region class containing the gridcells for a given region with MPI support"""

    def __init__(self,
                name:str,
                clim_data:Union[str,Path],
                soil_data:Tuple[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
                                Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
                co2:Union[str, Path],
                pls_table:NDArray,
                gridlist:Optional[pl.DataFrame] = None)->None:
        """Initialize region with MPI awareness"""
        
        self.config: Config = fetch_config()
        self.compression = tuple(self.config.compression.model_dump().values())
        self.gridlist = gridlist
        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # MPI-specific attributes
        self.rank = rank
        self.size = size
        self.is_root = is_root
        
        # IO
        self.climate_files = []
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
        self.nx = len(np.arange(0, 180, self.config.crs.xres/2))
        self.ny = len(np.arange(0,  90, self.config.crs.yres/2))
        # Grid mask of the region
        self.grid_mask = np.ones((self.ny, self.nx), dtype=bool)
        self.epsg = f"EPSG:{self.config.crs.epsg_id}"
        self.datum = self.config.crs.datum


        # Number of PLS in the main table (global table)
        self.npls_main_table = self.pls_table.npls
        
        # Read the metadata from the climate files
        if self.input_method == "legacy":
            try:
                metadata_file = list(self.input_data.glob("METADATA.pbz2"))[0]
            except:
                raise FileNotFoundError("Metadata file not found in the input data folder")

            try:
                mtd = str_or_path(metadata_file, check_is_file=True)
            except:
                raise AssertionError("Metadata file path could not be resolved. Cannot proceed without metadata")

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
            self.grid_mask[int(y), int(x)] = False

        # Only root process creates output directories
        if self.is_root:
            os.makedirs(self.config.output.output_dir, exist_ok=True)
            self.output_root = self.config.output.output_dir
            self.output_path = self.output_root / self.name
            self.state_path:Path | None = self.output_path / f"state_files_{uuid4().hex}"
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.state_path, exist_ok=True)
        else:
            self.output_root = self.config.output.output_dir
            self.output_path = self.output_root / self.name
            self.state_path = None
        
        # Broadcast paths to all processes
        self.output_path = comm.bcast(self.output_path, root=0)
        self.state_path = comm.bcast(self.state_path, root=0)
        
        self.gridcell_address_file_objects = {}
        self.gridcells:List[grd_mt] = []


    def run_region_map_mpi(self, func: Callable):
        """Run a function across all gridcells using MPI"""
        
        if self.file_objects:
            # Process existing file objects
            if self.is_root:
                print_progress(0, len(self.file_objects), prefix='Progress:', suffix='Complete')
                
            for file_idx, f in enumerate(self.file_objects):
                # Root loads and distributes work
                if self.is_root:
                    with open(f, 'rb') as fh:
                        chunk = load(fh)
                    chunk_size = len(chunk)
                    
                    # Update gridcell address mapping
                    start_idx = file_idx * self.config.multiprocessing.max_processes
                    for k in range(chunk_size):
                        self.gridcell_address_file_objects[start_idx + k] = (k, f)
                else:
                    chunk = None
                    chunk_size = 0
                
                # Broadcast chunk size
                chunk_size = comm.bcast(chunk_size, root=0)
                
                # Distribute work among MPI processes
                base_work = chunk_size // self.size
                remainder = chunk_size % self.size
                
                # # Calculate work distribution for this process
                # if self.rank < remainder:
                #     local_start = self.rank * (base_work + 1)
                #     local_count = base_work + 1
                # else:
                #     local_start = remainder * (base_work + 1) + (self.rank - remainder) * base_work
                #     local_count = base_work
                
                # Scatter gridcells
                if self.is_root:
                    send_data = []
                    for r in range(self.size):
                        if r < remainder:
                            start = r * (base_work + 1)
                            count = base_work + 1
                        else:
                            start = remainder * (base_work + 1) + (r - remainder) * base_work
                            count = base_work
                        send_data.append(chunk[start:start+count] if count > 0 else [])
                else:
                    send_data = None
                
                local_gridcells = comm.scatter(send_data, root=0)
                
                # Process local gridcells
                local_results = []
                for gc in local_gridcells:
                    try:
                        result = func(gc)
                        local_results.append(result)
                    except Exception as e:
                        if self.is_root:
                            print(f"Rank {self.rank}: Error processing gridcell: {e}")
                        local_results.append(gc)
                
                # Gather results back to root
                all_results = comm.gather(local_results, root=0)
                
                # Root saves results
                if self.is_root:
                    final_results = []
                    for r in range(self.size):
                        if all_results[r]:
                            final_results.extend(all_results[r])
                    
                    with open(f, 'wb') as fh:
                        dump(final_results, fh, compress=self.compression)
                        fh.flush()
                    
                    print_progress(file_idx + 1, len(self.file_objects), 
                                 prefix='Progress:', suffix='Complete')
            
            # Synchronize all processes
            comm.Barrier()
            
        else:
            # First run - create gridcells
            self._create_gridcells_mpi(func)
    
    
    def run_region_starmap_mpi(self, func: Callable, args):
        """Run a function with arguments across all gridcells using MPI"""
        
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot run starmap without file objects")
        
        if self.is_root:
            print_progress(0, len(self.file_objects), prefix='Progress:', suffix='Complete')
        
        for file_idx, f in enumerate(self.file_objects):
            # Root loads and distributes work
            if self.is_root:
                with open(f, 'rb') as fh:
                    chunk = load(fh)
                chunk_size = len(chunk)
                
                # Update gridcell address mapping
                start_idx = file_idx * self.config.multiprocessing.max_processes
                for k in range(chunk_size):
                    self.gridcell_address_file_objects[start_idx + k] = (k, f)
            else:
                chunk = None
                chunk_size = 0
            
            # Broadcast chunk size
            chunk_size = comm.bcast(chunk_size, root=0)
            
            # Distribute work
            base_work = chunk_size // self.size
            remainder = chunk_size % self.size
            
            # Calculate work distribution for ALL processes at once
            work_distribution = []
            for r in range(self.size):
                if r < remainder:
                    start = r * (base_work + 1)
                    count = base_work + 1
                else:
                    start = remainder * (base_work + 1) + (r - remainder) * base_work
                    count = base_work
                work_distribution.append((start, count))
            
            # Get this process's portion
            # local_start, local_count = work_distribution[self.rank]
            
            # Scatter gridcells
            if self.is_root:
                send_data = []
                for start, count in work_distribution:
                    send_data.append(chunk[start:start+count] if count > 0 else [])
            else:
                send_data = None
            
            local_gridcells = comm.scatter(send_data, root=0)
            
            # Process with arguments
            local_results = []
            for gc in local_gridcells:
                try:
                    result = func(gc, args)
                    local_results.append(result)
                except Exception as e:
                    if self.is_root:
                        print(f"Rank {self.rank}: Error: {e}")
                    local_results.append(gc)
            
            # Gather results
            all_results = comm.gather(local_results, root=0)
            
            # Root saves results
            if self.is_root:
                final_results = []
                for r in range(self.size):
                    if all_results[r]:
                        final_results.extend(all_results[r])
                
                with open(f, 'wb') as fh:
                    dump(final_results, fh, compress=self.compression)
                    fh.flush()
                
                print_progress(file_idx + 1, len(self.file_objects), 
                             prefix='Progress:', suffix='Complete')
        
        comm.Barrier()

    
    def _create_gridcells_mpi(self, func: Callable):
        """Create gridcells for first run using MPI"""
        
        is_netcdf_input = self.input_type == "netcdf" and self.input_method == "ih"
        jobs = list(zip(self.climate_files, self.yx_indices))
        total_jobs = len(jobs)
        
        # Distribute jobs among MPI processes
        base_work = total_jobs // self.size
        remainder = total_jobs % self.size
        
        if self.rank < remainder:
            local_start = self.rank * (base_work + 1)
            local_count = base_work + 1
        else:
            local_start = remainder * (base_work + 1) + (self.rank - remainder) * base_work
            local_count = base_work
        
        local_jobs = jobs[local_start:local_start + local_count] if local_count > 0 else []
        
        # Each process creates its gridcells
        local_gridcells = []
        local_lats = []
        local_lons = []
        
        for f, pos in local_jobs:
            y, x = pos
            gridcell_dump_directory = self.output_path / Path(f"grd_{y}-{x}")
            
            # Create directory (all processes can do this safely)
            os.makedirs(gridcell_dump_directory, exist_ok=True)
            
            grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
            
            if self.input_method == "ih":
                if is_netcdf_input:
                    self.input_handler = input_handler(self.input_data,
                                                       self.gridlist,
                                                       batch_size=self.nchunks)
                batch_data = self.input_handler.get_batch_data(0)  # Simplified for MPI
                input_data = batch_data['data'][grd_cell.station_name]
                grd_cell.set_gridcell(input_data, stime_i=self.stime, co2=self.co2_data,
                                    tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
            else:
                grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                    tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
            
            # Process the gridcell
            result = func(grd_cell)
            local_gridcells.append(result)
            local_lats.append(grd_cell.lat)
            local_lons.append(grd_cell.lon)
        
        # Gather all results to root
        all_gridcells = comm.gather(local_gridcells, root=0)
        all_lats = comm.gather(local_lats, root=0)
        all_lons = comm.gather(local_lons, root=0)
        
        if self.is_root:
            # Flatten and organize results
            flat_gridcells = []
            flat_lats = []
            flat_lons = []
            
            for gc_list, lat_list, lon_list in zip(all_gridcells, all_lats, all_lons):
                flat_gridcells.extend(gc_list)
                flat_lats.extend(lat_list)
                flat_lons.extend(lon_list)
            
            # Save to files in chunks
            cpu_count = self.config.multiprocessing.max_processes
            chunks = [flat_gridcells[i:i + cpu_count] for i in range(0, len(flat_gridcells), cpu_count)]
            
            for chunk_idx, chunk in enumerate(chunks):
                fname = self.state_path / Path(f"region_file_{uuid4().hex}.z")
                self.file_objects.append(fname)
                
                with open(fname, 'wb') as fh:
                    dump(chunk, fh, compress=self.compression)
                    fh.flush()
                
                # Update gridcell address mapping
                for local_idx, gc in enumerate(chunk):
                    global_idx = chunk_idx * cpu_count + local_idx
                    self.gridcell_address_file_objects[global_idx] = (local_idx, fname)
            
            # Update lat/lon arrays
            self.lats[:len(flat_lats)] = np.array(flat_lats, dtype=np.float32)
            self.lons[:len(flat_lons)] = np.array(flat_lons, dtype=np.float32)
            
            print(f"Created {len(flat_gridcells)} gridcells using {self.size} MPI processes")
        
        # Broadcast file objects to all processes
        self.file_objects = comm.bcast(self.file_objects, root=0)
        self.gridcell_address_file_objects = comm.bcast(self.gridcell_address_file_objects, root=0)
        
        comm.Barrier()
    

    def update_input_mpi(self, input_data=None, co2=None):
        """Update input data using MPI parallelization"""
        
        is_netcdf_input = self.input_type == "netcdf" and self.input_method == "ih"
        
        # Root handles file loading and CO2 update
        if self.is_root:
            if co2 is not None:
                self.co2_path = str_or_path(co2)
                self.co2_data = get_co2_concentration(self.co2_path)
            
            if input_data is not None:
                self.input_data = str_or_path(input_data)
                if self.input_method == "legacy":
                    try:
                        metadata_file = list(self.input_data.glob("METADATA.pbz2"))[0]
                        mtd = str_or_path(metadata_file, check_is_file=True)
                        self.metadata = read_bz2_file(mtd)
                    except:
                        raise FileNotFoundError("Metadata file not found")
                else:
                    self.input_handler = input_handler(self.input_data,
                                                       self.gridlist,
                                                       batch_size=self.nchunks)
                    self.metadata = self.input_handler.get_metadata
                
                self.stime = copy.deepcopy(self.metadata[0])
        
        # Broadcast updated data to all processes
        if co2 is not None:
            self.co2_data = comm.bcast(self.co2_data if self.is_root else None, root=0)
        
        if input_data is not None:
            self.metadata = comm.bcast(self.metadata if self.is_root else None, root=0)
            self.stime = comm.bcast(self.stime if self.is_root else None, root=0)
        
        if not self.file_objects:
            raise ValueError("No file objects found")
        
        # Distribute files among MPI processes
        files_per_process = len(self.file_objects) // self.size
        remainder = len(self.file_objects) % self.size
        
        if self.rank < remainder:
            start_idx = self.rank * (files_per_process + 1)
            local_files = self.file_objects[start_idx:start_idx + files_per_process + 1]
        else:
            start_idx = remainder * (files_per_process + 1) + (self.rank - remainder) * files_per_process
            local_files = self.file_objects[start_idx:start_idx + files_per_process]
        
        # Process local files
        for f in local_files:
            with open(f, 'rb') as file:
                gridcells = load(file)
            
            for gridcell in gridcells:
                if self.input_method == "ih":
                    # Note: This would need adjustment for proper data distribution
                    # For simplicity, assuming data is available
                    pass
                else:
                    if co2 is not None:
                        gridcell.change_input(self.input_data, self.stime, self.co2_data)
                    else:
                        gridcell.change_input(self.input_data, self.stime, None)
            
            with open(f, 'wb') as file:
                dump(gridcells, file, compress=self.compression)
                file.flush()
        
        comm.Barrier()
    

    def clean_model_state_mpi(self):
        """Clean model state using MPI parallelization"""
        
        if not self.file_objects:
            raise ValueError("No file objects found")
        
        attributes_to_keep = {'calendar', 'time_unit', 'cell_area', 'config',
                              'executed_iterations', 'lat', 'lon', 'ncomms',
                              'out_dir', 'outputs', 'metacomm_output', 'run_counter',
                              'x', 'xres', 'xyname', 'y', 'yres', 'grid_filename'}
        
        # Distribute files among processes
        files_per_process = len(self.file_objects) // self.size
        remainder = len(self.file_objects) % self.size
        
        if self.rank < remainder:
            start_idx = self.rank * (files_per_process + 1)
            local_files = self.file_objects[start_idx:start_idx + files_per_process + 1]
        else:
            start_idx = remainder * (files_per_process + 1) + (self.rank - remainder) * files_per_process
            local_files = self.file_objects[start_idx:start_idx + files_per_process]
        
        # Process local files
        for f in local_files:
            with open(f, 'rb') as file:
                gridcells = load(file)
            
            for gridcell in gridcells:
                all_attributes = set(gridcell.__dict__.keys())
                for attr in all_attributes - attributes_to_keep:
                    delattr(gridcell, attr)
            
            with open(f, 'wb') as file:
                dump(gridcells, file, compress=self.compression)
                file.flush()
        
        comm.Barrier()
        
        if self.is_root:
            print("Model state cleaned successfully")
    

    # Keep non-MPI methods unchanged
    def load_gridcell(self, idx:int):
        """Load a gridcell by its index (only on root process)"""
        if not self.is_root:
            return None
        
        if not self.file_objects:
            raise ValueError("No file objects found")
        if idx < 0 or idx >= self.region_size:
            raise IndexError("Gridcell index out of range")
        
        k, f = self.gridcell_address_file_objects[idx]
        with open(f, 'rb') as fh:
            gridcells = load(fh)
        return gridcells[k]
    

    def load_gridcells(self):
        """Load all gridcells (only on root process)"""
        if not self.is_root:
            return
        
        if not self.file_objects:
            raise ValueError("No file objects found")
        
        if self.gridcells:
            return
        
        def load_file(f):
            try:
                with open(f, 'rb') as fh:
                    return load(fh)
            except Exception as e:
                print(f"Error loading file {f}: {e}")
                raise
        
        with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
            results = list(executor.map(load_file, self.file_objects))
        
        for gridcell_list in results:
            self.gridcells.extend(gridcell_list)
    

    def save_state(self, state_file:Union[str, Path], new_state=False):
        """Save state (only root process)"""
        if self.is_root:
            from worker import worker
            worker.save_state_zstd(self, self.output_root / state_file)
            if new_state:
                self.set_new_state()
        comm.Barrier()
    

    def get_from_main_table(self, comm_npls) -> Tuple[Union[int, NDArray[np.intp]], NDArray[np.float32]]:
        """Get PLS from main table"""
        assert comm_npls > 0, "Number of PLS must be greater than 0"
        
        if comm_npls == 1:
            idx = np.random.choice(self.npls_main_table, 1, replace=False)[0]
            return idx, self.pls_table.table[:, idx]
        
        assert comm_npls <= self.npls_main_table, "Number of PLS must be less than the number of PLS in the main table"
        
        idx = np.random.choice(self.npls_main_table, comm_npls, replace=False)
        return idx, self.pls_table.table[:, idx]
    
    # Keep remaining methods unchanged...
    def unload_gridcells(self):
        """Force the gridcells writing to the intermediate files (root only)"""
        if not self.is_root:
            return
            
        if not self.file_objects:
            raise ValueError("No file objects found")
        if not self.gridcells:
            raise ValueError("No gridcells found")
            
        chunks = [self.gridcells[i:i + self.config.multiprocessing.max_processes] 
                  for i in range(0, len(self.gridcells), self.config.multiprocessing.max_processes)]
        
        for f, chunk in zip(self.file_objects, chunks):
            with open(f, 'wb') as fh:
                dump(chunk, fh, compress=self.compression)
        
        self._build_gridcell_address_mapping()
    

    def set_new_state(self):
        """Create new state folder (root only)"""
        if not self.is_root:
            return
            
        new_id = uuid4().hex
        new_state_folder = self.output_path / f"state_files_{new_id}"
        os.makedirs(new_state_folder, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=min(64, len(self.file_objects))) as executor:
            new_file_objects = []
            
            def copy_file(f):
                dest_path = new_state_folder / f.name
                copy2(f, dest_path)
                return dest_path
            
            futures = [executor.submit(copy_file, f) for f in self.file_objects]
            
            for f, future in zip(self.file_objects, futures):
                try:
                    new_file_objects.append(future.result())
                except Exception as e:
                    print(f"Error copying file {f}: {e}")
                    raise e
        
        self.file_objects = new_file_objects
        self.state_path = new_state_folder
    

    def delete_state(self):
        """Delete state files (root only)"""
        if not self.is_root:
            return
            
        if not self.file_objects:
            raise ValueError("No file objects found")
        
        with ThreadPoolExecutor(max_workers=min(32, len(self.file_objects))) as executor:
            futures = [executor.submit(os.remove, f) for f in self.file_objects]
            
            for f, future in zip(self.file_objects, futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
                    raise e
        
        self.file_objects = []
        os.rmdir(str(self.state_path))
        self.state_path = None
    

    def update_dump_directory(self, new_name:str="copy"):
        """Update output directory (all processes)"""
        self.name = Path(f"{new_name}")
        self.output_path = self.output_root / self.name
        
        if self.is_root:
            os.makedirs(self.output_path, exist_ok=True)
        
        comm.Barrier()
        
        if not self.file_objects and not self.gridcells:
            raise ValueError("No file objects or gridcells found")
        
        if self.file_objects:
            # Distribute work among processes
            files_per_process = len(self.file_objects) // self.size
            remainder = len(self.file_objects) % self.size
            
            if self.rank < remainder:
                start_idx = self.rank * (files_per_process + 1)
                local_files = self.file_objects[start_idx:start_idx + files_per_process + 1]
            else:
                start_idx = remainder * (files_per_process + 1) + (self.rank - remainder) * files_per_process
                local_files = self.file_objects[start_idx:start_idx + files_per_process]
            
            for f in local_files:
                gridcells = load(f)
                for gridcell in gridcells:
                    gridcell.run_counter = 0
                    gridcell.outputs = {}
                    gridcell.metacomm_output = {}
                    gridcell.executed_iterations = []
                    gridcell.out_dir = self.output_path / Path(f"grd_{gridcell.xyname}")
                    os.makedirs(gridcell.out_dir, exist_ok=True)
                
                with open(f, 'wb') as fh:
                    dump(gridcells, fh, compress=self.compression)
                    fh.flush()
        
        elif self.gridcells and self.is_root:
            for gridcell in self.gridcells:
                gridcell.run_counter = 0
                gridcell.outputs = {}
                gridcell.metacomm_output = {}
                gridcell.executed_iterations = []
                gridcell.out_dir = self.output_path / Path(f"grd_{gridcell.xyname}")
                os.makedirs(gridcell.out_dir, exist_ok=True)
        
        comm.Barrier()
    

    def get_mask(self)->np.ndarray:
        """Get region mask"""
        return self.grid_mask
    

    def get_crs(self) -> Dict[str, Union[str, int, float]]:
        """Get CRS information"""
        return {
            'proj4': f'{self.config.crs.proj4}',
            'lats' : self.lats,
            'lons' : self.lons,
            'epsg' : self.epsg,
            'datum': self.datum,
            'lat_units': self.config.crs.lat_units,
            'lon_units': self.config.crs.lon_units,
            'lat_zero' : self.config.crs.lat_zero,
            'lon_zero' : self.config.crs.lon_zero,
        }
    

    def __getitem__(self, idx:int):
        """Get gridcell by index (root only)"""
        if not self.is_root:
            return None
            
        if self.gridcells:
            if idx < 0 or idx >= len(self.gridcells):
                raise IndexError(f"Index {idx} out of range")
            return self.gridcells[idx]
        else:
            if not self.file_objects:
                raise ValueError("No file objects found")
            if idx < 0 or idx >= self.region_size:
                raise IndexError(f"Index {idx} out of range")
            return self.load_gridcell(idx)
    

    def __len__(self):
        """Get number of gridcells"""
        if self.gridcells:
            return len(self.gridcells)
        return self.region_size
    

    def __iter__(self):
        """Iterate over gridcells (root only)"""
        if not self.is_root:
            return
            
        if self.gridcells:
            yield from self.gridcells
        else:
            if not self.file_objects:
                raise ValueError("No file objects found")
            
            if not self.gridcell_address_file_objects:
                self._build_gridcell_address_mapping()
            
            for f in self.file_objects:
                with open(f, 'rb') as fh:
                    gridcells_chunk = load(fh)
                    yield from gridcells_chunk
    

    def _build_gridcell_address_mapping(self):
        """Build gridcell address mapping (root only)"""
        if not self.is_root:
            return
            
        if self.gridcell_address_file_objects:
            return
        
        self.gridcell_address_file_objects = {}
        global_idx = 0
        
        for f in self.file_objects:
            with open(f, 'rb') as fh:
                gridcells_chunk = load(fh)
                for local_idx in range(len(gridcells_chunk)):
                    self.gridcell_address_file_objects[global_idx] = (local_idx, f)
                    global_idx += 1
    

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'gridcells') and self.gridcells:
                for gridcell in self.gridcells:
                    if hasattr(gridcell, 'outputs'):
                        gridcell.outputs.clear()
                    if hasattr(gridcell, 'metacomm_output'):
                        gridcell.metacomm_output.clear()
        except Exception as e:
            if self.is_root:
                print(f"Error during cleanup: {e}")