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

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import copy
import multiprocessing as mp
import os

from pathlib import Path
from threading import Thread
from typing import Callable, Dict, List,Tuple, Union
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from joblib import dump, load

from caete import str_or_path, get_co2_concentration, read_bz2_file, print_progress, grd_mt
from config import Config, fetch_config
from parameters import hsoil, ssoil, tsoil
import metacommunity as mc

# Tuples with hydrological parameters for the soil water calculations

# Global lock. Used to lock the access to the main table of Plant Life Strategies
lock = mp.Lock()

from parameters import output_path


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
        self.config: Config = fetch_config(os.path.join(os.path.dirname(__file__), 'caete.toml'))

        # self.nproc = self.config.multiprocessing.nprocs # type: ignore
        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # IO
        self.climate_files = []
        self.input_data = str_or_path(clim_data)
        self.soil_data = copy.deepcopy(soil_data)
        self.pls_table = mc.pls_table(pls_table)
        self.file_objects = []

        # calculate_matrix dimension size from grid resolution
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
        os.makedirs(output_path, exist_ok=True)

        # This is the output path for this region
        self.output_path = output_path/self.name
        os.makedirs(self.output_path, exist_ok=True)

        # A list to store this region's gridcells
        # Some magic methods are defined to deal with this list
        self.gridcells:List[grd_mt] = []


    def load_gridcells(self):
        """Load the gridcells from the intermediate files"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")
        if self.file_objects:
            for f in self.file_objects:
                with open(f, 'rb') as fh:
                    gridcells = load(fh)
                for gridcell in gridcells:
                    self.gridcells.append(gridcell)


    def unload_gridcells(self):
        """Unload the gridcells to the intermediate files"""
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
                dump(chunk, f, compress=("lz4", 1))


    def update_dump_directory(self, new_name:str="copy"):
        """Update the output folder for the region

        Args:
            new_name (Union[str, Path]): name of the new folder where outputs of the region should be saved. Defaults to "copy"
        """
        self.name = Path(f"{new_name}")
        self.output_path = output_path / self.name # Update region output folder path
        os.makedirs(self.output_path, exist_ok=True)
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

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
                dump(gridcells, fh, compress=("lz4", 1))

        with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
            executor.map(process_file, self.file_objects)

        # for f in self.file_objects:
        #     gridcells = load(f)

        #     for gridcell in gridcells:
        #         # Update the output folder for each gridcell
        #         gridcell.run_counter = 0
        #         gridcell.outputs = {}
        #         gridcell.metacomm_output = {}
        #         gridcell.executed_iterations = []

        #         gridcell.out_dir  = self.output_path/Path(f"grd_{gridcell.xyname}")
        #         os.makedirs(gridcell.out_dir, exist_ok=True)
        #     dump(gridcells, f, compress=("lz4", 1))


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

        if co2 is not None:
            self.co2_path = str_or_path(co2)
            self.co2_data = get_co2_concentration(self.co2_path)

        if input_folder is not None:
            # Read the climate data
            self.input_data = str_or_path(input_folder)
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

        # if not self.file_objects:
        #     raise ValueError("No file objects found. Cannot read intermediate files")

        # for f in self.file_objects:
        #     gridcells = load(f)
        #     if co2 is not None:
        #         for gridcell in gridcells:
        #             gridcell.change_input(self.input_data, self.stime, self.co2_data)
        #     else:
        #         for gridcell in gridcells:
        #             gridcell.change_input(self.input_data, self.stime)
        #     dump(gridcells, f, compress=("lz4", 1))

        if not self.file_objects:
            raise ValueError("No file objects found. Cannot read intermediate files")

        def process_file(f):
            try:
                with open(f, 'rb') as file:
                    gridcells:grd_mt = load(file)
                if co2 is not None:
                    for gridcell in gridcells:
                        gridcell.change_input(self.input_data, self.stime, self.co2_data)
                else:
                    for gridcell in gridcells:
                        gridcell.change_input(self.input_data, self.stime)
                with open(f, 'wb') as file:
                    dump(gridcells, file, compress=("lz4", 1))
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                raise e

        with ThreadPoolExecutor(max_workers=len(self.file_objects)) as executor:
            executor.map(process_file, self.file_objects)


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
            idx = np.random.choice(self.npls_main_table, 1, replace=False)[0]
            with lock:
                return idx, self.pls_table.table[:, idx]

        assert comm_npls <= self.npls_main_table, "Number of PLS must be less than the number of PLS in the main table"

        idx = np.random.choice(self.npls_main_table, comm_npls, replace=False)
        with lock:
            return idx, self.pls_table.table[:, idx]


    # def set_gridcells(self):
    #     """_summary_
    #     """
    #     print("Starting gridcells")
    #     i = 0
    #     print_progress(i, len(self.yx_indices), prefix='Progress:', suffix='Complete')
    #     for f,pos in zip(self.climate_files, self.yx_indices):
    #         y, x = pos
    #         gridcell_dump_directory = self.output_path/Path(f"grd_{y}-{x}") # The gridcell folder
    #         grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
    #         # grd_cell = grd_mt(y, x, grd_cell.grid_filename, self.get_from_main_table)
    #         grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
    #                                 tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
    #         self.gridcells.append(grd_cell)
    #         self.lats[i] = grd_cell.lat
    #         self.lons[i] = grd_cell.lon
    #         print_progress(i+1, len(self.yx_indices), prefix='Progress:', suffix='Complete')
    #         i += 1
    #         # Print data about the model execution: number of metacommunities, number of gridcells, etc.
    #     print(f"Number of gridcells: {len(self.gridcells)}")
    #     print(f"Number of metacommunities: {self.config.metacomm.n}") # type: ignore
    #     print(f"Maximum number of PLS per community: {self.config.metacomm.npls_max}") # type: ignore


    def run_region_map(self, func: Callable):
        """Run a function across all gridcells using multiprocessing.Pool"""

        result = []

        if self.file_objects:
            j = 0
            print_progress(j, len(self.file_objects), prefix='Progress:', suffix='Complete')
            # Read file objects and do the provessing
            for f in self.file_objects:
                with open(f, 'rb') as fh:
                    new_chunk = load(fh)
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
                result = []
                with open(f, 'wb') as fh:
                    dump(new_chunk, fh, compress=("lz4", 1))
                print_progress(j+1, len(self.file_objects), prefix='Progress:', suffix='Complete')
                j += 1

        else:
            # First run -> no file objects
            jobs = list(zip(self.climate_files, self.yx_indices))
            cpu_count = self.config.multiprocessing.max_processes # type: ignore
            chunks = [jobs[i:i + cpu_count] for i in range(0, len(jobs), cpu_count)]
            i = 0
            j = 0
            print_progress(j, len(chunks), prefix='Progress:', suffix='Complete')
            for chunk in chunks:
            # prepare jobs for processing
                for f, pos in chunk:
                    y, x = pos
                    gridcell_dump_directory = self.output_path/Path(f"grd_{y}-{x}")
                    grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
                    grd_cell.set_gridcell(f, stime_i=self.stime, co2=self.co2_data,
                                            tsoil=tsoil, ssoil=ssoil, hsoil=hsoil)
                    result.append(grd_cell)
                    self.lats[i] = grd_cell.lat
                    self.lons[i] = grd_cell.lon
                    i += 1
                # Save the intermediate file
                fname = self.output_path/Path(f"region_file{uuid4()}.lz4")
                self.file_objects.append(fname)
                num_workers = min(self.config.multiprocessing.max_processes, len(result))
                with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                    try:
                        result = pool.map(func, result, chunksize=1)
                    except Exception as e:
                        print(f"Error during multiprocessing: {e}")
                        pool.terminate()
                        raise
                    finally:
                        pool.close()
                        pool.join()
                with open(fname, 'wb') as f:
                    dump(result, f, compress=("lz4", 1))
                result = []
                print_progress(j+1, len(chunks), prefix='Progress:', suffix='Complete')
                j += 1


    def run_region_starmap(self, func: Callable, args):
        """Run a function with arguments across all gridcells using multiprocessing.Pool"""
        if not self.file_objects:
            raise ValueError("No file objects found. Cannot run starmap without file objects")
        j = 0
        print_progress(j, len(self.file_objects), prefix='Progress:', suffix='Complete')
        for f in self.file_objects:
            with open(f, 'rb') as fh:
                new_chunk = load(fh)
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
            with open(f, 'wb') as fh:
                dump(result, fh, compress=("lz4", 1))
            print_progress(j+1, len(self.file_objects), prefix='Progress:', suffix='Complete')
            j += 1


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
                              'metacomm_output',
                              'run_counter',
                              'x',
                              'xres',
                              'xyname',
                              'y',
                              'yres',
                              'grid_filename',
                              'file_objects'}



        for gridcell in self:
            all_attributes = set(gridcell.__dict__.keys())

            # # Delete attributes that are not in the subset
            for attr in all_attributes - attributes_to_keep:
                delattr(gridcell, attr)
        # self.unload_gridcells()


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
        self.load_gridcells()
        yield from self.gridcells


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
        finally:
            # Explicitly release the global lock if necessary
            global lock
            if lock:
                lock = None
