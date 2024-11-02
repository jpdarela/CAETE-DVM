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
import multiprocessing as mp
import os

from pathlib import Path

from typing import Callable, Dict, List,Tuple, Union

import numpy as np
from numpy.typing import NDArray

from caete import str_or_path, get_co2_concentration, read_bz2_file, print_progress, grd_mt
from config import Config, fetch_config
import metacommunity as mc

# Tuples with hydrological parameters for the soil water calculations
from parameters import hsoil, ssoil, tsoil

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
        self.config: Config = fetch_config("caete.toml")
        # self.nproc = self.config.multiprocessing.nprocs # type: ignore
        self.name = Path(name)
        self.co2_path = str_or_path(co2)
        self.co2_data = get_co2_concentration(self.co2_path)

        # IO
        self.climate_files = []
        self.input_data = str_or_path(clim_data)
        self.soil_data = copy.deepcopy(soil_data)
        self.pls_table = mc.pls_table(pls_table)

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


    def update_dump_directory(self, new_name:str="copy"):
        """Update the output folder for the region

        Args:
            output_folder (Union[str, Path]): _description_
        """
        self.name = Path(f"{new_name}")
        self.output_path = output_path / self.name # Update region output folder path
        os.makedirs(self.output_path, exist_ok=True)

        for gridcell in self.gridcells:
            # Update the output folder for each gridcell
            gridcell.run_counter = 0
            gridcell.outputs = {}
            gridcell.metacomm_output = {}
            gridcell.executed_iterations = []

            gridcell.out_dir  = self.output_path/Path(f"grd_{gridcell.xyname}")
            os.makedirs(gridcell.out_dir, exist_ok=True)


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

        if co2 is not None:
            for gridcell in self.gridcells:
                gridcell.change_input(self.input_data, self.stime, self.co2_data)
        else:
            for gridcell in self.gridcells:
                gridcell.change_input(self.input_data, self.stime)

        if input_folder is not None:
            self.climate_files = []
            for file_path in sorted(list(self.input_data.glob("input_data_*-*.pbz2"))):
                self.climate_files.append(file_path)

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
            gridcell_dump_directory = self.output_path/Path(f"grd_{y}-{x}") # The gridcell folder
            grd_cell = grd_mt(y, x, gridcell_dump_directory, self.get_from_main_table)
            # grd_cell = grd_mt(y, x, grd_cell.grid_filename, self.get_from_main_table)
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
        cpu_count = mp.cpu_count()
        num_gridcells = len(self.gridcells)
        num_workers = min(cpu_count, num_gridcells)
        with mp.Pool(processes=num_workers, maxtasksperchild=1) as p:
            self.gridcells = p.map(func, self.gridcells, chunksize=1)
        return None


    def run_region_starmap(self, func:Callable, args):
        """_summary_

        Args:
            func (Callable): _description_
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        cpu_count = mp.cpu_count()
        num_gridcells = len(self.gridcells)
        num_workers = min(cpu_count, num_gridcells)
        with mp.Pool(processes=num_workers, maxtasksperchild=1) as p:
            self.gridcells = p.starmap(func, [(gc, args) for gc in self.gridcells], chunksize=1)
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
                              'metacomm_output',
                              'run_counter',
                              'x',
                              'xres',
                              'xyname',
                              'y',
                              'yres',
                              'grid_filename'}

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
