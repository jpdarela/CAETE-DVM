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


import pickle as pkl
from pathlib import Path
from typing import Tuple, Union, Any
import zstandard as zstd
from caete import grd_mt
from config import fetch_config

config = fetch_config()


class worker:

    """Worker functions used to run the model in parallel and other utilities"""

    @staticmethod
    def create_run_breaks(start_year:int, end_year:int, interval:int):
        """Create run breaks for the model
        Args:
            start_year (int): Start year of the run
            end_year (int): End year of the run
            interval (int): Interval in years for the run breaks
        Returns:
            run_breaks_hist (list): List of tuples with the start and end date of the intervals
        """
        run_breaks_hist = []
        current_year = start_year

        # Create intervals
        while current_year + interval - 1 <= end_year:
            start_date = f"{current_year}0101"
            end_date = f"{current_year + interval - 1}1231"
            run_breaks_hist.append((start_date, end_date))
            current_year += interval

        # Adjust the last interval if it is not uniform
        if current_year <= end_year:
            start_date = f"{current_year}0101"
            end_date = f"{end_year}1231"
            run_breaks_hist.append((start_date, end_date))

        return run_breaks_hist


    @staticmethod
    def soil_pools_spinup(gridcell:grd_mt):
        """Spin to attain equilibrium in soil pools, In this phase the communities are reset if there are no PLS

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 200 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc="1765",
                              save=False, nutri_cycle=False, reset_community=True)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc="1765",
                              save=False, nutri_cycle=False, reset_community=True)

        return gridcell

    @staticmethod
    def soil_pools_spinup_glacial(gridcell:grd_mt):
        """Spin to attain equilibrium in soil pools, In this phase the communities are reset if there are no PLS

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 500 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc=190.0,
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc=190.0,
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True)

        return gridcell

    @staticmethod
    def soil_pools_spinup_interglacial(gridcell:grd_mt):
        """Spin to attain equilibrium in soil pools, In this phase the communities are reset if there are no PLS

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 500 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc=280.0,
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc=280.0,
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True)

        return gridcell


    @staticmethod
    def quit_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in the community without adding new PLS

        Spinup time: 500 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)

        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=5, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)

        return gridcell


    @staticmethod
    def run_spinup_transer(gridcell: grd_mt):
        """Run the model in the first half of spinclim.
        THe result from here will be tranfered to translcim run

        run length: 50  years

        gridcell.run_gridcell("1801-01-01", "1850-12-31", fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)
        """

        gridcell.run_gridcell("1801-01-01", "1850-12-31", fixed_co2_atm_conc=None, save=False, nutri_cycle=True)

        return gridcell


    @staticmethod
    def transclim_run(gridcell:grd_mt):
        """transfer from spinup to transient run. Run along transclim

        Args:
            gridcell (grd_mt): Gridcell object
            interval (Tuple[str, str]): Tuple with the start and end date of the interval: Date format "YYYYMMDD"

        """
        start_date, end_date = "18510101", "19001231"
        gridcell.run_gridcell(start_date, end_date, fixed_co2_atm_conc=None, save=False, nutri_cycle=True)

        return gridcell


    @staticmethod
    def transient_run_brk(gridcell:grd_mt, interval:Tuple[str, str]):
        """transient run

        Args:
            gridcell (grd_mt): Gridcell object
            interval (Tuple[str, str]): Tuple with the start and end date of the interval: Date format "YYYYMMDD"


        """
        start_date, end_date = interval
        gridcell.run_gridcell(start_date, end_date, spinup=0, fixed_co2_atm_conc=None,
                              save=True, nutri_cycle=True)

        return gridcell


    @staticmethod
    def transient_piControl_brk(gridcell:grd_mt, interval:Tuple[str, str]):
        """transient run

        Args:
            gridcell (grd_mt): Gridcell object
            interval (Tuple[str, str]): Tuple with the start and end date of the interval: Date format "YYYYMMDD"


        """
        start_date, end_date = interval
        gridcell.run_gridcell(start_date, end_date, spinup=0, fixed_co2_atm_conc="1901",
                              save=True, nutri_cycle=True)

        return gridcell



    @staticmethod
    def save_state_zstd(region: Any, fname: Union[str, Path]):
        """Save apython serializable object using zstd compression with 12 threads

        Args:
            region (Any): python object to be compressed and saved. Must be serializable
            fname (Union[str, Path]): filename to save the compressed object

        Returns:
            None: None
        """
        with open(fname, 'wb') as fh:
            compressor = zstd.ZstdCompressor(level=22, threads=12)
            with compressor.stream_writer(fh) as compressor_writer:
                pkl.dump(region, compressor_writer)


    @staticmethod
    def load_state_zstd(fname:Union[str, Path]):
        """Used to load a region object from a zstd compressed file

        Args:
            fname (Union[str, Path]): filename of the compressed object
        """
        with open(fname, 'rb') as fh:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(fh) as decompressor_reader:
                region = pkl.load(decompressor_reader)
        return region
