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
from config import fetch_config

import sys
if sys.platform == "win32":
    from config import fortran_runtime, update_sys_pathlib
    update_sys_pathlib(fortran_runtime)

from caete import grd_mt

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
    def spinup(gridcell:grd_mt):
        """Spin up
        """
        # PHASE 1 - Initial Spinup - equilibrium with pre-industrial conditions. No nutrient cycling,
        # pre-industrial CO2, and PLS sampling
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc="1765",
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True)
        # Glacial cycle CO2
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc=190.0,
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True)
        # Interglacial cycle CO2
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc=280.0,
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True)

        # Final phase without resetting the community and without adding new PLS
        # pre-industrial CO2
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)
        # Transfer to the second half of spinclim
        gridcell.run_gridcell("1801-01-01", "1850-12-31", fixed_co2_atm_conc=None, save=False, nutri_cycle=True)

        return gridcell


    @staticmethod
    def spinup_cmip6(gridcell:grd_mt):
        """Spin up
        """
        verb = False
        # Spin up the model to attain equilibrium in the community and soil pools.
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc="1765",
                              save=False, nutri_cycle=False, reset_community=True, verbose=verb)
        # # Glacial cycle
        # gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=3, fixed_co2_atm_conc=190.0,
        #                       save=False, nutri_cycle=False, reset_community=True, env_filter=True, verbose=verb)
        # Interglacial cycle
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=6, fixed_co2_atm_conc=280.0,
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True, verbose=verb)
        # # Glacial cycle
        # gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc=190.0,
        #                       save=False, nutri_cycle=True, reset_community=False, env_filter=False, verbose=verb)
        # Interglacial cycle
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc="1850",
                              save=False, nutri_cycle=True, reset_community=False, env_filter=False, verbose=verb)
        # Final phase without resetting the community and without adding new PLS
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True)

        return gridcell

    @staticmethod
    def spinup_cmip6_fast(gridcell:grd_mt):
        """Spin up
        """
        verb = False
        # Spin up the model to attain equilibrium in the community and soil pools.
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc="1765",
                              save=False, nutri_cycle=False, reset_community=True, verbose=verb)
        # # Glacial cycle
        # gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=3, fixed_co2_atm_conc=190.0,
        #                       save=False, nutri_cycle=False, reset_community=True, env_filter=True, verbose=verb)
        # Interglacial cycle
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc=280.0,
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True, verbose=verb)
        # # # Glacial cycle
        # # gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc=190.0,
        # #                       save=False, nutri_cycle=True, reset_community=False, env_filter=False, verbose=verb)
        # # Interglacial cycle
        # gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=2, fixed_co2_atm_conc="1850",
        #                       save=False, nutri_cycle=True, reset_community=False, env_filter=False, verbose=verb)
        # Final phase without resetting the community and without adding new PLS
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=1, fixed_co2_atm_conc="1901",
                              save=False, nutri_cycle=True)

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
    def test_run(gridcell:grd_mt):
        """transfer from spinup to transient run. Run along transclim

        Args:
            gridcell (grd_mt): Gridcell object
            interval (Tuple[str, str]): Tuple with the start and end date of the interval: Date format "YYYYMMDD"

        """
        start_date, end_date = "18910101", "19001231"
        gridcell.run_gridcell(start_date, end_date, fixed_co2_atm_conc=None, save=True, nutri_cycle=True)

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
                              save=True, nutri_cycle=True, process_limitation=True)

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
                              save=True, nutri_cycle=True, process_limitation=True)

        return gridcell



    @staticmethod
    def save_state_zstd(region: Any, fname: Union[str, Path]):
        """Save a python serializable object using zstd compression with available threads

        Args:
            region (Any): python object to be compressed and saved. Must be serializable
            fname (Union[str, Path]): filename to save the compressed object

        Returns:
            None: None
        """
        with open(fname, 'wb') as fh:
            compressor = zstd.ZstdCompressor(level=3, threads=-1)
            with compressor.stream_writer(fh) as compressor_writer:
                pkl.dump(region, compressor_writer)


    @staticmethod
    def load_state_zstd(fname:Union[str, Path]):
        """Used to load a region object from a zstd compressed file.
        This method is used to load the state of the region from a compressed file.
        It uses the zstd library to decompress the file and then loads the object using pickle.
        If the file is not found in the current directory, it will look for it in the output directory.

        Args:
            fname (Union[str, Path]): filename of the compressed object
        """
        try:
            with open(fname, 'rb') as fh:
                decompressor = zstd.ZstdDecompressor()
                with decompressor.stream_reader(fh) as decompressor_reader:
                    region = pkl.load(decompressor_reader)
        except FileNotFoundError:
            with open(config.output.output_dir / fname, 'rb') as fh:
                decompressor = zstd.ZstdDecompressor()
                with decompressor.stream_reader(fh) as decompressor_reader:
                    region = pkl.load(decompressor_reader)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {fname} not found in the current directory or output directory.")
        except Exception as e:
            raise Exception(f"Error loading state from {fname}: {e}")
        return region
