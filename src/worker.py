
import gc
import pickle as pkl
from pathlib import Path
from typing import Tuple, Union, Any, List
import numpy as np
from numpy.typing import NDArray
import zstandard as zstd
from caete import grd_mt

class worker:

    """Worker functions used to run the model in parallel"""

    @staticmethod
    def create_run_breaks(start_year:int, end_year:int, interval:int):
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
        Spinup time: 400 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=False, reset_community=True, kill_and_reset=True)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=10, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=False, reset_community=True, env_filter=True,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def community_spinup(gridcell:grd_mt):
        """Spin to attain equilibrium in the community, In this phase, communities can be reset if there are no PLS

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 400 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True, reset_community=True)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=10, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def env_filter_spinup(gridcell:grd_mt):
        """Spin to attain equilibrium in the community while adding new PLS if there are free slots

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 400 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True, reset_community=True, env_filter=True,
                              verbose=False)
        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True, reset_community=False, env_filter=False,
                              verbose=False)
        gc.collect()
        return gridcell


    @staticmethod
    def final_spinup(gridcell:grd_mt):
        """spin to attain equilibrium in the community while adding new PLS if there are free slots

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        Spinup time: 200 years

        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)        """
        gridcell.run_gridcell("1801-01-01", "1900-12-31", spinup=4, fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)
        gc.collect()
        return gridcell


    @staticmethod
    def spinup_transer(gridcell: grd_mt):
        """Run the model in the first half of spinclim.
        THe result from here will be tranfered to translcim run

        This method uses spinclim data to run the model.
        Check the init and end dates to match input data.
        run length: 50  years

        gridcell.run_gridcell("1801-01-01", "1850-12-31", fixed_co2_atm_conc="1801",
                              save=False, nutri_cycle=True)
        """

        gridcell.run_gridcell("1801-01-01", "1850-12-31", fixed_co2_atm_conc=None, save=False, nutri_cycle=True)
        gc.collect()
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
        gc.collect()
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
        gc.collect()
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
