# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from pathlib import Path
import multiprocessing as mp
from typing import Tuple
import time

import polars as pl


from metacommunity import pls_table
from parameters import tsoil, ssoil, hsoil


# Adapted from the caete_driver.py script to run the model with CMIP6 data.

# This is a script that exemplify the usage of the new implementation of the CAETÊ model.
# Please, refer to the summary section in caete.py for more information.

# We do everything after the if __name__ == "__main__": statement.
# This avoids the duplication of data placed before the if __name__ == "__main__":
# statement when using multiprocessing in Python with the spawn method.
# Please refer to the Python multiprocessing library documentation for more information.


if __name__ == "__main__":
    # Force spawn method to avoid issues with multiprocessing use with threading in Linux
    # This statement is awways necessary when running the model. Specifically, it needs to be
    # always after the if __name__ == "__main__": statement. This is a Python requirement.
    mp.set_start_method('spawn', force=True)

    from region import region
    from worker import worker
    from dataframes import output_manager

    fn: worker = worker()

    input_path = Path("../input/").resolve()

    # Input files. The model will look for the input files in these folders.
    piControl_files = input_path / "MPI-ESM1-2-HR/piControl/caete_input_MPI-ESM1-2-HR_piControl.nc"
    hist_files = input_path / "MPI-ESM1-2-HR/historical/caete_input_MPI-ESM1-2-HR_historical.nc"
    ssp370_files = input_path / "MPI-ESM1-2-HR/ssp370/caete_input_MPI-ESM1-2-HR_ssp370.nc"
    ssp585_files = input_path / "MPI-ESM1-2-HR/ssp585/caete_input_MPI-ESM1-2-HR_ssp585.nc"

    gridlist = pl.read_csv("../grd/gridlist_cities.csv")

    # Read CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path_hist = input_path / "co2/historical_CO2_annual_1765-2024.csv"
    co2_path_ssp370 = input_path / "co2/ssp370_CO2_annual_2015-2100.csv"
    co2_path_ssp585 = input_path / "co2/ssp585_CO2_annual_2015-2100.csv"

    # Soil hydraulic parameters, e.g.,  wilting point(RWC), field capacity(RWC) and water saturation(RWC) for soil layers
    # tsoil = # Top soil
    # ssoil = # Sub soil
    # hsoil = # Parameter used in Gabriela's model
    soil_tuple = tsoil, ssoil, hsoil

    # Read PLS table. The model expects csv file created with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. This table represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-30000.csv"))

    # Create the region using the piControl climate files and historical CO2 data
    region_name = "cities_MPI-ESM1-2-HR_hist" # Name of the run (the outputs of this region will be saved in this folder).

    time_start = time.time()
    r = region(region_name,
               piControl_files,
               soil_tuple,
               co2_path_hist,
               main_table,
               gridlist=gridlist)

    # Start gridcells
    # r.set_gridcells()

    # spinup ----------------------------------------------
    # Pre industrial control input data range 1801-2050
    # Spinup using 1801-1900 climatology with pre-industrial co2 concentration ~278.05ppm
    r.run_region_map(fn.spinup_cmip6)  # type: ignore
    # ------------------End of the spinup--------------------------------

    # Save initial state right after spinup - This will be used to run the piControl simulation
    piControl_state = Path(f"./{region_name}_piControl_1900.psz")
    # fn.save_state_zstd(r, piControl_state)

    # Save the state
    r.save_state(piControl_state)  # type: ignore

    # Transient run with historical data
    # Update the input source for the transient run - historical files
    print("\nUpdate input and run historical")
    r.set_new_state() # Set a new state and update inputs
    r.update_input(hist_files)  # type: ignore

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2014, 30)  # type: ignore
    for period in run_breaks:  # type: ignore
        print(f"Running period {period[0]} - {period[1]}")  # type: ignore
        r.run_region_starmap(fn.transient_run_brk, period)  # type: ignore
    # End of historical simulation

    # Save region state file.
    # This will be used to run the ssp370 and ssp585 simulations.
    # Again we save the state defining a new state to avoid overwriting
    # the previous state files.
    state_file = Path(f"./{region_name}.psz") # Historical state file
    r.save_state(state_file)  # type: ignore


    # Clean the model state, We create a new state folder structure to access model outputs
    # We are finished with this region object. No need to keep it in memory.
    r.set_new_state()
    r.clean_model_state()
    output_file = Path(f"./{region_name}_output.psz")
    # The state file for outputs does not need a new state.
    r.save_state(output_file)  # type: ignore
    # fn.save_state_zstd(r, Path(f"./{region_name}_output.psz"))
    del r

    # Run ssp370
    r_ssp370:region = fn.load_state_zstd(state_file)
    r_ssp370.set_new_state()  # Set a new state and update inputsb by copying the previous state.
                              # This avoids overwriting the historical state files.
    print("\nUpdate input to ssp370")
    r_ssp370.update_input(ssp370_files, co2_path_ssp370)  # type: ignore
    # Update region dump directory ssp370
    ssp370_out = "cities_MPI-ESM1-2-HR-ssp370"
    r_ssp370.update_dump_directory(ssp370_out)

    run_breaks = fn.create_run_breaks(2015, 2100, 30)  # type: ignore
    period: Tuple[int, int]
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r_ssp370.run_region_starmap(fn.transient_run_brk, period)

    r_ssp370.clean_model_state()
    ssp370_out = Path(f"{ssp370_out}_output.psz")
    # r_ssp370.delete_state() # Delete the state files to free memory
    r_ssp370.save_state(ssp370_out)  # type: ignore
    del r_ssp370


    # ssp585
    r_ssp585:region = fn.load_state_zstd(state_file)
    r_ssp585.set_new_state()  # Set a new state and update inputs by copying the previous state.

    print("\nUpdate input to ssp585")
    r_ssp585.update_input(ssp585_files, co2_path_ssp585)
    # Update region dump directory ssp585
    ssp585_out = "cities_MPI-ESM1-2-HR-ssp585"
    r_ssp585.update_dump_directory(ssp585_out)

    run_breaks = fn.create_run_breaks(2015, 2100, 30)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r_ssp585.run_region_starmap(fn.transient_run_brk, period)

    r_ssp585.clean_model_state()
    ssp585_out = Path(f"{ssp585_out}_output.psz")
    # r_ssp585.delete_state()  # Delete the state files to free memory
    r_ssp585.save_state(ssp585_out)
    del r_ssp585

    # piControl
    r_piControl:region = fn.load_state_zstd(piControl_state)
    r_piControl.set_new_state()  # Set a new state and update inputs by copying the previous state.

    print("\nUpdate input to piControl")
    r_piControl.update_input(piControl_files, co2_path_hist)
    # Update region dump directory piControl
    piControl_out = "cities_MPI-ESM1-2-HR-piControl"
    r_piControl.update_dump_directory(piControl_out)

    run_breaks = fn.create_run_breaks(1901, 2100, 30)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r_piControl.run_region_starmap(fn.transient_piControl_brk, period)

    r_piControl.clean_model_state()
    piControl_out = Path(f"{piControl_out}_output.psz")
    # r_piControl.delete_state()  # Delete the state files to free memory
    r_piControl.save_state(piControl_out)  # type: ignore
    del r_piControl


    # END of the simulation
    print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
    print("Saving outputs...")
    output_manager.cities_output()
    print("Outputs saved.")
    print("Saving time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
    print("Simulation finished.")
