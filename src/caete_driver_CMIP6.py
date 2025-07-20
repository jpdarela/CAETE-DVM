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
from dataframes import output_manager



# Adapted from the caete_driver.py script to run the model with CMIP6 data.

# This is a script that exemplify the usage of the new implementation of the CAETÊ model.
# Please, refer to the summary section in caete.py for more information.

# We do everything after the if __name__ == "__main__": statement.
# This avoids the duplication of data placed before the if __name__ == "__main__":
# statement when using multiprocessing in Python with the spawn method.
# Please refer to the Python multiprocessing library documentation for more information.


if __name__ == "__main__":

    import time
    time_start = time.time()

    from metacommunity import pls_table
    from parameters import tsoil, ssoil, hsoil
    from region import region
    from worker import worker

    # Force spawn method to avoid issues with multiprocessing use with threading in Linux
    # This statement is awways necessary when running the model. Specifically, it needs to be
    # always after the if __name__ == "__main__": statement. This is a Python requirement.
    mp.set_start_method('spawn', force=True)
    fn: worker = worker()

    input_path = Path("../input/").resolve()

    # Input files. The model will look for the input files in these folders.
    piControl_files = input_path / "MPI-ESM1-2-HR/piControl_cities/"
    hist_files = input_path / "MPI-ESM1-2-HR/historical_cities/"
    ssp370_files = input_path / "MPI-ESM1-2-HR/ssp370_cities/"
    ssp585_files = input_path / "MPI-ESM1-2-HR/ssp585_cities/"


    # Read CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path_hist = input_path / "co2/historical_CO2_annual_1765-2024.csv"
    co2_path_ssp370 = input_path / "co2/ssp370_CO2_annual_2015-2100.csv"
    co2_path_ssp585 = input_path / "co2/ssp585_CO2_annual_2015-2100.csv"


    # Soil hydraulic parameters e.g., wilting point(RWC), field capacity(RWC), water saturation(RWC), etc.
    soil_tuple = tsoil, ssoil, hsoil

    # Read PLS table. The model expects csv file created with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. This table represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-9999.csv"))

    # Create the region using the piControl climate files and historical CO2 data
    region_name = "cities_MPI-ESM1-2-HR_hist" # Name of the run (the outputs of this region will be saved in this folder).

    r = region(region_name,
               piControl_files,
               soil_tuple,
               co2_path_hist,
               main_table)

    # Start gridcells
    # r.set_gridcells()

    # spinup ----------------------------------------------
    # Pre industrial control input data range 1801-2050
    # Spinup using 1801-1900 climatology with pre-industrial co2 concentration ~278.05ppm
    r.run_region_map(fn.spinup_cmip6)  # type: ignore
    # ------------------End of the spinup--------------------------------

    # Save initial state right after spinup - This will be used to run the piControl simulation
    piControl_state = Path(f"./{region_name}_piControl_1900.psz")
    fn.save_state_zstd(r, piControl_state)

    # Transient run with historical data
    # # Update the input source for the transient run - historical files
    print("\nUpdate input and run historical")
    r.update_input(hist_files)  # type: ignore

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2014, 30)  # type: ignore
    for period in run_breaks:  # type: ignore
        print(f"Running period {period[0]} - {period[1]}")  # type: ignore
        r.run_region_starmap(fn.transient_run_brk, period)  # type: ignore
    # End of historical simulation

    # Save region state file. This will be used to run the ssp370 and ssp585 simulations.
    state_file = Path(f"./{region_name}.psz")
    print(f"\n\nSaving state file as {state_file}")
    fn.save_state_zstd(r, state_file)

    # Clean the model state, this state file is used to access model outputs
    r.clean_model_state()
    fn.save_state_zstd(r, Path(f"./{region_name}_output.psz"))
    del r

    # Run ssp370
    r_ssp370:region = fn.load_state_zstd(state_file)
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
    fn.save_state_zstd(r_ssp370, Path(f"{ssp370_out}_output.psz"))

    # ssp585
    r_ssp585:region = fn.load_state_zstd(state_file)
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
    fn.save_state_zstd(r_ssp585, Path(f"{ssp585_out}_output.psz"))

    # piControl
    r_piControl:region = fn.load_state_zstd(piControl_state)
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
    fn.save_state_zstd(r_piControl, Path(f"{piControl_out}_output.psz"))

    # END of the simulation
    print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
    print("Saving outputs...")
    output_manager.cities_output()
    print("Outputs saved.")
    print("Saving time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
    print("Simulation finished.")
