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
import copy
import multiprocessing as mp



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

    # Create the region
    region_name = "pan_amazon_MPI-ESM1-2-HR_hist" # Name of the run (the outputs of this region will be saved in this folder). Look at caete.toml

    # Input files. The model will look for the input files in these folders.
    piControl_files = "../input/MPI-ESM1-2-HR/piControl_test/"
    hist_files = "../input/MPI-ESM1-2-HR/historical_test/"
    ssp370_files = "../input/MPI-ESM1-2-HR/ssp370_test/"
    ssp585_files = "../input/MPI-ESM1-2-HR/ssp585_test/"


    # Read CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path_hist = Path("../input/co2/historical_CO2_annual_1765-2024.csv")
    co2_path_ssp370 = Path("../input/co2/ssp370_CO2_annual_2015-2100.csv")
    co2_path_ssp585 = Path("../input/co2/ssp585_CO2_annual_2015-2100.csv")

    # Soil hydraulic parameters wilting point(RWC), field capacity(RWC) and water saturation(RWC)
    soil_tuple = tsoil, ssoil, hsoil

    # Name for the state file. In general you can save a region with gridcells (including input data)
    # in a state file. This file can be used to restart the model from a specific point. Its useful
    # for store a initial state (after spinup) and restart the model from this point.
    state_file = Path(f"./{region_name}_state_file.psz")


    # Read PLS table. The model expects csv file created with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. This table represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-60000.csv"))

    # Create the region using the piControl climate files and historical CO2 data
    r = region(region_name,
               piControl_files,
               soil_tuple,
               co2_path_hist,
               main_table)

    # Start gridcells
    r.set_gridcells()

    # piControl spinup
    # Pre industrial control range 1801-2100
    # Spinup using 1801-1900 climatology
    print("START soil pools spinup")
    r.run_region_map(fn.soil_pools_spinup)

    # Spinup and run
    # Spinup using 1801-1900 climatology
    print("START soil pools spinup")
    r.run_region_map(fn.soil_pools_spinup)

    # Spinup using 1801-1900 climatology
    print("\nSTART community spinup")
    r.run_region_map(fn.community_spinup)

    # Spinup using 1801-1900 climatology
    # Final phase where new PLS are seed
    print("\nSTART community spinup with PLS seed")
    r.run_region_map(fn.env_filter_spinup)

    print("\nSTART final_spinup")
    r.run_region_map(fn.final_spinup)

    # Transient run with historical data

    # # Update the input source to the transient run - historical files
    print("\nUpdate input and run historical")
    r.update_input(hist_files)

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2014, 5)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r.run_region_starmap(fn.transient_run_brk, period)

    # End of historical simulation

    # Copy region for ssp370 and ssp585 runs
    r_ssp370 = copy.deepcopy(r)
    r_ssp585 = copy.deepcopy(r)

    # Save state file used to access the historical model outputs and export it to other formats.
    r.clean_model_state()
    fn.save_state_zstd(r, Path(f"./{region_name}_result.psz"))

    # Update the input source to the transient run - ssp370 files
    print("\nUpdate input to ssp370")
    r_ssp370.update_input(ssp370_files)

    # Update region dump directory ssp370
    region_name = "pan_amazon_MPI-ESM1-2-HR-ssp370"
    r_ssp370.update_dump_directory("pan_amazon_MPI-ESM1-2-HR_ssp370")




    # # # Create a copy of the region to run counterclim
    # #
    # # r_copy = copy.deepcopy(r)
    # # r_copy.update_input(counterclim_files)
    # # r_copy.update_dump_directory(output_path/Path(f"./{region_name}"), "counterclim")

    # # print("\nSTART transient run - COUNTERCLIM")
    # # run_breaks = fn.create_run_breaks(1901, 2021, 20)
    # # for period in run_breaks:
    # #     print(f"Running period {period[0]} - {period[1]}")
    # #     r_copy.run_region_starmap(fn.transient_run_brk, period)

    # # # Save state after spinup. This state file can be used to restart the model from this point.
    # # print(f"\n\nSaving state file as {state_file}")
    # # fn.save_state_zstd(r, state_file)

    # # Update the input source to the transient run - obsclim files
    # print("\nUpdate input and run obsclim")
    # # r.update_input(obsclim_files)

    # # print("\n\nSTART transient run")
    # # run_breaks = fn.create_run_breaks(1901, 2021, 5)
    # # for period in run_breaks:
    # #     print(f"Running period {period[0]} - {period[1]}")
    # #     r.run_region_starmap(fn.transient_run_brk, period)

    # # final_state:
    # # We clean the state of the gridcells to save the final state of the region
    # # THis final state is not useful to restart the model, but it is useful to
    # # access the model outputs and export it to other formats.
    # r.clean_model_state()
    # fn.save_state_zstd(r, Path(f"./{region_name}_result.psz"))

    # print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
