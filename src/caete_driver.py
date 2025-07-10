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


# This is a script that exemplify the usage of the new implementation of the CAETÊ model.
# Please, refer to the summary section in caete.py for more information.

# We do everything after the if __name__ == "__main__": statement.
# This avoids the duplication of data placed before it when using multiprocessing in Python with the spawn method.
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

    # Name of the region. This name will be used to create the output folder.
    region_name = "pan_amazon_hist" # Name of the run (the outputs of this region will be saved in this folder). Look at caete.toml

    # # Input files. The model will look for the input files in these folders.
    obsclim_files = "../input/20CRv3-ERA5/obsclim/"
    spinclim_files = "../input/20CRv3-ERA5/spinclim/"
    transclim_files = "../input/20CRv3-ERA5/transclim/"
    counterclim_files = "../input/20CRv3-ERA5/counterclim/"

    # Soil hydraulic parameters wilting point(RWC), field capacity(RWC) and water saturation(RWC)
    soil_tuple = tsoil, ssoil, hsoil

    # Read CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path = Path("../input/co2/historical_CO2_annual_1765-2024.csv")

    # Read PLS table. The model expects csv file created with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. it represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-99999.csv"))

    # Create the region using the spinup climate files
    r = region(region_name,
               spinclim_files,
               soil_tuple,
               co2_path,
               main_table)

    # # Spinup and run
    print("START soil pools spinup")
    r.run_region_map(fn.spinup)

    # # Change input source to transclim files 1851-1900
    print("\nSTART transclim run")
    r.update_input(transclim_files)

    # # Run the model
    r.run_region_map(fn.transclim_run)

    # # # Save state after spinup.
    # # This state file can be used to restart the model from this point.
    state_file = Path(f"./{region_name}_after_spinup_state_file.psz")
    print(f"\n\nSaving state file as {state_file}")
    fn.save_state_zstd(r, state_file)

    # # Update the input source to the transient run - obsclim files
    # print("\nUpdate input and run obsclim")
    r.update_input(obsclim_files)

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2021, 30)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r.run_region_starmap(fn.transient_run_brk, period)

    # final_state:
    # We clean the state of the gridcells to save the final state of the region
    # THis final state is not useful to restart the model, but it is useful to
    # access the model outputs and export it to other formats.
    fn.save_state_zstd(r, Path(f"./{region_name}_{period[1]}_final_state.psz"))
    r.clean_model_state()
    fn.save_state_zstd(r, Path(f"./{region_name}_result.psz"))

    print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
