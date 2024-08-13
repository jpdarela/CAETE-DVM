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

import multiprocessing as mp
from pathlib import Path


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
    from caete import region, worker

    # Force spawn method to avoid issues with multiprocessing use with threading in Linux
    # This statement is awways necessary when running the model. Specifically, it needs to be
    # always after the if __name__ == "__main__": statement. This is a Python requirement.
    mp.set_start_method('spawn', force=True)
    fn: worker = worker()

    # Create the region
    region_name = "region_test" # Name of the run (the outputs of thi region will be saved in this folder). Look at caete.toml

    # Point to the input files. The model will look for the input files in this folder.
    input_files = "../input/test_input"

    # Soil hydraulic parameters wilting point(RWC), field capacity(RWC) and water saturation(RWC)
    soil_tuple = tsoil, ssoil, hsoil

    # Name for the state file. In general you can save a region with gridcells (including input data)
    # in a state file. This file can be used to restart the model from a specific point. Its useful
    # for store a initial state (after spinup) and restart the model from this point.
    state_file = Path(f"./{region_name}_state.psz")

    # Read CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path = Path("../input/co2/historical_CO2_annual_1765_2018.txt")

    # Read PLS table. The model expects csv file createt with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. This table represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))

    r = region(region_name,
               input_files,
               soil_tuple,
               co2_path,
               main_table)

    # Start gridcells
    r.set_gridcells()

    # # Spinup and run
    print("START soil pools spinup")
    r.run_region_map(fn.soil_pools_spinup)


    print("\n\nSTART community spinup")
    r.run_region_map(fn.community_spinup)

    print("\n\nSTART community spinup with PLS seed")
    r.run_region_map(fn.env_filter_spinup)

    print("\n\nFinal community spinup")
    r.run_region_map(fn.final_spinup)

    # # Save state after spinup. This state file can be used to restart the model from this point.
    # print(f"\n\nSaving state file as {state_file}")
    # fn.save_state_zstd(r, state_file)

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2016, 10)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r.run_region_starmap(fn.transient_run_brk, period)

    # final_state:
    # We clean the state of the gridcells to save the final state of the region
    # THis final state is not useful to restart the model, but it is useful to
    # access the model outputs and export it to other formats.
    r.clean_model_state()
    fn.save_state_zstd(r, Path(f"./{region_name}_result.psz"))

    print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
