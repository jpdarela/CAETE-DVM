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
import os
from pathlib import Path

from polars import read_csv

# This is a script that exemplify the usage of the new implementation of the CAETÊ model.
# Please, refer to the summary section in caete.py for more information.

# We do everything after the if __name__ == "__main__": statement.
# This avoids the duplication of data placed before it when using multiprocessing in Python with the spawn method.
# Please refer to the Python multiprocessing library documentation for more information.

# Profiling is enabled by default if the environment variable CAETE_PROFILING is set to True.
# The results of the profiling at this level are not very useful because most of the time is spent
# in the region and worker methods. However, it can be useful to identify bottlenecks in the
# overall execution of the model. Most of the activity captured in this profiling is related waiting time of parallel processes and threads.
# In the end of the caete.py module there is another profiling implementation that provides more detailed information about
# the execution of the model at the gridcell level. It may be outdated. Please refer to the comments there for more information.
PROFILING = False or os.environ.get("CAETE_PROFILING", "False").lower() in ("true", "1", "yes")

if PROFILING:
    import cProfile
    import io
    import pstats

if __name__ == "__main__":

    if PROFILING:
        # Create a profile object
        print("Profiling is enabled. This will slow down the execution of the model.")
        pr = cProfile.Profile()
        pr.enable()

    import time

    from metacommunity import pls_table
    from parameters import hsoil, ssoil, tsoil
    from region import region
    from worker import worker


    time_start = time.time()
    # Force spawn method to avoid issues with multiprocessing use with threading in Linux
    # This statement is always necessary when running the model. Specifically, it needs to be
    # always after the if __name__ == "__main__": statement. This is a Python requirement.
    mp.set_start_method('spawn', force=True)

    # Create a worker instance to access the gridcell-wise functions and other utilities
    fn: worker = worker()

    # Name of the region. This name will be used to create the output folder.
    region_name = "pan_amazon_hist" # Name of the run (the outputs of this region will be saved in this folder). Look at caete.toml

    # Paths to input data
    obsclim_files = "../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc"
    spinclim_files = "../input/20CRv3-ERA5/spinclim/caete_input_20CRv3-ERA5_spinclim.nc"

    # Gridlists control which gridcells will be used in the simulation. In the grd folder there
    # are some examples of gridlists that can be used to run the model in different regions or
    # with different number of gridcells.
    gridlist = read_csv("../grd/gridlist_test.csv") # Small test gridlist n=16
    # gridlist = read_csv("../grd/gridlist_pa.csv") # Pan-Amazon gridlist n=2726
    # gridlist = read_csv("../grd/gridlist_random_cells_pa.csv") # Random sample of 128 gridcells in the Pan-Amazon region
    # gridlist = read_csv("../grd/gridlist_pan_amazon_05d_FORESTS_MAPBIOMASS_2000.csv") # Pan-Amazon gridlist with only tropical forest cells n=2080

    # Soil hydraulic parameters, e.g.,  wilting point(RWC), field capacity(RWC) and water saturation(RWC) for soil layers
    # tsoil = # Top soil
    # ssoil = # Sub soil
    # hsoil = # Parameter used in Gabriela's model
    soil_tuple = tsoil, ssoil, hsoil

    #CO2 atmospheric data. The model expects a formated table in a text file with
    # exactly 2 columns (year, co2 concentration) separetd by a space, a coma, a semicolon etc.
    # A header is optional. The model also expects annual records in ppm (parts per million).
    co2_path = Path("../input/co2/historical_CO2_annual_1765-2024.csv")

    # Read PLS table. The model expects csv file created with the table_gen defined in
    # the plsgen.py script. This table contains the global PLS definitions. We also refer to
    # this table as main table. it represents all possible plant functional types
    # that can be used in the model. The model will use this table to create (subsample)
    # the metacommunities. Everthing is everywhere, but the environment selects.
    PLS_TABLE_PATH = Path("./PLS_MAIN/pls_attrs-200000.csv")
    assert PLS_TABLE_PATH.exists(), f"PLS table not found at {PLS_TABLE_PATH.resolve()}"

    main_table = pls_table.read_pls_table(PLS_TABLE_PATH)

    # Create the region using the spinup climate files
    print("creating region with spinclim files")
    pan_amazon = region(region_name,
                        spinclim_files,
                        soil_tuple,
                        co2_path,
                        main_table,
                        gridlist=gridlist)

    # print(f"Region {region_name} created with {len(r.gridcells)} gridcells")
    # Spinup and run
    print("START soil pools spinup")
    s1 = time.perf_counter()

    # Run the spinup. run_region_map maps the function fn.spinup to all gridcells in the region.
    pan_amazon.run_region_map(fn.spinup)

    e1 = time.perf_counter()
    print(f"Spinup time: {(e1 - s1) // 60 :.0f}:{(e1 - s1) % 60:.0f}")

    # Run the model
    s3 = time.perf_counter()

    # run transclim
    print("\n\nSTART transclim run")
    pan_amazon.run_region_map(fn.transclim_run)

    e3 = time.perf_counter()
    print(f"Transclim run: {(e3 - s3) // 60 :.0f}:{(e3 - s3) % 60:.0f}")

    # # # Save state after spinup.
    # # This state file can be used to restart the model from this point. 1900-12-31
    state_file = Path(f"./{region_name}_after_spinup_state_file.psz")
    print(f"\n\nSaving state file as {state_file}")
    s4 = time.perf_counter()

    pan_amazon.save_state(state_file)

    e4 = time.perf_counter()
    print(f"State file save time: {(e4 - s4) // 60 :.0f}:{(e4 - s4) % 60 :.0f}")

    # Set a new state for the pan_amazon region. This is necessary to
    # update the model state with the new input data (obsclim file).
    # If this step is ommited, the model will continue to use the old state, modifying it with the new input data.
    # This is not desired, as we would like to use that state after the spinup to start other transient runs with different forcings.
    print(f"\n\nExecution time so far: ", (time.time() - time_start) / 60, " minutes", end="\n\n")
    print("setting new state")
    s5 = time.perf_counter()

    pan_amazon.set_new_state()

    e5 = time.perf_counter()
    print(f"Set new state time: {(e5 - s5) // 60 :.0f}:{(e5 - s5) % 60 :.0f}")

    # # Update the input source to the transient run - we now use the obsclim file
    print("Uodating input to obsclim files")
    s2 = time.perf_counter()

    pan_amazon.update_input(obsclim_files)

    e2 = time.perf_counter()
    print(f"Update input time: {(e2 - s2) // 60 :.0f}:{(e2 - s2) % 60:.0f}")

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2024, 30)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        pan_amazon.run_region_starmap(fn.transient_run_brk, period)

    # final_state:
    # We clean the state of the gridcells to save the final state of the region
    # THis final state is not useful to restart the model, but it is useful to
    # access the model outputs and export it to other formats.
    # r.save_state(Path(f"./{region_name}_{period[1]}_final_state.psz"))
    # r.set_new_state()
    print("\nCleaning model state and saving final state file")
    s6 = time.perf_counter()
    pan_amazon.clean_model_state_fast()
    # r.clean_model_state()
    e6 = time.perf_counter()
    print("Time in seconds: " + str(e6 - s6))
    print(f"Clean model state time: {(e6 - s6) // 60 :.0f}:{(e6 - s6) % 60 :.4f}")

    print(f"\n\nSaving final state file as ./{region_name}_result.psz")
    s7 = time.perf_counter()
    pan_amazon.save_state(Path(f"./{region_name}_result.psz"))
    e7 = time.perf_counter()
    print(f"Final state file save time: {(e7 - s7) // 60 :.0f}:{(e7 - s7) % 60 :.0f}")

    print("\n\nExecution time: ", (time.time() - time_start) / 60, " minutes", end="\n\n")

    # Generate outputs - Import here to avoid multiprocessing issues
    from dataframes import output_manager
    output_manager.pan_amazon_output()

    # Copy the PLS table used in the run to the output folder
    from shutil import copy2
    output_folder = Path(f"../outputs")
    copy2(PLS_TABLE_PATH, output_folder / PLS_TABLE_PATH.name)


    if PROFILING:
        # Disable profiling
        pr.disable()

        # Save profiling results to file with region name
        profile_filename = f"{region_name}_profile.prof"
        pr.dump_stats(profile_filename)
        print(f"\nProfiling results saved to: {profile_filename}")
        print("Analyze with: python -m pstats pan_amazon_hist_profile.prof")
        print("Or visualize with: snakeviz pan_amazon_hist_profile.prof")

        # Quick summary to console
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative').print_stats(10)
        print("\nQuick Summary - Top 10 functions by cumulative time:")
        print(s.getvalue())
