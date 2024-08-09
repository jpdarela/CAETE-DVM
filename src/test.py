import multiprocessing as mp
from pathlib import Path


# This is a script that exemplify the usage of the CAETÊ model

if __name__ == "__main__":

    from metacommunity import pls_table
    from parameters import tsoil, ssoil, hsoil
    from caete import region, worker


    mp.set_start_method('spawn', force=True) # Force spawn method to avoid issues with threading in Linux
    fn: worker = worker()

    # Create the region
    region_name = "region_test" # Name of the run (the outputs of thi region will be saved in this folder). Look at caete.toml
    input_files = "../input/test_input" # Folder containing inputs
    soil_tuple = tsoil, ssoil, hsoil # Soil water characteristics
    state_file = Path(f"./region_test.psz") # N name for the state file

    # Read CO2 data
    co2_path = Path("../input/co2/historical_CO2_annual_1765_2018.txt")

    # Read PLS
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

    #Save state
    print(f"\n\nSaving state file as {state_file}")
    fn.save_state_zstd(r, state_file)

    print("\n\nSTART transient run")
    run_breaks = fn.create_run_breaks(1901, 2016, 15)
    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        r.run_region_starmap(fn.transient_run_brk, period)
