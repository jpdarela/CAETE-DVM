from pathlib import Path

from caete import region, worker
from metacommunity import read_pls_table
from parameters import *


fn = worker()

if __name__ == "__main__":
    # Read CO2 data
    co2_path = Path("../input/co2/historical_CO2_annual_1765_2018.txt")
    main_table = read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))

    r = region("region_test",
                   "../input/central",
                   (tsoil, ssoil, hsoil),
                   co2_path,
                   main_table)
    r.set_gridcells()

    # Spinup and run
    r.run_region(fn.soil_pools_spinup)
    r.run_region(fn.community_spinup)
    r.run_region(fn.final_spinup)
    r.run_region(fn.transient_run)
