# Example script to run a region from a saved state file.
# Outdated: THe region and worker modules have been updated, and this script is no longer compatible with the latest version.

from typing import Tuple
from worker import worker
from dataframes import output_manager

if __name__ == "__main__":

    region = worker.load_state_zstd("../outputs/cities_MPI-ESM1-2-HR_hist.psz")  # Load the region from a saved state file.

    region.set_new_state()  # Set a new state and update inputs by copying the previous state.
    region.update_dump_directory("doing_something")

    run_breaks = worker.create_run_breaks(1901, 2014, 10)

    for period in run_breaks:
        print(f"Running period {period[0]} - {period[1]}")
        region.run_region_starmap(worker.transient_run_brk, period)

    region.clean_model_state()
    worker.save_state_zstd(region, "./state.psz2")

    variables_to_read: Tuple[str,...] = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp",
                                    "photo", "npp", "evapm", "lai", "vcmax", "ls",
                                    "specific_la", "rcm")

    region_output = worker.load_state_zstd("./state.psz2")
    # output_manager.generic_text_output_grd(region_output, variables_to_read)

