#       ____    _    _____ _____ _/\__
#      / ___|  / \  | ____|_   _| ____|
#     | |     / _ \ |  _|   | | |  _|
#     | |___ / ___ \| |___  | | | |___
#      \____/_/   \_\_____| |_| |_____|
import os
import pickle as pkl
import bz2
import copy
import multiprocessing as mp
from os import mkdir
from pathlib import Path
import numpy as np

from caete import grd
from caete import mask, npls, print_progress
import plsgen as pls


def check_start():
    while True:
        i = input("---RUN IN SOMBRERO(y/n): ")
        if i == 'y':
            r = True
            break
        elif i == 'n':
            r = False
            break
        else:
            print("---RUN IN SOMBRERO(y/n):")
    return r


sombrero = check_start()

# Select the location of input climate and soil data (for each grid cell )
if sombrero:
    s_data = Path("/home/amazonfaceme/shared_data").resolve()
    clim_and_soil_data = Path("HISTORICAL-RUN")
else:
    s_data = Path("../input").resolve()
    clim_and_soil_data = Path("caete_input")


# Shared data among grid cells
#
input_path = Path(os.path.join(s_data, clim_and_soil_data))

# Open time attributes
clim_metadata = Path(os.path.join(s_data, clim_and_soil_data,
                                  "ISIMIP_HISTORICAL_METADATA.pbz2"))
with bz2.BZ2File(clim_metadata, mode='r') as fh:
    clim_metadata = pkl.load(fh)

stime = copy.deepcopy(clim_metadata[0])
del clim_metadata

# open co2 data
with open(os.path.join(s_data, "co2/historical_CO2_annual_1765_2018.txt")) as fh:
    co2_data = fh.readlines()

# FUNCTIONAL TRAITS DATA
pls_table = pls.table_gen(npls)

mask = np.load(os.path.join(s_data, "mask/mask_raisg-360-720.npy"))

# # Create the gridcell objects
if sombrero:
    # Running in all gridcells of mask
    grid_mn = []
    for Y in range(360):
        for X in range(720):
            if not mask[Y, X]:
                grid_mn.append(grd(X, Y))

else:
    grid_mn = [grd(238, 183), grd(238, 184), grd(238, 185), grd(238, 186)]
    assert len(grid_mn) < 5, "To many gridcells to run in your PC, Change" \
        "the code in the source file to erase this assertion"


def apply_init(grid):
    grid.init_caete_dyn(input_path, stime, co2_data, pls_table)
    return grid


# # START GRIDCELLS
print("Starting gridcells")
print_progress(0, len(grid_mn), prefix='Progress:', suffix='Complete')
for i, g in enumerate(grid_mn):
    apply_init(g)
    print_progress(i + 1, len(grid_mn), prefix='Progress:', suffix='Complete')


# # APPLY AN "ANALYTICAL" SPINUP - IT is a pre-spinup filling of soil organic pools
def apply_spin(grid):
    w, ll, cwd, rl, lnc = grid.bdg_spinup()
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def apply_fun(grid):
    grid.run_spinup('19010101', '19301231', spinup=5, coupled=False)
    return grid


def apply_fun1(grid):
    grid.run_spinup('19010101', '19301231', spinup=10, coupled=True)
    return grid


def apply_fun2(grid):
    grid.run_spinup('19010101', '20151231', spinup=1, coupled=True)
    return grid


# # Garbage collection
# #
del pls_table
del co2_data
del stime

if __name__ == "__main__":

    if sombrero:
        n_proc = 64
    else:
        n_proc = 2

    output_path = Path("./outputs")

    if output_path.exists():
        pass
    else:
        mkdir(output_path)

    print("SPINUP...")
    with mp.Pool(processes=n_proc) as p:
        result = p.map(apply_spin, grid_mn)

    print("MODEL EXEC - spinup deco")
    with mp.Pool(processes=n_proc) as p:
        result1 = p.map(apply_fun, result)
    del result

    print("MODEL EXEC - spinup coup")
    with mp.Pool(processes=n_proc) as p:
        result2 = p.map(apply_fun1, result1)
        del result1

        print("MODEL EXEC- RUN")
    with mp.Pool(processes=n_proc) as p:
        result3 = p.map(apply_fun2, result2)
        del result2
