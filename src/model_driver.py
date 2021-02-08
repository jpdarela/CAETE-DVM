#       ____    _    _____ _____ _/\__
#      / ___|  / \  | ____|_   _| ____|
#     | |     / _ \ |  _|   | | |  _|
#     | |___ / ___ \| |___  | | | |___
#      \____/_/   \_\_____| |_| |_____|
import os
import _pickle as pkl
import bz2
import copy
import multiprocessing as mp
from os import mkdir
from pathlib import Path
import joblib

from netCDF4 import Dataset
import numpy as np

import caete
from caete import grd, mask, npls, print_progress, run_breaks
import plsgen as pls

from post_processing import write_h5

__author__ = "João Paulo Darela Filho"
__descr__ = """RUN CAETÊ --- TODO"""

FUNCALLS = 0


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
            pass
    return r


sombrero = check_start()

# Water saturation, field capacity & wilting point
# Topsoil
map_ws = np.load("../input/soil/ws.npy")
map_fc = np.load('../input/soil/fc.npy')
map_wp = np.load('../input/soil/wp.npy')

# Subsoil
map_subws = np.load("../input/soil/sws.npy")
map_subfc = np.load("../input/soil/sfc.npy")
map_subwp = np.load("../input/soil/swp.npy")

tsoil = (map_ws, map_fc, map_wp)
ssoil = (map_subws, map_subfc, map_subwp)

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

# # Create the gridcell objects
if sombrero:
    # Running in all gridcells of mask
    grid_mn = []
    for Y in range(360):
        for X in range(720):
            if not mask[Y, X]:
                grid_mn.append(grd(X, Y))

else:
    grid_mn = []
    for Y in range(168, 171):
        for X in range(225, 228):
            if not mask[Y, X]:
                grid_mn.append(grd(X, Y))


def apply_init(grid):
    grid.init_caete_dyn(input_path, stime, co2_data, pls_table, tsoil, ssoil)
    return grid


# # START GRIDCELLS
print("Starting gridcells")
print_progress(0, len(grid_mn), prefix='Progress:', suffix='Complete')
for i, g in enumerate(grid_mn):
    apply_init(g)
    print_progress(i + 1, len(grid_mn), prefix='Progress:', suffix='Complete')

# # APPLY AN "ANALYTICAL" SPINUP - IT is a pre-spinup filling of soil organic pools


def apply_spin(grid):
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


# Make a model spinup
def apply_fun(grid):
    grid.run_caete('19790101', '20101231', spinup=1,
                   fix_co2='1999', save=False, nutri_cycle=False)
    return grid


def apply_fun0(grid):
    grid.run_caete('19790101', '20101231', spinup=9,
                   fix_co2='1979', save=False)
    return grid


def apply_fun1(grid):
    dates = run_breaks[0]
    grid.run_caete(dates[0], dates[1])
    return grid


def apply_fun2(grid):
    dates = run_breaks[1]
    grid.run_caete(dates[0], dates[1])
    return grid


def apply_fun3(grid):
    dates = run_breaks[2]
    grid.run_caete(dates[0], dates[1])
    return grid


def apply_fun4(grid):
    dates = run_breaks[3]
    grid.run_caete(dates[0], dates[1])
    return grid


def apply_fun5(grid):
    dates = run_breaks[4]
    grid.run_caete(dates[0], dates[1])
    return grid


def apply_fun6(grid):
    dates = run_breaks[5]
    grid.run_caete(dates[0], dates[1])
    return grid


# Garbage collection
#
del pls_table
del co2_data
del stime


if __name__ == "__main__":

    output_path = Path("../outputs").resolve()

    if output_path.exists():
        pass
    else:
        mkdir(output_path)

    import time

    n_proc = mp.cpu_count() // 2 if not sombrero else 128

    fh = open('logfile.log', mode='w')

    def applyXy(fun, input):
        global FUNCALLS
        FUNCALLS += 1
        fh.writelines(f"MODEL EXEC - {FUNCALLS} - \n",)
        print(f"MODEL EXEC - RUN {FUNCALLS}")
        with mp.Pool(processes=n_proc) as p:
            result = p.map(fun, input)
        end_spinup = time.time() - start
        fh.writelines(f"MODEL EXEC - spinup coup END after (s){end_spinup}\n",)
        return result

    fh.writelines(time.ctime(),)
    fh.writelines("\n\n",)
    fh.writelines("SPINUP...",)
    start = time.time()
    print("SPINUP...")

    # SOIL SPINUP
    with mp.Pool(processes=n_proc) as p:
        _spinup_ = p.map(apply_spin, grid_mn)
    end_spinup = time.time() - start
    fh.writelines(f"END_OF_SPINUP after (s){end_spinup}\n",)
    del grid_mn

    # MAIN SPINUP
    result = applyXy(apply_fun, _spinup_)
    del _spinup_

    result1 = applyXy(apply_fun0, result)
    del result

    # Save Ground 0
    with open("RUN0.pkz", 'wb') as fh2:
        print("Saving gridcells with init state in RUN0.pkz")
        joblib.dump(result1, fh2, compress=('zlib', 9), protocol=4)

    # Apply MODEL
    result2 = applyXy(apply_fun1, result1)
    del result1

    result3 = applyXy(apply_fun2, result2)
    del result2

    result4 = applyXy(apply_fun3, result3)
    del result3

    result5 = applyXy(apply_fun4, result4)
    del result4

    result6 = applyXy(apply_fun5, result5)
    del result5

    result7 = applyXy(apply_fun6, result6)
    del result6

    fh.close()

    print("Salvando db")
    write_h5()
