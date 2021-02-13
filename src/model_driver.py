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
from h52nc import h52nc

__author__ = "João Paulo Darela Filho"
__descr__ = """RUN CAETÊ"""

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
    for Y in range(168, 175):
        for X in range(225, 230):
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


# DEFINE HARVERSTERS - funcs that will apply grd methods(run the CAETÊ model) over the instanvces
def apply_spin(grid):
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def apply_fun(grid):
    grid.run_caete('19790101', '19891231', spinup=2,
                   fix_co2='1983', save=False, nutri_cycle=False)
    return grid


def apply_fun0(grid):
    grid.run_caete('19790101', '19881231', spinup=45,
                   fix_co2='1983', save=False)
    return grid


def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res


def apply_funX(grid, brk):
    grid.run_caete(brk[0], brk[1])
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
    print("START: ", time.ctime())
    fh.writelines(time.ctime(),)
    fh.writelines("\n\n",)
    fh.writelines("SOIL SPINUP...\n",)
    start = time.time()
    print("SOIL SPINUP...")

    # SOIL SPINUP
    with mp.Pool(processes=n_proc) as p:
        _spinup_ = p.map(apply_spin, grid_mn)
    end_spinup = time.time() - start
    fh.writelines(f"END_OF_SPINUP after (s){end_spinup}\n",)
    del grid_mn

    # MAIN SPINUP
    def applyXy(fun, input):
        global FUNCALLS
        FUNCALLS += 1
        fh.writelines(f"MODEL EXEC - {FUNCALLS} - \n",)
        print(f"MODEL EXEC - RUN {FUNCALLS}")
        with mp.Pool(processes=n_proc) as p:
            # reserve 2 funcalls for the main spinup
            if FUNCALLS > 2:
                result = p.starmap(fun, input)
            else:
                result = p.map(fun, input)
        end_spinup = time.time() - start
        fh.writelines(f"MODEL EXEC - spinup coup END after (s){end_spinup}\n",)
        return result

    print("Applying main spinup. This process can take hours (RUN 1 & 2)")

    # The first 2 calls of applyXy are reserved to the MAIN spinup
    # These 2 calls will use the method map of the Pool(multiprocessing)
    # the remaining calls will use the starmap method
    result = applyXy(apply_fun, _spinup_)
    del _spinup_

    result1 = applyXy(apply_fun0, result)
    del result

    # Save Ground 0
    with open("RUN0.pkz", 'wb') as fh2:
        print("Saving gridcells with init state in RUN0.pkz")
        joblib.dump(result1, fh2, compress=('zlib', 1), protocol=4)

    result = result1
    del result1
    for i, brk in enumerate(run_breaks):
        print(f"Applying model to the interval {brk[0]}-{brk[1]}")
        result = zip_gridtime(result, (brk,))
        result = applyXy(apply_funX, result)

    fh.close()

    print(time.ctime())
    print("Saving db - This will take some hours")
    write_h5()
    print("\n\nSaving netCDF4 files")
    h52nc("../outputs/CAETE.h5")
    print(time.ctime())
