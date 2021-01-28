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

from netCDF4 import Dataset
import numpy as np

import caete
from caete import grd
from caete import npls, print_progress
import plsgen as pls

__author__ = "João Paulo Darela Filho"
__descr__ = """RUN CAETÊ --- TODO"""


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
dt_ws = Dataset('../input/soil/WS.nc').variables['WS'][:]
map_ws = np.flipud(dt_ws.__array__())
dt_fc = Dataset('../input/soil/FC.nc').variables['FC'][:]
map_fc = np.flipud(dt_fc.__array__())
dt_wp = Dataset('../input/soil/WP.nc').variables['WP'][:]
map_wp = np.flipud(dt_wp.__array__())

# Subsoil
dt_subws = Dataset('../input/soil/S_WS.nc').variables['WS'][:]
map_subws = np.flipud(dt_subws.__array__())
dt_subfc = Dataset('../input/soil/S_FC.nc').variables['FC'][:]
map_subfc = np.flipud(dt_subfc.__array__())
dt_subwp = Dataset('../input/soil/S_WP.nc').variables['WP'][:]
map_subwp = np.flipud(dt_subwp.__array__())

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


# for x in range(10000):
#     grid_mn[0]._update_pool(np.random.randint(1, 5), np.random.randint(1, 5))

# APPLY AN "ANALYTICAL" SPINUP - IT is a pre-spinup filling of soil organic pools
def apply_spin(grid):
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


# Make a model spinup
def apply_fun(grid):
    grid.run_caete('19790101', '20101231', spinup=2, fix_co2='1999')
    return grid


# RUn
def apply_fun1(grid):
    grid.run_caete('19790101', '20101231', spinup=1)
    return grid


def apply_fun2(grid):
    grid.run_caete('20110101', '20151231', fix_co2=610)
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

    n_proc = mp.cpu_count() // 2 if not sombrero else 55

    fh = open('logfile.log', mode='w')

    fh.writelines(time.ctime(),)
    fh.writelines("\n\n",)
    fh.writelines("SPINUP...",)
    start = time.time()
    print("SPINUP...")

    with mp.Pool(processes=n_proc) as p:
        result = p.map(apply_spin, grid_mn)
    end_spinup = time.time() - start
    fh.writelines(f"END_OF_SPINUP after (s){end_spinup}\n",)
    del grid_mn

    fh.writelines("MODEL EXEC - MAIN SPINUP",)
    print("MODEL EXEC - spinup")
    with mp.Pool(processes=n_proc) as p:
        result1 = p.map(apply_fun, result)
    end_spinup = time.time() - start
    del result  # clean memory
    fh.writelines(f"MODEL EXEC - spinup deco END after (s){end_spinup}\n",)

    fh.writelines("MODEL EXEC - RUN\n",)
    print("MODEL EXEC - RUN")
    with mp.Pool(processes=n_proc) as p:
        result2 = p.map(apply_fun1, result1)
    end_spinup = time.time() - start
    del result1
    fh.writelines(f"MODEL EXEC - spinup coup END after (s){end_spinup}\n",)
    fh.close()

    # a = apply_spin(grid_mn[2])
    # print('A = OK')
    # b = apply_fun(a)
    # del a
    # print('B = OK - DEBUG')
    # c = apply_fun1(b)
    # # # del b
    # # # d = apply_fun2(c)
    # # # del c
