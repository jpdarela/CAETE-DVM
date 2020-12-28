#       ____    _    _____ _____ _/\__
#      / ___|  / \  | ____|_   _| ____|
#     | |     / _ \ |  _|   | | |  _|
#     | |___ / ___ \| |___  | | | |___
#      \____/_/   \_\_____| |_| |_____|
import pickle as pk
import multiprocessing as mp
from os import mkdir
from pathlib import Path
import numpy as np

from caete import grd
from caete import mask, npls, print_progress
import plsgen as pls


# FUNCTIONAL TRAITS DATA
pls_table = pls.table_gen(npls)

# open co2 data
with open("../input/co2/historical_CO2_annual_1765_2018.txt") as fh:
    co2_data = fh.readlines()


# # Create the gridcell objects
grid_mn = [grd(238, 183), grd(238, 184), grd(238, 185), grd(238, 186)]


def apply_init(grid):
    grid.init_caete_dyn(dt1, nut, co2_data, pls_table, 'TESTE')
    return grid


# # START GRIDCELLS
print("Starting gridcells")
print_progress(0, len(grid_mn), prefix='Progress:', suffix='Complete')
for i, g in enumerate(grid_mn):
    apply_init(g)
    print_progress(i + 1, len(grid_mn), prefix='Progress:', suffix='Complete')


# APPLY AN "ANALYTICAL" SPINUP - IT is a pre-spinup filling of soil organic pools
def apply_spin(grid):
    w, ll, cwd, rl, lnc = grid.bdg_spinup()
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def apply_fun(grid):
    grid.run_spinup('19810101', '20101231', spinup=20, coupled=True)
    return grid


# Garbage collection
#
del pls_table
del pr
del ps
del rsds
del tas
del hurs
del dt1
del tn
del tp
del ap
del ip
del op
del nut
del co2_data

if __name__ == "__main__":

    n_proc = 2
    output_path = Path("./outputs")

    if output_path.exists():
        pass
    else:
        mkdir(output_path)

    print("SPINUP...")
    with mp.Pool(processes=n_proc) as p:
        result = p.map(apply_spin, grid_mn)

    print("MODEL EXEC")
    with mp.Pool(processes=n_proc) as p:
        result1 = p.map(apply_fun, result)
