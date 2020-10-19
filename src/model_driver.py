#       ____    _    _____ _____ _/\__
#      / ___|  / \  | ____|_   _| ____|
#     | |     / _ \ |  _|   | | |  _|
#     | |___ / ___ \| |___  | | | |___
#      \____/_/   \_\_____| |_| |_____|
import os
import pickle as pk
import multiprocessing as mp
import numpy as np

from caete import grd
from caete import mask, npls, print_progress
import plsgen as pls

import bz2
import _pickle as pkl

try:
    os.mkdir('./outputs')
except:
    os.rmdir('./outputs')
    os.mkdir('./outputs')

# FUNCTIONAL TRAITS DATA
pls_table = pls.table_gen(npls)

# Open historical CLIMATE data
with open('../input/climate/pr.pkl', 'rb') as fpr,\
        open('../input/climate/ps.pkl', 'rb') as fps,\
        open('../input/climate/rsds.pkl', 'rb') as frsds,\
        open('../input/climate/tas.pkl', 'rb') as ftas,\
        open('../input/climate/hurs.pkl', 'rb') as fhurs:
    pr = pk.load(fpr)
    ps = pk.load(fps)
    rsds = pk.load(frsds)
    tas = pk.load(ftas)
    hurs = pk.load(fhurs)

dt1 = pr, ps, rsds, tas, hurs

# Open soil Stuff
tn = np.load('../input/soil/total_n_PA.npy')
tp = np.load('../input/soil/total_p.npy')
ap = np.load('../input/soil/avail_p.npy')
ip = np.load('../input/soil/inorg_p.npy')
op = np.load('../input/soil/org_p.npy')

nut = tn, tp, ap, ip, op

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


def apply_spin(grid):
    w, ll, cwd, rl, lnc = grid.bdg_spinup()
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


# def apply_spinup(grid):
#     grid.run_spinup('19710101', '19811231', spinup=30, coupled=False)
#     return grid


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
    print("SPINUP...")
    with mp.Pool(processes=1) as p:
        result = p.map(apply_spin, grid_mn)

    # with mp.Pool(processes=2) as p:
    #     result1 = p.map(apply_spin, result)

    print("MODEL EXEC")
    with mp.Pool(processes=1) as p:
        result1 = p.map(apply_fun, result)
