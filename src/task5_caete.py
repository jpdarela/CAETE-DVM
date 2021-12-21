# task5_caete.py

# Generate data for the environmental vulnerability index calculation

import os
import _pickle as pkl
import copy
import bz2
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import numpy as np 
import caete as mod
import plsgen as pls

coord = pd.read_csv("../input/task5/task5_coordinates.csv", index_col="NM_MUNICIP")


s_data = Path("../input/task5").resolve()
clim_metadata = Path(os.path.join(s_data, "ISIMIP_HISTORICAL_METADATA.pbz2"))

with bz2.BZ2File(clim_metadata, mode='r') as fh:
    clim_metadata = pkl.load(fh)

stime = copy.deepcopy(clim_metadata[0])
del clim_metadata
# # open co2 data
with open(os.path.join(s_data, "co2/historical_CO2_annual_1765_2018.txt")) as fh:
    co2_data = fh.readlines()

# Soil Parameters
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

# Hydraulics
theta_sat = np.load("../input/hydra/theta_sat.npy")
psi_sat = np.load("../input/hydra/psi_sat.npy")
soil_texture = np.load("../input/hydra/soil_text.npy")

hsoil = (theta_sat, psi_sat, soil_texture)

GRD_CELLS = []

for i in coord.index:
    x, y = int(coord.loc[[str(i)]].xindex), int(coord.loc[[str(i)]].yindex)
    GRD_CELLS.append(mod.grd(x, y, str(i)))

PLS_TABLE = pls.table_gen(mod.npls, Path('./'))

for grd in GRD_CELLS:
    grd.init_caete_dyn(s_data, stime, co2_data, PLS_TABLE, tsoil, ssoil, hsoil)

def apply_spin(grid):
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19010101", end_date="19030101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid

def spinup_no_cycle(grid):
    grid.run_caete('19010101', '19300101', 10, '1901', False, False, False)
    return grid    

def spinup(grid):
    grid.run_caete('19010101', '19300101', 10, '1901', False, True, False)
    return grid

def transient_run(grid):
    grid.run_caete('19010101', '20161231', 1, None, True, True, False)
    return grid

if __name__ == "__main__":
    
    with mp.Pool(processes=3) as p:
        GRD_CELLSa = p.map(apply_spin, GRD_CELLS)

    with mp.Pool(processes=3) as p:
        GRD_CELLSb = p.map(spinup_no_cycle, GRD_CELLSa)

    with mp.Pool(processes=3) as p:
        GRD_CELLSc = p.map(spinup, GRD_CELLSb)

    with mp.Pool(processes=3) as p:
        GRD_CELLSd = p.map(transient_run, GRD_CELLSc)
