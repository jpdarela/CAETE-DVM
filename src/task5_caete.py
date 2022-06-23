# task5_caete.py

# Generate data for the environmental vulnerability index calculation

import os
import pickle as pkl
import copy
import bz2
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import numpy as np 
import joblib

import caete as mod
# import plsgen as pls

assert mod.gp.npls == 2000, "Compile for 2000 PLS"
NPLS = mod.gp.npls
MODS = ['GFDL-ESM2M','HadGEM2-ES','IPSL-CM5A-LR','MIROC5']
SCEN = ['historical','rcp26','rcp60','rcp85']

# Open file with centroids and other metadata from Task5 cities
coord = pd.read_csv("../input/task5/task5_coordinates.csv", index_col="NM_MUNICIP")

RUN_NAME = int(input("  1 for hist_obs or 2 for CMIP5_ISIMIP2b: "))

PLS_TABLE = np.load("./pls_attrs_TASK5.npy")


def read_pkz(pkz):
    with open (pkz, 'rb') as fh:
        dt = joblib.load(fh)
    return dt


def pkz2csv(pkz, rpath, mod, scen):

    from cftime import num2pydate
    spin_dt = read_pkz(pkz)
    s = num2pydate(spin_dt['sind'], spin_dt['time_unit'], spin_dt['calendar'])
    e = num2pydate(spin_dt['eind'], spin_dt['time_unit'], spin_dt['calendar'])
    start = s.strftime("%Y%m%d")
    end   = e.strftime("%Y%m%d") 
    idxT1 = pd.date_range(start, end, freq='D', closed=None)
    idx = idxT1.to_series()
    area = spin_dt['area']
    n = area.shape[0]
    PLS_NAMES = np.array([f'PLS-{i}' for i in range(n)])
    
    df = pd.DataFrame(area.T, columns=PLS_NAMES)
    df.index = idx
    df.to_csv(os.path.join(rpath, mod, f"{scen}.csv"), index_label='date')
    
    
def read_soil_data():
    global PLS_TABLE
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

    # Hydraulics (Waiting for the introdution of gabi's work)
    theta_sat = np.load("../input/hydra/theta_sat.npy")
    psi_sat = np.load("../input/hydra/psi_sat.npy")
    soil_texture = np.load("../input/hydra/soil_text.npy")

    hsoil = (theta_sat, psi_sat, soil_texture)
    return PLS_TABLE, tsoil, ssoil, hsoil


def read_hist_obs():
    s_data = Path("../input/task5/hist_obs").resolve()
    clim_metadata = Path(os.path.join(s_data, "ISIMIP_HISTORICAL_METADATA.pbz2"))

    # Load time related metadata 
    with bz2.BZ2File(clim_metadata, mode='r') as fh:
        clim_metadata = pkl.load(fh)

    stime = copy.deepcopy(clim_metadata[0])
    del clim_metadata
        # # # open co2 data
    with open(os.path.join(s_data, "historical_CO2_annual_1765_2018.txt")) as fh:
        co2_data = fh.readlines()
    pt, tsoil, ssoil, hsoil = read_soil_data()
    return s_data, stime, co2_data, pt, tsoil, ssoil, hsoil


def read_CMIP5_run(mod_name, scen_name):
    s_data = Path(f"../input/task5/CMIP5_ISIMIP2b/{mod_name}/{mod_name}_{scen_name}").resolve()
    clim_metadata = Path(os.path.join(s_data, f"{mod_name}-{scen_name}_METADATA.pbz2"))
    # Load time related metadata 
    with bz2.BZ2File(clim_metadata, mode='r') as fh:
        clim_metadata = pkl.load(fh)

    stime = copy.deepcopy(clim_metadata[0])
    del clim_metadata
        # # # open co2 data
    with open(os.path.join(Path(f"../input/task5/CMIP5_ISIMIP2b/{mod_name}"), f"co2-{mod_name}-{scen_name}.txt")) as fh:
        co2_data = fh.readlines()
    pt, tsoil, ssoil, hsoil = read_soil_data()
    return s_data, stime, co2_data, pt, tsoil, ssoil, hsoil

        
# Define helper/wrapper functions for multiprocessing (parallel execution)

# Pre spinup phase -> soil and vegetation pools
def apply_spin(grid: mod.grd)-> mod.grd:
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19010101", end_date="19030101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


# Entire model spinup (10 * 30 years; fixed co2; no nutrients dynamics - FIXED from input)
def spinup_no_cycle(grid: mod.grd)-> mod.grd:
    grid.run_caete('19010101', '19300101', 10,
                   '1901', False, False, False)
    return grid


# Entire model spinup (10 * 30 years; fixed co2; with nutrients dynamics)
def spinup(grid: mod.grd)-> mod.grd:
    grid.run_caete('19010101', '19300101', 10,
                   '1901', False, True, False)
    return grid

# Transient run (1901-2016) Observed - historcal
def transient_run_hist_obs(grid: mod.grd)-> mod.grd:
    grid.run_caete('19010101', '20161231', 1,
                   None, True, True, False)
    return grid

# transient run (1901-2005) Simulated - historical by CMIP5 MOdels
def transient_run_hist_cmip5(grid: mod.grd)-> mod.grd:
    grid.run_caete('19010101', '20051231', 1,
                   None, True, True, False)
    return grid

# Transient + Projected (2006-2099)
def proj_run_cmip5(grid: mod.grd)-> mod.grd:
    grid.run_caete('20060101', '20991231', 1,
                   None, True, True, False)
    return grid


if __name__ == "__main__":
    
    if RUN_NAME == 1:
        # Run observed historical CAETÊ simulation
        data = read_hist_obs()
        GRD_CELLS = []
        # Create gridcells that lies in the municipality polygon centroid
        for i in coord.index:
            x, y = int(coord.loc[[str(i)]].xindex), int(coord.loc[[str(i)]].yindex)
            GRD_CELLS.append(mod.grd(x, y, str(i)))

            # Start the gridcells 
        for grd in GRD_CELLS:
            grd.init_caete_dyn(*data)
            
        # Use pools of processes to run each phase for all locations
        # PHASE 1: Pre spinup
        with mp.Pool(processes=4) as p:
            GRD_CELLSa = p.map(apply_spin, GRD_CELLS)

        # PHASE 2: SPINUP 1 (300y)
        with mp.Pool(processes=4) as p:
            GRD_CELLSb = p.map(spinup_no_cycle, GRD_CELLSa)

        # PHASE 3: SPINUP 2 (300y, FINAL) GRD_CELLS_IC with initial conditions  
        with mp.Pool(processes=4) as p:
            GRD_CELLS_IC = p.map(spinup, GRD_CELLSb)
        
        # save INIT STATE (SENSITIVITY EXPÈRIMENTS):
        INITIAL_CONDITIONS = []
        for grd in GRD_CELLS_IC:
            folder = Path(os.path.join(grd.out_dir, (grd.plot_name + "_IC.pkz")))
            INITIAL_CONDITIONS.append([grd.plot_name, folder])
            with open(folder, mode='wb') as fh:
                joblib.dump(grd, folder)
            

        with mp.Pool(processes=4) as p:
            GRD_CELLS_hist_obs = p.map(transient_run_hist_obs, GRD_CELLS_IC)
        for grd in GRD_CELLS_hist_obs:
            fpathin = f"../outputs/{grd.plot_name}"
            fname = f"{grd.grid_filename}/spin01.pkz"
            pkz2csv(os.path.join(fpathin, fname), fpathin, f"{grd.grid_filename}" , 'hist_obs')

    elif RUN_NAME == 2:
        
        mod_name = int(input(f" 1 - {MODS[0]};\n 2 - {MODS[1]};\n 3 - {MODS[2]};\n 4 - {MODS[3]}\n\t_:")) -1
        m = MODS[mod_name]
        
        # historical
        data = read_CMIP5_run(m, 'historical')
        
        GRD_CELLS = []
        # Create gridcells that lies over the municipality polygon centroid
        for i in coord.index:
            x, y = int(coord.loc[[str(i)]].xindex), int(coord.loc[[str(i)]].yindex)
            GRD_CELLS.append(mod.grd(x, y, str(i)))

            # Start the gridcells 
        for grd in GRD_CELLS:
            grd.init_caete_dyn(*data)
            
        # Use pools of processes to run each phase for all locations
        # PHASE 1: Pre spinup
        with mp.Pool(processes=4) as p:
            GRD_CELLSa = p.map(apply_spin, GRD_CELLS)

        # PHASE 2: SPINUP 1 (300y)
        with mp.Pool(processes=4) as p:
            GRD_CELLSb = p.map(spinup_no_cycle, GRD_CELLSa)

        # PHASE 3: SPINUP 2 (300y, FINAL) GRD_CELLS_IC with initial conditions  
        with mp.Pool(processes=4) as p:
            GRD_CELLS_IC = p.map(spinup, GRD_CELLSb)
        
        print("END OF SPINUP")
        print("RUN - TRANSIENT")
        with mp.Pool(processes=4) as p:
            GRD_CELLS_F = p.map(transient_run_hist_cmip5, GRD_CELLS_IC)
            
        c1 = copy.deepcopy(GRD_CELLS_F)
        c2 = copy.deepcopy(GRD_CELLS_F)
        c3 = copy.deepcopy(GRD_CELLS_F)
        c4 = copy.deepcopy(GRD_CELLS_F)
        
        # # Final conditions of historical RUN
        # INITIAL_CONDITIONS = []
        print(f"Save historical = {m}")
        for grd in c1:
            # save the preceeding spin
            fpathin = f"../outputs/{grd.plot_name}"
            fout = os.path.join(fpathin, f"{m}")
            os.makedirs(fout, exist_ok=True)
            fname = f"{grd.grid_filename}/spin01.pkz"
            pkz2csv(os.path.join(fpathin, fname), fpathin, m, 'historical')
            os.system(f"cp {fpathin}/{fname} {fout}/historical.pkz")
            os.system(f"rm -rf {fpathin}/{fname}") 
         
        
        data = read_CMIP5_run(m, 'rcp26')       
        for grd in c2:
            grd.change_clim_input(*data[:3])
        
        with mp.Pool(processes=4) as p:
            GRD_CELLS_26 = p.map(proj_run_cmip5, c2)
        print(f"Save RCP 2.6 = {m}")
        for grd in GRD_CELLS_26:
            fpathin = f"../outputs/{grd.plot_name}"
            fout = os.path.join(fpathin, f"{m}")
            fname = f"{grd.grid_filename}/spin02.pkz"
            pkz2csv(os.path.join(fpathin, fname), fpathin, m, 'rcp26')
            os.system(f"cp {fpathin}/{fname} {fout}/rcp26.pkz") 
            os.system(f"rm -rf {fpathin}/{fname}")
    
        data = read_CMIP5_run(m, 'rcp60')       
        for grd in c3:
            grd.change_clim_input(*data[:3])
        
        with mp.Pool(processes=4) as p:
            GRD_CELLS_60 = p.map(proj_run_cmip5, c3)
        print(f"Save RCP 6.0 = {m}")
        for grd in GRD_CELLS_60:
            fpathin = f"../outputs/{grd.plot_name}"
            fout = os.path.join(fpathin, f"{m}")
            fname = f"{grd.grid_filename}/spin02.pkz"
            pkz2csv(os.path.join(fpathin, fname), fpathin, m, 'rcp60')
            os.system(f"cp {fpathin}/{fname} {fout}/rcp60.pkz") 
            os.system(f"rm -rf {fpathin}/{fname}")
        
        data = read_CMIP5_run(m, 'rcp85')       
        for grd in c4:
            grd.change_clim_input(*data[:3])
        
        with mp.Pool(processes=4) as p:
            GRD_CELLS_85 = p.map(proj_run_cmip5, c4)
        print(f"Save RCP 8.5 = {m}")
        for grd in GRD_CELLS_85:
            fpathin = f"../outputs/{grd.plot_name}"
            fout = os.path.join(fpathin, f"{m}")
            fname = f"{grd.grid_filename}/spin02.pkz"
            pkz2csv(os.path.join(fpathin, fname), fpathin, m, 'rcp85')
            os.system(f"cp {fpathin}/{fname} {fout}/rcp85.pkz")
            os.system(f"rm -rf {fpathin}/{fname}")
