# ! Copyright 2017- LabTerra

# !     This program is free software: you can redistribute it and/or modify
# !     it under the terms of the GNU General Public License as published by
# !     the Free Software Foundation, either version 3 of the License, or
# !     (at your option) any later version.)

# !     This program is distributed in the hope that it will be useful,
# !     but WITHOUT ANY WARRANTY; without even the implied warranty of
# !     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# !     GNU General Public License for more details.

# !     You should have received a copy of the GNU General Public License
# !     along with this program.  If not, see <http://www.gnu.org/licenses/>.


#       ____    _    _____ _____ _/\__
#      / ___|  / \  | ____|_   _| ____|
#     | |     / _ \ |  _|   | | |  _|
#     | |___ / ___ \| |___  | | | |___
#      \____/_/   \_\_____| |_| |_____|

# Driver script to execute/run CAETÊ-DVM (OFFLINE) with historical Observed and Simulated climate (4 MODELS FROM CMIP5, ISIMIP2b Protocol)

# SPINUP 0 - PHASE 1 grd.spin_bdg <- Start water pools and generate initial estimated CNP fluxes from Vegetation to Soil
# SPINUP 0 - PHASE 2 grd.spin_sdc <- Estimate initial Values of CNP in the soil POOLS (From data generated in PHASE 1)
# SPINUP 1 - PHASE 1 grd.run_caete <- Cycle the model n times with the 1979-1989 climate (11 * n Years) 
#                                     totalizing X years with fixed co2=1980 values [~340 µmol mol⁻¹]
#                                     + Fixed N and P pools - i.e., No cycles of Nutrients
# SPINUP 2 - PHASE 1 grd.run_caete <- Cycles the full model n times with the 1979-1989 climate (11 Years) 11 * n years
# TOTAL SPINUP TIME = 418 years  

README = """SPINUP Breakdown - Initially all PLS receive 0.1 Kg (C) m⁻² for the Fine root and leaf reservoirs.
            The woody tissues start with 1 kg (C) m⁻². Then the model is cycled 4 years (1979-1983) using the method bdg_spinup
            of the grd class (caete.py) to estimate initial values of Water, and the approximate C, N, and P fluxes from vegetation
            to the the soil pools. Then these estimated fluxes are employed to run the sdc_spinup method (i.e. Run the soil cycles submodel)
            3000 times.
            
            After this initial numerical approximation of the soil POOLs of W,C,N,P the model is applied in a STANDARD DVM Spin-up 
            divided into 2 PHASES:
            The numbers can variate. Look the funcion definitions to the exact values
            1 - Cycle the model 10 times with the 1979-1989 climate, total = 110 years, with fixed co2=1980 values
                [~340 µmol mol⁻¹] & Fixed N and P pools - i.e., No cycling of Nutrients (Values fixed from reference data).
            
            2 - Cycles the full model 28 times with the 1979-1989 climate, total = 308, years with fixed co2=1980 values
                [~340 µmol mol⁻¹]. Full cycle of nutrients.
            
            This end up with a set of grd objects with the initial conditions prepared to model experiments
            
            The netCDF4 CF-compliant output generator do not work with data generated in the spinup PHASE.
            Thus, the calls to the grd.run_caete method during the spinup are set with the argument save=False
            Calling this method with save=True in the spinup calls can generate unexpected errors. (Because of the time data/metadata)
            
            This is a script intended to execute model experiments for the Pan-Amazon region in a HPC environment.
            Optionally you can run subsets of study area with ~60 grid points in a personal computer 
            
            The grd class can be easily applied to experiments and tests as shown in the python scripts k43_experiment.py and task5_caete.py in /src
            After the spinup the chosen historical transient run is executed
            """

import os
import _pickle as pkl
import bz2
import copy
import multiprocessing as mp
from pathlib import Path
from random import shuffle

import joblib
from netCDF4 import Dataset
import numpy as np

import caete
from caete import grd, mask, npls, print_progress, rbrk
import plsgen as pls

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


# Check sombrero
sombrero = check_start()

zone = ""
y0, y1 = 0, 0
x0, x1 = 0, 0
folder = "central"


# Water saturation, field capacity & wilting point (maps of 0.5° res)
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



if not sombrero:
    print("Set the folder to store outputs:")
    outf = input(
        "Give a name to your run (ASCII letters and numbers only. No spaces): ")
    dump_folder = Path(f'../outputs/{outf}').resolve()
    nc_outputs = Path(os.path.join(dump_folder, Path("nc_outputs"))).resolve()
    print(
        f"The raw model results & the PLS table will be saved at: {dump_folder}\n")
    print(f"The final netCDF files will be stored at: {nc_outputs}\n")

if not sombrero:
    zone = input("Select a zone [c: central, s: south, e: east, nw: NW]: ")
    if zone in ['c', 's', 'e', 'nw']:
        print("Running in the zone:", zone)
        pass
    else:
        print("Running in the zone: c")
        zone = 'c'

if zone == 'c':
    y0, y1 = 175, 186
    x0, x1 = 235, 241
    folder = "central"

elif zone == 's':
    y0, y1 = 200, 211
    x0, x1 = 225, 231
    folder = "south"

elif zone == 'nw':
    y0, y1 = 168, 175
    x0, x1 = 225, 230
    folder = "north_west"

elif zone == 'e':
    y0, y1 = 190, 201
    x0, x1 = 255, 261
    folder = "east"
else:
    assert sombrero


if sombrero:
    clim_list = ["HISTORICAL-RUN",
                 "GFDL-ESM2M",
                 "HadGEM2-ES",
                 "IPSL-CM5A-LR",
                 "MIROC5"]

    # Select the location of input climate and soil data (for each grid cell )
    CLIM_DATA_str = """
        You have the option to run any of the historical climatologies:

        HISTORICAL-RUN   1
        GFDL-ESM2M       2
        HadGEM2-ES       3
        IPSL-CM5A-LR     4
        MIROC5           5

        Choose one:     _"""
    while True:
        climatology = input(CLIM_DATA_str)
        if climatology in ['1', '2', '3', '4', '5']:
            climatology = int(climatology)
            outf = clim_list[climatology - 1]
            break
        else:
            pass
    # SELECT MODEL -HISTORICAL SPINUP + RUN
    s_data = Path("/home/amazonfaceme/shared_data").resolve()
    model_root = Path(os.path.join(s_data, Path(outf)))
    dump_folder = Path(f'../outputs/{outf}').resolve()
    if climatology == 1:
        clim_metadata = Path(os.path.join(s_data, model_root,
                                          "ISIMIP_HISTORICAL_METADATA.pbz2"))
        input_path = model_root
        with bz2.BZ2File(clim_metadata, mode='r') as fh:
            clim_metadata = pkl.load(fh)

        stime = copy.deepcopy(clim_metadata[0])
        del clim_metadata
        # # open co2 data
        with open(os.path.join(s_data, "co2/historical_CO2_annual_1765_2018.txt")) as fh:
            co2_data = fh.readlines()
        run_breaks = rbrk[0]
        rbrk_index = 0
        # save the attributes table to the HISTORICAL OBSERVED RUN - It will be used in all other experiments
        pls_table = pls.table_gen(npls, dump_folder)

    else:
        clim_and_soil_data = Path(os.path.join(model_root, Path("historical")))
        clim_metadata = Path(os.path.join(
            clim_and_soil_data, f"{outf}-historical_METADATA.pbz2"))
        with bz2.BZ2File(clim_metadata, mode='r') as fh:
            clim_metadata = pkl.load(fh)

        stime = copy.deepcopy(clim_metadata[0])
        del clim_metadata
        input_path = clim_and_soil_data
        # open co2 data
        with open(os.path.join(model_root, f"co2-{outf}-historical.txt")) as fh:
            co2_data = fh.readlines()
        run_breaks = rbrk[1]
        rbrk_index = 1

        # READ the PLS table employed in the base run
        from parameters import pls_path, ATTR_FILENAME
        if pls_path.exists():
            from caete_utils import read_pls_table
            print("Using PLS TABLE from BASE_RUN")
            os.makedirs(dump_folder, exist_ok=True)
            pls_table = read_pls_table(out=Path(os.path.join(dump_folder, ATTR_FILENAME)))
        else:
            print(f"WARNING: Creating a new PLS table for a historical simulated ({outf}) run ")
            pls_table = pls.table_gen(npls, dump_folder)

    with open("stime.txt", 'w') as fh:
        fh.writelines([f"{stime['units']}\n",
                       f"{stime['calendar']}\n",
                       f"{outf}-ISIMIP2b-hist\n",
                       f"{rbrk_index}\n"])
        
    
    
    nc_outputs = Path(os.path.join(dump_folder, Path("nc_outputs"))).resolve()
    print(f"The raw model results & the PLS table will be saved at: {dump_folder}\n")
    print(f"The final netCDF files will be stored at: {nc_outputs}\n")
    
        
else:
    assert not sombrero
    # HISTORICAL OBSERVED DATA
    s_data = Path("../input").resolve()
    clim_and_soil_data = Path(folder)

    input_path = Path(os.path.join(s_data, clim_and_soil_data))
    clim_metadata = Path(os.path.join(s_data, clim_and_soil_data,
                                      "ISIMIP_HISTORICAL_METADATA.pbz2"))
    with bz2.BZ2File(clim_metadata, mode='r') as fh:
        clim_metadata = pkl.load(fh)

    stime = copy.deepcopy(clim_metadata[0])
    del clim_metadata
    # # open co2 data
    with open(os.path.join(s_data, "co2/historical_CO2_annual_1765_2018.txt")) as fh:
        co2_data = fh.readlines()
    run_breaks = rbrk[0]
    rbrk_index = 0

    with open("stime.txt", 'w') as fh:
        fh.writelines([f"{stime['units']}\n",
                       f"{stime['calendar']}\n",
                       f"historical-ISIMIP2b-TEST-{folder}\n",
                       f"{rbrk_index}\n"])
    # FUNCTIONAL TRAITS DATA
    pls_table = pls.table_gen(npls, dump_folder)





# # Create the gridcell objects
if sombrero:
    # Running in all gridcells of mask
    grid_mn = []
    for Y in range(360):
        for X in range(720):
            if not mask[Y, X]:
                grid_mn.append(grd(X, Y, outf))

else:
    grid_mn = []
    for Y in range(y0, y1):
        for X in range(x0, x1):
            if not mask[Y, X]:
                grid_mn.append(grd(X, Y, outf))


def apply_init(grid:grd)->grd: 
    # wraper to the grd method
    grid.init_caete_dyn(input_path, stime, co2_data,
                        pls_table, tsoil, ssoil, hsoil)
    return grid


def chunks(lst, chunck_size):
    shuffle(lst)
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunck_size):
        yield lst[i:i + chunck_size]


# # START GRIDCELLS
print("Starting gridcells")
print_progress(0, len(grid_mn), prefix='Progress:', suffix='Complete')
for i, g in enumerate(grid_mn):
    apply_init(g)
    print_progress(i + 1, len(grid_mn), prefix='Progress:', suffix='Complete')


# DEFINE HARVERSTERS - funcs that will apply grd methods(run the CAETÊ model) over the instanvces
def apply_spin(grid:grd)->grd:
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def apply_fun(grid:grd)->grd:
    grid.run_caete('19790101', '19891231', spinup=5, 
                   fix_co2='1980', save=False, nutri_cycle=False)
    return grid

 
def apply_fun0(grid:grd)->grd:
    grid.run_caete('19790101', '19891231', spinup=35,
                   fix_co2='1980', save=False)
    return grid


def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res


def apply_funX(grid:grd, brk:list)->grd:
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
        from os import mkdir
        mkdir(output_path)

    import time

    from post_processing import write_h5
    from h52nc import h52nc

    n_proc = mp.cpu_count()

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
                # # Divide in chunks to leverage the work
                # result = []
                # for l in chunks(input, n_proc * 2):
                #     r1 = p.map(fun, input)
                # result += r1
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

    # Save Ground 0 # END OF SPINUP
    g0_path = Path(os.path.join(
        dump_folder, Path(f"CAETE_STATE_START_{outf}_.pkz"))).resolve()
    with open(g0_path, 'wb') as fh2:
        print(f"Saving gridcells with init(POST-SPINUP) state in: {g0_path}\n")
        joblib.dump(result1, fh2, compress=('zlib', 1), protocol=4)

    result = result1
    del result1

    # RUNNING THE experiment
    for i, brk in enumerate(run_breaks):
        print(f"Applying model to the interval {brk[0]}-{brk[1]}")
        result = zip_gridtime(result, (brk,))
        result = applyXy(apply_funX, result)

    # Save FINAL STATE (TO feed CMIP5 proj. experiments)
    g1_path = Path(os.path.join(
        dump_folder, Path(f"CAETE_STATE_END_{outf}_.pkz"))).resolve()
    with open(g1_path, 'wb') as fh2:
        print(f"Saving gridcells with END state in: {g1_path}\n")
        joblib.dump(result, fh2, compress=('zlib', 1), protocol=4)

    fh.close()

    print("\nEND OF MODEL EXECUTION ", time.ctime(), "\n\n")
    print("Saving db - This will take some hours\n")
    write_h5(dump_folder)
    print("\n\nSaving netCDF4 files")
    h5path = Path(os.path.join(dump_folder, Path('CAETE.h5'))).resolve()
    h52nc(h5path, nc_outputs)
    print(time.ctime())
