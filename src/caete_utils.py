import copy
import bz2
import os
import pickle as pkl
import joblib
from numpy import asfortranarray
from pandas import read_csv, DataFrame, concat
from parameters import *
from caete import grd
from aux_plot import get_var
import plsgen as pls

regions = ["central", "east", "north_west", "south", "cax", "k34"]

def read_pls_table(pls_file=pls_path):
    """Read the standard attributes table saved in csv format.
       Return numpy array (shape=(ntraits, npls), F_CONTIGUOUS)"""
    return asfortranarray(read_csv(pls_file).__array__()[:,1:].T)

def get_spin(grid: grd, spin) -> dict:

    if spin < 10:
        name = f'spin0{spin}.pkz'
    else:
        name = f'spin{spin}.pkz'
    with open(grid.outputs[name], 'rb') as fh:
        spin_dt = joblib.load(fh)
    return spin_dt

def apply_spin(grid:grd)->grd:
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid

def prep_env(nx, ny, reg):
    pos = (nx, ny)
    if reg  in ["cax","k34"]:
        s_data = Path("../").resolve()
    else:
        s_data = Path("../input").resolve()
    clim_and_soil_data = Path(reg)
    input_path = Path(os.path.join(s_data, clim_and_soil_data))
    clim_metadata = Path(os.path.join(s_data, clim_and_soil_data,
                                      "ISIMIP_HISTORICAL_METADATA.pbz2"))
    with bz2.BZ2File(clim_metadata, mode='r') as fh:
        clim_metadata = pkl.load(fh)

    stime = copy.deepcopy(clim_metadata[0])
    with open(os.path.join(Path("../input").resolve(), "co2/historical_CO2_annual_1765_2018.txt")) as fh:
        co2_data = fh.readlines()
    return input_path, stime, co2_data, pos

def make_grd(nx, ny, reg, NPLS):
    input_path, stime, co2_data, pos = prep_env(nx, ny, reg)

    grid_cell_in = grd(pos[0], pos[1], f"{reg}_{pos[0]}_{pos[1]}")

    pls_table = pls.table_gen(NPLS)
    grid_cell_in.init_caete_dyn(input_fpath=input_path, stime_i=stime, co2=co2_data,
                        pls_table=pls_table, tsoil=tsoil,
                        ssoil=ssoil, hsoil=hsoil)

    grid_cell_in = apply_spin(grid_cell_in)

    grid_cell_in.run_caete('19790101', '20091231', spinup=5, fix_co2="1979", save=False, nutri_cycle=False)
    grid_cell_in.run_caete('19790101', '20091231', spinup=10, fix_co2="1979", save=True)
    return grid_cell_in

def make_table_HD(nx, ny, reg, NPLS):
    """ Call this function to build a PLS table filled with adapted EV for the Amazon region."""
    lim = 100
    def lplss():
        print("FST")
        grid_cell_in = make_grd(nx=nx, ny=ny, reg=reg, NPLS=NPLS)
        area = get_var(grid_cell_in, 'area', (9, 10))
        lpls = area[:, -1] > 0.0
        arr1 = grid_cell_in.pls_table[:, lpls]
        print(f"INIT SUCC EV UN: {lpls.sum()}")
        while True:
            try:
                print("REPEAT ...")
                print(f"ARR SHP: {arr1.shape}")
                grid_cell_in = make_grd(nx=nx, ny=ny, reg=reg, NPLS=NPLS)
                area = get_var(grid_cell_in, 'area', (9, 10))
                lpls = area[:, -1] > 0.0
                lpls_arr = grid_cell_in.pls_table[:, lpls]
                arr1 = np.concatenate((arr1, lpls_arr), axis=1)
                if arr1.shape[-1] >= lim:
                    break
            except:
                pass
        return arr1
    head = ['g1', 'resopfrac', 'tleaf', 'twood', 'troot', 'aleaf', 'awood', 'aroot', 'c4',
            'leaf_n2c', 'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c', 'froot_p2c',
            'amp', 'pdia']

    FIT_PLS = lplss()
    pls_table_HD = FIT_PLS[:, 0:lim]
    # np.save("./k34_PLS_TABLE/pls_attrs_HD.npy", pls_table_HD)
    dtf = DataFrame(pls_table_HD.T, columns=head)
    dtf.to_csv(f"pls_testing_{reg}_{nx}-{ny}_{lim}.csv", index=False)

def reg_table(nx, ny, reg, NPLS):
    complete = False
    counter = 0
    while True:
        try:
            make_table_HD(nx, ny, reg, NPLS)
            complete = True
        except:
            counter += 1
        if complete or counter > 10:
            break

def merge_tables():
    files  = Path("./").glob("pls_testing*.csv")
    df = concat([read_csv(f) for f in files], ignore_index=True)
    df.to_csv("pls_attrs-selec.csv", index=False)

def main():
    npls = 10000
    reg_table(257, 183, "cax", npls)
    reg_table(240, 185, "k34", npls)
    reg_table(238, 183, "central", npls)
    reg_table(238, 175, "central", npls)
    reg_table(240, 179, "central", npls)
    reg_table(239, 175, "central", npls)
    reg_table(255, 190, "east", npls)
    reg_table(260, 200, "east", npls)
    reg_table(260, 190, "east", npls)
    reg_table(260, 191, "east", npls)
    reg_table(256, 190, "east", npls)
    reg_table(225, 168, "north_west", npls)
    reg_table(225, 174, "north_west", npls)
    reg_table(228, 172, "north_west", npls)
    reg_table(228, 168, "north_west", npls)
    reg_table(230, 200, "south", npls)
    reg_table(230, 201, "south", npls)
    reg_table(225, 200, "south", npls)
    reg_table(225, 210, "south", npls)
    reg_table(230, 205, "south", npls)

if __name__ == "__main__":
    main()
    merge_tables()