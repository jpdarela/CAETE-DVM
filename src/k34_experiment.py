import os
import _pickle as pkl
import bz2
from pathlib import Path

from numpy import asfortranarray
from pandas import read_csv

import cfunits
import numpy as np
import pandas as pd
import cftime as cf
import caete as mod
import plsgen as pls
from aux_plot import get_var

#

from sklearn.cluster import KMeans

idxT = pd.date_range("2000-01-01", "2015-12-31", freq='D', closed=None)

dt = pd.read_csv("../k34/MetData_AmzFACE2000_2015_CAETE.csv")
dt.index = idxT

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

with bz2.BZ2File("../k34/input_data_185-240.pbz2", mode='rb') as fh:
    dth = pkl.load(fh)

# input climatic + N/P init day
sdata = {'hurs': dt['RH'].__array__(dtype=np.float64),
         'tas': dt['Temp'].__array__(dtype=np.float64) + 273.15,
         'ps': dt['Press'].__array__(dtype=np.float64) * 100.0,
         'pr': dt['Precip'].__array__(dtype=np.float64) * 1.15741e-05,
         'rsds': dt['Rad'].__array__(dtype=np.float64),
         'tn': dth['tn'],
         'tp': 18.70133513,
         'ap': 0.22684275,
         'ip': 3.241618875,
         'op': 4.421773125}
        #  'tn': dth['tn'],
        #  'tp': dth['tp'],
        #  'ap': dth['ap'],
        #  'ip': dth['ip'],
        #  'op': dth['op']}

# TIME
tu = 'days since 1860-01-01 00:00:00'
ca = 'proleptic_gregorian'

stime_i = {'standard_name': 'time',
           'units': tu,
           'calendar': ca,
           'time_index': cf.date2num(idxT.to_pydatetime(), units=tu, calendar=ca)}

# The above two dictionaries with inputa data are based on the original input:

# ISIMIP INPUT DATA FOR THE GRIDCELL OF K34 and METADATA (TIME LON LAT)
# with bz2.BZ2File("../k34/input_data_185-240.pbz2", mode='rb') as fh:
#     dth = pkl.load(fh)

# with bz2.BZ2File("../k34/ISIMIP_HISTORICAL_METADATA.pbz2", mode='rb') as fh:
#     mdt = pkl.load(fh)

# CO2 DATA
AMB = "../k34/CO2_AMB_AmzFACE2000_2100.csv"
ELE = "../k34/CO2_ELE_AmzFACE2000_2100.csv"

with open(ELE, 'r') as fh:
    co2 = fh.readlines()
    co2.pop(0)

EXP = ['AMB_LD', 'AMB_MD', 'AMB_HD', 'ELE_LD', 'ELE_MD', 'ELE_HD']

# PLS DATA:
NPLS = 1000


def apply_spin(grid):
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="20000102", end_date="20050102")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def make_table_HD():
    """ Call this function to build a PLS table filled with adapted EV for the K34 site
    This may take some time"""
    def lplss():
        pls_table = pls.table_gen(NPLS)
        print("FST")
        k34_plot = mod.plot(-2.61, -60.20, 'k34-CUI')

        k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                           pls_table=pls_table, tsoil=tsoil,
                           ssoil=ssoil, hsoil=hsoil)

        k34_plot = apply_spin(k34_plot)

        k34_plot.run_caete('20000102', '20151231', 10, save=True, nutri_cycle=False)
        k34_plot.run_caete('20000102', '20151231', spinup=10, save=True)

        area = get_var(k34_plot, 'area', (19, 20))
        lpls = area[:, -1] > 0.0
        arr1 = k34_plot.pls_table[:, lpls]
        print(f"INIT SUCC EV UN: {lpls.sum()}")
        while True:
            print("REPEAT ...")
            print(f"ARR SHP: {arr1.shape}")
            pls_table = pls.table_gen(NPLS)
            k34_plot = mod.plot(-2.61, -60.20, 'k34-CUI')
            k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                               pls_table=pls_table, tsoil=tsoil,
                               ssoil=ssoil, hsoil=hsoil)
            k34_plot = apply_spin(k34_plot)
            print("RUNNING")
            k34_plot.run_caete('20000102', '20151231', 10, save=True, nutri_cycle=False)
            k34_plot.run_caete('20000102', '20151231', spinup=10, save=True)

            area = get_var(k34_plot, 'area', (19, 20))
            lpls = area[:, -1] > 0.0
            lpls_arr = k34_plot.pls_table[:, lpls]
            arr1 = np.concatenate((arr1, lpls_arr), axis=1)
            if arr1.shape[-1] >= NPLS:
                break
        return arr1
    head = ['g1', 'resopfrac', 'tleaf', 'twood', 'troot', 'aleaf', 'awood', 'aroot', 'c4',
            'leaf_n2c', 'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c', 'froot_p2c',
            'amp', 'pdia']

    FIT_PLS = lplss()
    pls_table_HD = FIT_PLS[:, 0:1000]
    np.save("pls_attrs_HD.npy", pls_table_HD)
    dtf = pd.DataFrame(pls_table_HD.T, columns=head)
    dtf.to_csv("pls_attrs_HD.csv", index=False)


def make_table_LD():
    # Creating a low div dataset using kmeans
    head = ['g1', 'resopfrac', 'tleaf', 'twood', 'troot', 'aleaf', 'awood', 'aroot', 'c4',
            'leaf_n2c', 'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c', 'froot_p2c',
            'amp', 'pdia']
    # LOAD HD DATASET
    plsHD_arr = np.load("./pls_attrs_HD.npy")

    dtf = pd.DataFrame(plsHD_arr.T, columns=head)

    woodies = dtf[head][dtf['awood'] > 0.0]
    grasses = dtf[head][dtf['awood'] == 0.0]
    grassesC3 = grasses[head][grasses['c4'] == 0]
    grassesC4 = grasses[head][grasses['c4'] == 1]

    kmeans = KMeans(n_clusters=2).fit(woodies)

    dt1 = woodies.copy()

    dt1.loc[:, 'GROUP'] = kmeans.labels_

    W1 = dt1[head][dt1["GROUP"] == 0]
    W2 = dt1[head][dt1["GROUP"] == 1]

    # Low diversity table:
    # PLS 1 = Grass C3
    # PLS 2 = Grass C4
    # PLS 3 = Woody 1
    # PLS 4 = Woody 2 -> one of the woodies are a N fixer

    pls_attrs_LD = np.zeros(shape=(17, 4), dtype=np.float64, order="F")
    pls_attrs_LD[:, 0] = grassesC3.mean().__array__()
    pls_attrs_LD[:, 1] = grassesC4.mean().__array__()
    pls_attrs_LD[:, 2] = W1.mean().__array__()
    pls_attrs_LD[:, 3] = W2.mean().__array__()

    if pls_attrs_LD[16, 2] > pls_attrs_LD[16, 3]:
        pls_attrs_LD[16, 3] = 0.0
    else:
        pls_attrs_LD[16, 2] = 0.0

    dt2 = pd.DataFrame(pls_attrs_LD.T, columns=head)
    dt2.to_csv("pls_attrs_LD.csv", index=False)
    np.save("pls_attrs_LD.npy", pls_attrs_LD)


def read_pls_table(pls_path):
    """Read the standard attributes table saved in csv format. 
       Return numpy array (shape=(ntraits, npls), F_CONTIGUOUS)"""
    return asfortranarray(read_csv(pls_path).__array__()[:,1:].T)


def run_experiment(pls_table):

    # Create the plot object
    k34_plot = mod.plot(-2.61, -60.20, 'k34-ELE_LD_1000')

    # Fill the plot object with input data
    k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # Apply a numerical spinup in the soil pools of resources
    k34_plot = apply_spin(k34_plot)

    # Run the first sequence of repetitions with co2 fixed at 2000 year values
    # The binary files for each repetttion are stored as spinX.pkz elsewhere
    # We run the model trought 450 years to assure a steady state
    k34_plot.run_caete('20000101', '20151231', spinup=10,
                       fix_co2='2000', save=True, nutri_cycle=False)

    k34_plot.run_caete('20000101', '20151231', spinup=20,
                       fix_co2='2000', save=True)

    # Run the year of the experiment! the CO2 increases are generated here
    k34_plot.run_caete('20000101', '20151231', spinup=1, save=True)

    #
    k34_plot.run_caete('20000101', '20151231', spinup=10,
                       fix_co2="2020", save=True)
    return k34_plot


def get_spin(grd: mod.plot, spin) -> dict:
    import joblib
    if spin < 10:
        name = f'spin0{spin}.pkz'
    else:
        name = f'spin{spin}.pkz'
    with open(grd.outputs[name], 'rb') as fh:
        spin_dt = joblib.load(fh)
    return spin_dt


def pk2csv1(grd: mod.plot, spin) -> pd.DataFrame:

    spin_dt = get_spin(grd, spin)
    exp = 3
    CT1 = pd.read_csv("../k34/CODE_TABLE1.csv")

    MICV = ['year', 'doy', 'photo', 'npp', 'aresp', 'cleaf',
            'cawood', None, 'cfroot', None, 'rcm', 'evapm', 'lai']

    caete_units = {'photo': 'kg m-2 year-1',
                   'npp': 'kg m-2 year-1',
                   'aresp': 'kg m-2 year-1',
                   'cleaf': 'kg m-2',
                   'cawood': 'kg m-2',
                   'cfroot': 'kg m-2',
                   'evapm': 'kg m-2 day-1',
                   'lai': 'm2 m-2',
                   'rcm': 'mol m-2 s-1'}
    # return spin_dt
    cols = CT1.VariableCode.__array__()
    units_out = CT1.Unit.__array__()
    series = []
    idxT1 = pd.date_range("2000-01-01", "2015-12-31", freq='D', closed=None)
    idx = idxT1.to_pydatetime()
    for i, var in enumerate(MICV):
        if var == 'year':
            data = [x.timetuple()[0] for x in idx]
            s1 = pd.Series(np.array(data), index=idxT1)
            series.append(s1)
        elif var == 'doy':
            data = [x.timetuple()[-2] for x in idx]
            s1 = pd.Series(np.array(data), index=idxT1)
            series.append(s1)
        elif var is None:
            series.append(pd.Series(np.zeros(idxT1.size,) - 9999.0, index=idxT1))
        elif var == 'rcm':
            tmp = ((np.zeros(idxT1.size,) + 1.0) / spin_dt[var]) / 0.0224
            series.append(pd.Series(tmp, index=idxT1))
        else:
            data = cfunits.Units.conform(spin_dt[var], cfunits.Units(
                caete_units[var]), cfunits.Units(units_out[i]))
            s1 = pd.Series(np.array(data), index=idxT1)
            series.append(s1)
    dt1 = pd.DataFrame(dict(list(zip(cols, series))))
    dt1.to_csv(f"AmzFACE_D_CAETE_{EXP[exp]}_spin{spin}.csv", index=False)
    return dt1

    # return code_table1


def pk2csv2(grd: mod.plot, spin) -> pd.DataFrame:
    exp = 3
    spin_dt = get_spin(grd, spin)
   
    CT1 = pd.read_csv("../k34/CODE_TABLE2.csv")

    # READ PLS_TABLE:
    # pls_attrs = pd.read_csv("./pls_attrs.csv")
    # traits = list(pls_attrs.columns.__array__()[1:])
    # caete_traits = [trait.upper() for trait in traits]

    MICV = ['year', 'pid', None, 'ocp', None, None,
            None, None, None, None, None, None, None,]

    area = spin_dt['area']
    idx1 = np.where(area[:,0] > 0.0)[0]
    cols = CT1.VariableCode.__array__()
    # LOOP over living strategies in the simulation start
    idxT1 = pd.date_range("2000-01-01", "2015-12-31", freq='D', closed=None)
    fname = f"AmzFACE_Y_CAETE_{EXP[exp]}_spin{spin}"
    os.mkdir(f"./{fname}")
    
    for lev in idx1:
        area_TS = area[lev,:]
        area_TS = pd.Series(area_TS, index=idxT1)
        idxT2 = pd.date_range("2000-12-31", "2015-12-31", freq='Y')
        YEAR = []
        PID = []
        OCP = []
        for i in idxT2:
            YEAR.append(i.year)
            PID.append(int(lev))
            OCP.append(float(area_TS.loc[[i.date()]]))
        ocp_ts = pd.Series(OCP, index=idxT2)
        pid_ts = pd.Series(PID, index=idxT2)
        y_ts = pd.Series(YEAR, index=idxT2)
        # return ocp_ts, pid_ts, y_ts
        series = []
        for i, var in enumerate(MICV):
            if var == 'year':
                series.append(y_ts)
            elif var == 'pid':
                series.append(pid_ts)
            elif var is None:
                series.append(pd.Series(np.zeros(idxT2.size,) - 9999.0, index=idxT2))
            elif var == 'ocp':
                series.append(ocp_ts)
            else:
                pass
        dt1 = pd.DataFrame(dict(list(zip(cols, series))))
        dt1.to_csv(f"./{fname}/AmzFACE_Y_CAETE_{EXP[exp]}_spin{spin}_EV_{int(lev)}.csv", index=False)


if __name__ == "__main__":
    pass
    # make_table_HD() ## Run just one time
    # make_table_LD()  # Run just one time


    # LOW FD  # RECOMPILE WITH NPLS=4
    # pls_table = np.load("./pls_attrs.npy")
    # ld = run_experiment(pls_table)

    # INTERMEDIATE FD
    # pls_table = pls.table_gen(NPLS, Path("./k34_PLS_TABLE/"))
    pls_table = read_pls_table("./k34_PLS_TABLE/pls_attrs-1000.csv")
    md = run_experiment(pls_table)

    # Open HIGH FD traits table
    # pls_table = np.load("./pls_attrs_HD.npy")
    # hd = run_experiment(pls_table)
