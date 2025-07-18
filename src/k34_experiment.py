import os
import sys
import pickle as pkl
import bz2
from pathlib import Path

from numpy import asfortranarray
from pandas import read_csv
import joblib

# import cfunits
import numpy as np
import pandas as pd
import cftime as cf
#
# from aux_plot import get_var
from parameters import tsoil, ssoil, hsoil
from metacommunity import pls_table
from config import get_fortran_runtime
# from sklearn.cluster import KMeans

if sys.platform == "win32":
    from config import update_sys_pathlib, fortran_runtime
    update_sys_pathlib(get_fortran_runtime())


import caete as mod


idxT = pd.date_range("2000-01-01", "2015-12-31", freq='D')
dt = pd.read_csv("../k34/MetData_AmzFACE2000_2015_CAETE.csv")
dt.index = idxT

with bz2.BZ2File("../input/20CRv3-ERA5/obsclim/input_data_185-240.pbz2", mode='rb') as fh:
    dth = pkl.load(fh)

with bz2.BZ2File("../input/20CRv3-ERA5/obsclim/METADATA.pbz2", mode='rb') as fh:
    mdt = pkl.load(fh)


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

# CO2 DATA
AMB = "../k34/CO2_AMB_AmzFACE2000_2100.csv"
ELE = "../k34/CO2_ELE_AmzFACE2000_2100.csv"

main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-9999.csv"))

# this function mimics the behavior of the get_from_main_table function in the region class
# Only for testing purposes
def __get_from_main_table(comm_npls, table = main_table):
    """Returns a number of IDs (in the main table) and the respective
    functional identities (PLS table) to set or reset a community

    Args:
    comm_npls: (int) Number of PLS in the output table (must match npls_max (see caete.toml))"""
    if comm_npls == 1:
        idx = np.random.randint(0, table.shape[1] - 1)
        return idx, table[:, idx]
    idx = np.random.randint(0, comm_npls, comm_npls)
    return idx, table[:, idx]


def run_experiment(fname="../outputs/sdb"):

    # Create the plot object
    k34_plot = mod.grd_mt(-2.61, -60.20, fname, __get_from_main_table)


#     # Fill the plot object with input data
#     k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
#                        pls_table=pls_table, tsoil=tsoil,
#                        ssoil=ssoil, hsoil=hsoil)

#     # Apply a spinup in the soil pools of resources
#     k34_plot = apply_spin(k34_plot)

#     # Run the first sequence of repetitions with co2 fixed at 2000 year values
#     # The binary files for each repetttion are stored as spinX.pkz elsewhere
#     # We run the model trought 450 years to assure a steady state
#     k34_plot.run_caete('20000101', '20151231', spinup=5,
#                        fix_co2='2000', save=True, nutri_cycle=False)

#     k34_plot.run_caete('20000101', '20151231', spinup=5,
#                        fix_co2='2000', save=True)

#     # Run the year of the experiment! the CO2 increases are generated here
#     k34_plot.run_caete('20000101', '20151231', spinup=1, save=True)

#     # #
#     # k34_plot.run_caete('20000101', '20151231', spinup=10,
#     #                    fix_co2="2020", save=True)
    return k34_plot


# def get_spin(grd: mod.plot, spin) -> dict:
#     if spin < 10:
#         name = f'spin0{spin}.pkz'
#     else:
#         name = f'spin{spin}.pkz'
#     with open(grd.outputs[name], 'rb') as fh:
#         spin_dt = joblib.load(fh)
#     return spin_dt


# def pk2csv1(grd: mod.plot, spin) -> pd.DataFrame:

#     spin_dt = get_spin(grd, spin)
#     exp = 3
#     CT1 = pd.read_csv("../k34/CODE_TABLE1.csv")

#     MICV = ['year', 'doy', 'photo', 'npp', 'aresp', 'cleaf',
#             'cawood', None, 'cfroot', None, 'rcm', 'evapm', 'lai']

#     caete_units = {'photo': 'kg m-2 year-1',
#                    'npp': 'kg m-2 year-1',
#                    'aresp': 'kg m-2 year-1',
#                    'cleaf': 'kg m-2',
#                    'cawood': 'kg m-2',
#                    'cfroot': 'kg m-2',
#                    'evapm': 'kg m-2 day-1',
#                    'lai': 'm2 m-2',
#                    'rcm': 'mol m-2 s-1'}
#     # return spin_dt
#     cols = CT1.VariableCode.__array__()
#     units_out = CT1.Unit.__array__()
#     series = []
#     idxT1 = pd.date_range("2000-01-01", "2015-12-31", freq='D', closed=None)
#     idx = idxT1.to_pydatetime()
#     for i, var in enumerate(MICV):
#         if var == 'year':
#             data = [x.timetuple()[0] for x in idx]
#             s1 = pd.Series(np.array(data), index=idxT1)
#             series.append(s1)
#         elif var == 'doy':
#             data = [x.timetuple()[-2] for x in idx]
#             s1 = pd.Series(np.array(data), index=idxT1)
#             series.append(s1)
#         elif var is None:
#             series.append(pd.Series(np.zeros(idxT1.size,) - 9999.0, index=idxT1))
#         elif var == 'rcm':
#             tmp = ((np.zeros(idxT1.size,) + 1.0) / spin_dt[var]) / 0.0224
#             series.append(pd.Series(tmp, index=idxT1))
#         else:
#             data = cfunits.Units.conform(spin_dt[var], cfunits.Units(
#                 caete_units[var]), cfunits.Units(units_out[i]))
#             s1 = pd.Series(np.array(data), index=idxT1)
#             series.append(s1)
#     dt1 = pd.DataFrame(dict(list(zip(cols, series))))
#     dt1.to_csv(f"AmzFACE_D_CAETE_{EXP[exp]}_spin{spin}.csv", index=False)
#     return dt1

#     # return code_table1


# def pk2csv2(grd: mod.plot, spin) -> pd.DataFrame:
#     exp = 3
#     spin_dt = get_spin(grd, spin)

#     CT1 = pd.read_csv("../k34/CODE_TABLE2.csv")

#     # READ PLS_TABLE:
#     # pls_attrs = pd.read_csv("./pls_attrs.csv")
#     # traits = list(pls_attrs.columns.__array__()[1:])
#     # caete_traits = [trait.upper() for trait in traits]

#     MICV = ['year', 'pid', None, 'ocp', None, None,
#             None, None, None, None, None, None, None,]

#     area = spin_dt['area']
#     idx1 = np.where(area[:,0] > 0.0)[0]
#     cols = CT1.VariableCode.__array__()
#     # LOOP over living strategies in the simulation start
#     idxT1 = pd.date_range("2000-01-01", "2015-12-31", freq='D')
#     fname = f"AmzFACE_Y_CAETE_{EXP[exp]}_spin{spin}"
#     os.mkdir(f"./{fname}")

#     for lev in idx1:
#         area_TS = area[lev,:]
#         area_TS = pd.Series(area_TS, index=idxT1)
#         idxT2 = pd.date_range("2000-12-31", "2015-12-31", freq='Y')
#         YEAR = []
#         PID = []
#         OCP = []
#         for i in idxT2:
#             YEAR.append(i.year)
#             PID.append(int(lev))
#             OCP.append(float(area_TS.loc[[i.date()]]))
#         ocp_ts = pd.Series(OCP, index=idxT2)
#         pid_ts = pd.Series(PID, index=idxT2)
#         y_ts = pd.Series(YEAR, index=idxT2)
#         # return ocp_ts, pid_ts, y_ts
#         series = []
#         for i, var in enumerate(MICV):
#             if var == 'year':
#                 series.append(y_ts)
#             elif var == 'pid':
#                 series.append(pid_ts)
#             elif var is None:
#                 series.append(pd.Series(np.zeros(idxT2.size,) - 9999.0, index=idxT2))
#             elif var == 'ocp':
#                 series.append(ocp_ts)
#             else:
#                 pass
#         dt1 = pd.DataFrame(dict(list(zip(cols, series))))
#         dt1.to_csv(f"./{fname}/AmzFACE_Y_CAETE_{EXP[exp]}_spin{spin}_EV_{int(lev)}.csv", index=False)


# if __name__ == "__main__":
#     pass
#     # make_table_HD() ## Run just one time
#     # make_table_LD()  # Run just one time


#     # LOW FD  # RECOMPILE WITH NPLS=4
#     # pls_table = np.load("./pls_attrs.npy")
#     # ld = run_experiment(pls_table)

#     # INTERMEDIATE FD
#     tb = pls.table_gen(NPLS, Path("./"))
#     # tb = read_pls_table(Path(f"./pls_attrs-{NPLS}.csv"))
#     md1 = run_experiment(tb, "exp2")

#     a = get_spin(md1, 10)
#     print(a["ls"][-1])
#     print(a["area"][:,-1][a["area"][:,-1] > 0])
#     print(a["cawood"])
#     print(a["cfroot"])
#     print(a["cleaf"])
