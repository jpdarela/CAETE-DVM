import _pickle as pkl
import bz2
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
import caete as mod
import plsgen as pls


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

# Read time metadata
with bz2.BZ2File("../cax/ISIMIP_HISTORICAL_METADATA.pbz2", mode='rb') as fh:
    mdt = pkl.load(fh)
stime = copy.deepcopy(mdt[0])

# Read CO2 data
co2 = "../input/co2/historical_CO2_annual_1765_2018.txt"
with open(co2) as fh:
    co2_data = fh.readlines()

# 
def apply_spin(grid):
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date="19790101", end_date="19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid


def run_experiment(pls_table):
        # Open a dataset with the Standard time variable
    tm = nc.Dataset("./time_ISIMIP_hist_obs.nc4", 'r')
    tm1 = tm.variables["time"]

    t1 = datetime.datetime(year=2006,month=1,day=1,hour=0,minute=0,second=0)
    t2 = datetime.datetime(year=2006,month=12,day=31,hour=0,minute=0,second=0)

    # Find the index of the input data array for required dates
    # Will use this to manipulate the input data in sensitivity experiments  
    idx0 = int(nc.date2index(t1, tm1, calendar="proleptic_gregorian", select='nearest'))
    idx1 = int(nc.date2index(t2, tm1, calendar="proleptic_gregorian", select='nearest'))

    print(idx0, idx1)
    # # Create the plot object
    sdata = Path("../cax").resolve()
    cax_grd = mod.grd(257, 183, 'CAX-ISIMIP')

    # Fill the plot object with input data
    cax_grd.init_caete_dyn(sdata, stime_i=stime, co2=co2_data,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # # Apply a numerical spinup in the soil pools of resources
    cax_grd = apply_spin(cax_grd)

    # Run the first sequence of repetitions with co2 fixed at 2000 year values
    # The binary files for each repetttion are stored as spinX.pkz elsewhere
    # We run the model trought 450 years to assure a steady state
    cax_grd.run_caete('19790101', '19991231', spinup=5,
                       fix_co2='1999', save=True, nutri_cycle=False)

    cax_grd.run_caete('19790101', '19991231', spinup=15,
                       fix_co2='1999', save=True)
    
    cax_grd.pr[idx0:idx1 + 1] *= 0.01 

    # Run the experiment!
    cax_grd.run_caete('20000101', '20151231', spinup=1, save=True)
    tm.close()
    return cax_grd



def get_spin(grd: mod.grd, spin) -> dict:
    import joblib
    if spin < 10:
        name = f'spin0{spin}.pkz'
    else:
        name = f'spin{spin}.pkz'
    with open(grd.outputs[name], 'rb') as fh:
        spin_dt = joblib.load(fh)
    return spin_dt



if __name__ == "__main__":
    pass
    pls_table = pls.table_gen(1000, Path("./CAX_PLS_TABLE"))
    cax = run_experiment(pls_table)


# def pk2csv1(grd: mod.plot, spin) -> pd.DataFrame:

#     spin_dt = get_spin(grd, spin)
#     exp = 1
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
#     exp = 1
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
#     idxT1 = pd.date_range("2000-01-01", "2015-12-31", freq='D', closed=None)
#     fname = f"AmzFACE_D_CAETE_{EXP[exp]}_spin{spin}"
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
#         dt1.to_csv(f"./{fname}/AmzFACE_D_CAETE_{EXP[exp]}_spin{spin}_EV_{int(lev)}.csv", index=False)
