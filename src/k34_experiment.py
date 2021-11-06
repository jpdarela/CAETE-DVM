import numpy as np
# import _pickle as pkl
# import bz2
import pandas as pd
import cftime as cf
import caete as mod
import plsgen as pls
from aux_plot import get_var

from sklearn.cluster import KMeans

idxT = pd.date_range("2000-01-02", "2016-01-01", freq='D', closed=None)

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


# input climatic + N/P init day
sdata = {'hurs': dt['RH'].__array__(dtype=np.float64),
         'tas': dt['Temp'].__array__(dtype=np.float64) + 273.15,
         'ps': dt['Press'].__array__(dtype=np.float64) * 100.0,
         'pr': dt['Precip'].__array__(dtype=np.float64) * 1.15741e-05,
         'rsds': dt['RH'].__array__(dtype=np.float64),
         'tn': 1248.81,
         'tp': 66.13,
         'ap': 2.63,
         'ip': 9.11,
         'op': 7.05}

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

with open(AMB, 'r') as fh:
    co2 = fh.readlines()
    co2.pop(0)

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
        k34_plot = mod.plot(-2.61, -60.20, 'k34-CUI')

        k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                           pls_table=pls_table, tsoil=tsoil,
                           ssoil=ssoil, hsoil=hsoil)
        k34_plot = apply_spin(k34_plot)
        k34_plot.run_caete('20000102', '20151231', 10, save=True)
        area = get_var(k34_plot, 'area', (9, 10))
        lpls = area[:, -1] > 0.0
        arr1 = k34_plot.pls_table[:, lpls]
        while True:
            pls_table = pls.table_gen(NPLS)
            k34_plot = mod.plot(-2.61, -60.20, 'k34-CUI')
            k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                               pls_table=pls_table, tsoil=tsoil,
                               ssoil=ssoil, hsoil=hsoil)
            k34_plot = apply_spin(k34_plot)
            k34_plot.run_caete('20000102', '20151231', 10, save=True)
            area = get_var(k34_plot, 'area', (9, 10))
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


def run_HD():
    # Open HD traits table
    pls_table = np.load("./pls_attrs_HD.npy")

    # Create the plot object
    k34_plot = mod.plot(-2.61, -60.20, 'k34-HD')

    # Fill the plot object with input data
    k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # Apply a numerical spinup in the soil pools of resources
    k34_plot = apply_spin(k34_plot)

    # Run the first sequence of repetitions with co2 fixed at 2000 year values
    # The binary files for each repetttion is stored as spinX.pkz elsewhere
    # We run the model trought 450 years to assure a steady state
    k34_plot.run_caete('20000102', '20151231', spinup=30,
                       fix_co2=368.9, save=True)

    # Run the year of the experiment! the CO2 increases are generated here
    k34_plot.run_caete('20000102', '20151231', spinup=1, save=True)

    #
    k34_plot.run_caete('20000102', '20151231', spinup=30,
                       fix_co2="2020", save=True)
    return k34_plot


def run_MD():
    pls_table = pls.table_gen(NPLS)

    # Create the plot object
    k34_plot = mod.plot(-2.61, -60.20, 'k34-MD')

    # Fill the plot object with input data
    k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # Apply a numerical spinup in the soil pools of resources
    k34_plot = apply_spin(k34_plot)

    # Run the first sequence of repetitions with co2 fixed at 2000 year values
    # The binary files for each repetttion is stored as spinX.pkz elsewhere
    # We run the model trought 450 years to assure a steady state
    k34_plot.run_caete('20000102', '20151231', spinup=30,
                       fix_co2=368.9, save=True)

    # Run the year of the experiment! the CO2 increases are generated here
    k34_plot.run_caete('20000102', '20151231', spinup=1, save=True)

    #
    k34_plot.run_caete('20000102', '20151231', spinup=30,
                       fix_co2="2020", save=True)
    return k34_plot
    return k34_plot


def run_LD():
    pls_table = np.load("./pls_attrs_LD.npy")

    # Create the plot object
    k34_plot = mod.plot(-2.61, -60.20, 'k34-LD')

    # Fill the plot object with input data
    k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # Apply a numerical spinup in the soil pools of resources
    k34_plot = apply_spin(k34_plot)

    # Run the first sequence of repetitions with co2 fixed at 2000 year values
    # The binary files for each repetttion is stored as spinX.pkz elsewhere
    # We run the model trought 450 years to assure a steady state
    k34_plot.run_caete('20000102', '20151231', spinup=30,
                       fix_co2=368.9, save=True)

    # Run the year of the experiment! the CO2 increases are generated here
    k34_plot.run_caete('20000102', '20151231', spinup=1, save=True)

    #
    k34_plot.run_caete('20000102', '20151231', spinup=29,
                       fix_co2="2020", save=True)
    return k34_plot


def pk2csv(grd: mod.grd, spin) -> pd.DataFrame:
    import joblib
    if spin < 10:
        name = f'spin0{spin}.pkz'
    else:
        name = f'spin{spin}.pkz'
    with open(grd.outputs[name], 'rb') as fh:
        dt = joblib.load(fh)

    MICV = ['photo','npp', 'aresp']





if __name__ == "__main__":

    # make_table_HD() ## Run just one time
    # make_table_LD()  # Run just one time

    # hd = run_HD()
    # md = run_MD()

    # Need to compile the fortrna code do 4 PLSs
    # ld = run_LD()
    sd = pk2csv()
