import numpy as np
import _pickle as pkl
import bz2
import pandas as pd
import cftime as cf
import caete as mod
import plsgen as pls

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
with open("../k34/CO2_AMB_AmzFACE2000_2100.csv", 'r') as fh:
    co2 = fh.readlines()
    co2.pop(0)

# PLS DATA:
pls_table = pls.table_gen(1000)

k34_plot = mod.plot(-2.61, -60.20, 'k34-CUI')

k34_plot.init_plot(sdata=sdata, stime_i=stime_i, co2=co2,
                   pls_table=pls_table, tsoil=tsoil,
                   ssoil=ssoil, hsoil=hsoil)
