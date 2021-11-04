import numpy as np
import _pickle as pkl
import bz2
import pandas as pd
import cftime as cf
import caete as mod

idxT = pd.date_range("2000-01-02", "2016-01-01", freq='D', closed=None)

dt = pd.read_csv("./MetData_AmzFACE2000_2015_CAETE.csv")
dt.index = idxT

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
