import numpy as np
import _pickle as pkl
import bz2
import pandas as pd
import cftime as cf

idxT = pd.date_range("2000-01-02", "2016-01-01", freq='D', closed=None)

dt = pd.read_csv("./MetData_AmzFACE2000_2015_CAETE.csv")
dt.index = idxT


with bz2.BZ2File("./input_data_185-240.pbz2", mode='rb') as fh:
    dth = pkl.load(fh)

with bz2.BZ2File("./ISIMIP_HISTORICAL_METADATA.pbz2", mode='rb') as fh:
    mdt = pkl.load(fh)


units = 'days since 1860-01-01 00:00:00'
calendar = 'proleptic_gregorian'
time_num = cf.date2num(idxT.to_pydatetime(), units, calendar)
