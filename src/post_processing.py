# post_processing.py
# Process raw outputs from CAETÊ-DVM
# USe pytables to create h5 complex datasets
import os
import joblib
import numpy as np
import cftime
import pandas as pd

# Open a spinXX.pkz file


def open_fh(fl):
    try:
        with open(fl, 'rb') as fh:
            return joblib.load(fh)
    except:
        raise(FileNotFoundError)


def var_names_dims(dt):
    dims = []
    try:
        vars = list(dt.keys())
        for key, value in dt.items():
            if type(value) == np.ndarray:
                dims.append((value.shape, value.dtype))
            else:
                dims.append(0)
        return dict(zip(vars, dims))
    except:
        assert False, 'Input is not a dict'


def cf_date2str(cftime_in):
    return ''.join(cftime_in.strftime("%Y%m%d")[:10].split('-')).strip()


def str2cf_date(string_date, cf_time=True):
    if not cf_time:
        import datetime
        func = datetime.date
    else:
        import cftime
        func = cftime.real_datetime
    year = int(string_date[:4])
    month = int(string_date[4:6])
    day = int(string_date[6:])
    return func(year, month, day)


def se_dates(dt):
    assert type(dt) == dict
    calendar = dt['calendar']
    time_unit = dt['time_unit']
    start = dt['sind']
    end = dt['eind']

    sdate = cftime.num2date(start, time_unit, calendar)
    edate = cftime.num2date(end, time_unit, calendar)
    # pd.to_datetime(se_dates)

    tp = sdate.strftime(), edate.strftime()

    return pd.date_range(*tp)


def process_lim(pool_lim, area):

    pool_nolim = []
    pool_lim_n = []
    pool_lim_p = []
    pool_colim_n = []
    pool_colim_p = []
    pool_colim_np = []

    ndays = pool_lim.shape[1]
    npls = pool_lim.shape[0]

    for pls in range(npls):
        if area[pls]:
            no_lim = (pool_lim[pls, :] == 0).sum() / ndays
            lim_n = np.count_nonzero(pool_lim[pls, :] == 1) / ndays
            lim_p = np.count_nonzero(pool_lim[pls, :] == 2) / ndays
            colim_n = np.count_nonzero(pool_lim[pls, :] == 4) / ndays
            colim_p = np.count_nonzero(pool_lim[pls, :] == 5) / ndays
            colim_np = np.count_nonzero(pool_lim[pls, :] == 6) / ndays

            pool_nolim.append(no_lim)
            pool_lim_n.append(lim_n)
            pool_lim_p.append(lim_p)
            pool_colim_n.append(colim_n)
            pool_colim_p.append(colim_p)
            pool_colim_np.append(colim_np)

    return (np.mean(pool_nolim),
            np.mean(pool_lim_n),
            np.mean(pool_lim_p),
            np.mean(pool_colim_n),
            np.mean(pool_colim_p),
            np.mean(pool_colim_np))


def main():

    d1 = {'emaxm': (3652,), 'tsoil': (3652,), 'photo': (3652,), 'aresp': (3652,),
          'npp': (3652,), 'lai': (3652,), 'csoil': (4, 3652), 'inorg_n': (3652,),
          'inorg_p': (3652,), 'sorbed_n': (3652,), 'sorbed_p': (3652,), 'snc': (8, 3652),
          'hresp': (3652,), 'rcm': (3652,), 'f5': (3652,), 'runom': (3652,), 'evapm': (3652,),
          'wsoil': (3652,), 'swsoil': (3652,), 'rm': (3652,), 'rg': (3652,), 'cleaf': (3652,),
          'cawood': (3652,), 'cfroot': (3652,), 'area': (1000, 3652), 'wue': (3652,), 'cue': (3652,),
          'cdef': (3652,), 'nmin': (3652,), 'pmin': (3652,), 'vcmax': (3652,), 'specific_la': (3652,),
          'nupt': (2, 3652), 'pupt': (3, 3652), 'litter_l': (3652,), 'cwd': (3652,),
          'litter_fr': (3652,), 'lnc': (6, 3652), 'ls': (3652,), 'lim_status': (3, 1000, 3652),
          'c_cost': (1000, 3652), 'u_strat': (2, 1000, 3652), 'storage_pool': (3, 3652)}

    ok_dim = ['emaxm', 'tsoil', 'photo', 'aresp', 'npp', 'lai',
              'csoil', 'inorg_n', 'inorg_p', 'sorbed_n', 'sorbed_p', 'snc',
              'hresp', 'rcm', 'f5', 'runom', 'evapm', 'wsoil', 'swsoil', 'rm',
              'rg', 'cleaf', 'cawood', 'cfroot', 'area', 'wue', 'cue', 'cdef',
              'nmin', 'pmin', 'vcmax', 'specific_la', 'nupt', 'pupt', 'litter_l',
              'cwd', 'litter_fr', 'lnc']

    ex_dim = [('csoil', 4), ('snc', 8), ('lnc', 6), ('nupt', 2), ('pupt', 3)]

    # unfold 2/3 dim vars

    cells = []
    grds = os.listdir("../outputs")
    for grd in grds:
        fpath = os.path.join("../outputs", grd)
        files = sorted(os.listdir(fpath))
        for f in files:
            cells.append(os.path.join(fpath, f))
    for fp in cells:
        dt = open_fh(fp)
        print(f"\n\n{fp}\n")
        print(var_names_dims(dt))
        print(se_dates(dt))
        return dt
