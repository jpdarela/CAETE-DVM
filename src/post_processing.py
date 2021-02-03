# post_processing.py
# Process raw outputs from CAETÊ-DVM
# USe pytables to create h5 complex datasets
import os
from pathlib import Path
import joblib
import numpy as np
import cftime
import pandas as pd
import tables as tb
import template_tables as tt
from caete_module import global_par as gp


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
    """post processing of Nutrients limitation in the allocation process for leaf||wood||root.
       - pool_lim[np.ndarray shape(NPLS,NDAYS)] is the limitation status record for a pool (leaf||wood||root)
       of a bunch of PLS
       - area is a NPLS sized array with area percentage of the cuupied area of each pls
       output: A tuple with the percentage of the time that a specific limitation occured in a vegetation pool"""

    pool_nolim = []  # No limitation
    pool_lim_n = []  # N limitation
    pool_lim_p = []  # P limitation
    # Colimitation driven by N (When the realized NPP allocation is smaller
    # thant the potential due to N but the other element is also limitant)
    pool_colim_n = []
    # Colimitation driven by P (When the realized NPP allocation is smaller
    # than the potential due to P but the other element is also limitant
    pool_colim_p = []
    # Real Colimitation = K <= 1D-6 (K is difference between P and N realized NPP allocation)
    pool_colim_np = []

    ndays = pool_lim.shape[1]
    npls = pool_lim.shape[0]

    for pls in range(npls):
        if area[pls]:
            no_lim = (pool_lim[pls, :] == 0).sum() / ndays * area[pls]
            lim_n = (np.count_nonzero(
                pool_lim[pls, :] == 1) / ndays) * area[pls]
            lim_p = (np.count_nonzero(
                pool_lim[pls, :] == 2) / ndays) * area[pls]
            colim_n = (np.count_nonzero(
                pool_lim[pls, :] == 4) / ndays) * area[pls]
            colim_p = (np.count_nonzero(
                pool_lim[pls, :] == 5) / ndays) * area[pls]
            colim_np = (np.count_nonzero(
                pool_lim[pls, :] == 6) / ndays) * area[pls]

            pool_nolim.append(no_lim)
            pool_lim_n.append(lim_n)
            pool_lim_p.append(lim_p)
            pool_colim_n.append(colim_n)
            pool_colim_p.append(colim_p)
            pool_colim_np.append(colim_np)

    return (np.sum(pool_nolim),
            np.sum(pool_lim_n),
            np.sum(pool_lim_p),
            np.sum(pool_colim_n),
            np.sum(pool_colim_p),
            np.sum(pool_colim_np))


def process_ustrat(u_strat, area):

    nuts = u_strat.shape[0]
    npls = u_strat.shape[1]
    ndays = u_strat.shape[2]

    # Nitrogen strats
    pool_passive_uptkN = []
    pool_nmaN = []
    pool_nmeN = []
    pool_amN = []
    pool_emN = []
    pool_em0 = []

    # P strats
    pool_passive_uptkP = []
    pool_nmaP = []
    pool_nmeP = []
    pool_amP = []
    pool_emP = []
    pool_ramAP = []
    pool_remAP = []
    pool_amap = []
    pool_emx0 = []

    for pls in range(npls):
        if area[pls]:
            passive_uptkN = (u_strat[0, pls, :] == 0).sum() / ndays * area[pls]
            nmaN = (np.count_nonzero(
                u_strat[0, pls, :] == 1) / ndays) * area[pls]
            nmeN = (np.count_nonzero(
                u_strat[0, pls, :] == 2) / ndays) * area[pls]
            amN = (np.count_nonzero(
                u_strat[0, pls, :] == 3) / ndays) * area[pls]
            emN = (np.count_nonzero(
                u_strat[0, pls, :] == 4) / ndays) * area[pls]
            em0 = (np.count_nonzero(
                u_strat[0, pls, :] == 6) / ndays) * area[pls]

            passive_uptkP = (u_strat[1, pls, :] == 0).sum() / ndays * area[pls]
            nmaP = (np.count_nonzero(
                u_strat[1, pls, :] == 1) / ndays) * area[pls]
            nmeP = (np.count_nonzero(
                u_strat[1, pls, :] == 2) / ndays) * area[pls]
            amP = (np.count_nonzero(
                u_strat[1, pls, :] == 3) / ndays) * area[pls]
            emP = (np.count_nonzero(
                u_strat[1, pls, :] == 4) / ndays) * area[pls]
            ramAP = (np.count_nonzero(
                u_strat[1, pls, :] == 5) / ndays) * area[pls]
            remAP = (np.count_nonzero(
                u_strat[1, pls, :] == 6) / ndays) * area[pls]
            amap = (np.count_nonzero(
                u_strat[1, pls, :] == 7) / ndays) * area[pls]
            emx0 = (np.count_nonzero(
                u_strat[1, pls, :] == 8) / ndays) * area[pls]

            pool_passive_uptkN.append(passive_uptkN)
            pool_nmaN.append(nmaN)
            pool_nmeN.append(nmeN)
            pool_amN.append(amN)
            pool_emN.append(emN)
            pool_em0.append(em0)

            pool_passive_uptkP.append(passive_uptkP)
            pool_nmaP.append(nmaP)
            pool_nmeP.append(nmeP)
            pool_amP.append(amP)
            pool_emP.append(emP)
            pool_ramAP.append(ramAP)
            pool_remAP.append(remAP)
            pool_amap.append(amap)
            pool_emx0.append(emx0)

    return((
        np.sum(pool_passive_uptkN),
        np.sum(pool_nmaN),
        np.sum(pool_nmeN),
        np.sum(pool_amN),
        np.sum(pool_emN),
        np.sum(pool_em0)
    ), (
        np.sum(pool_passive_uptkP),
        np.sum(pool_nmaP),
        np.sum(pool_nmeP),
        np.sum(pool_amP),
        np.sum(pool_emP),
        np.sum(pool_ramAP),
        np.sum(pool_remAP),
        np.sum(pool_amap),
        np.sum(pool_emx0)
    ))


def write_h5(out_dir=Path('../outputs'), RUN=0):

    postp = os.path.join(Path(out_dir), Path("CAETE.h5"))
    h5file = tb.open_file(postp, mode="w", title="Test file")

    group_run = h5file.create_group(
        "/", f'RUN{RUN}', f'CAETÊ outputs tables Run {RUN}')

    table_g1 = h5file.create_table(
        group_run, 'Outputs_G1', tt.run_g1, "outputs for variables of group 1", expectedrows=9 * 365 * 40)
    table_g2 = h5file.create_table(
        group_run, 'Outputs_G2', tt.run_g2, "outputs for variables of group 2", expectedrows=9 * 365 * 40)
    table_g3 = h5file.create_table(
        group_run, 'Outputs_G3', tt.run_g3, "outputs for variables of group 3", expectedrows=9 * 365 * 40)
    PLS_table = h5file.create_table(
        group_run, 'PLS', tt.PLS_temp, f"PLS table for RUN{RUN}", expectedrows=gp.npls)
    spin_table = h5file.create_table(
        group_run, 'spin_snapshot', tt.spin_snapshots, "Area, Nutrient limitation and N/P uptake")

    h5file.flush()
    h5file.close()

    cells = []
    grds = os.listdir(out_dir)
    grds = [Path(os.path.join(out_dir, grd)).resolve()
            for grd in grds if Path(os.path.join(out_dir, grd)).is_dir()]
    for grd in grds:
        files = sorted(os.listdir(grd))
        for f in files:
            cells.append(os.path.join(grd, f))

    for fp in cells:
        dt = open_fh(fp)
        # FILL group 1
        # FILL group 2
        # FILL group 3
        # FILL PLS
        # FILL SPIN SNAPSHOT
        #
        # print(f"\n\n{fp}\n")
        # print(var_names_dims(dt))
        # print(se_dates(dt))
        # return dt
