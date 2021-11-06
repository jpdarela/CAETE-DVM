# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.)

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# AUTHOR: JP Darela


# post_processing.py
# Process raw outputs from CAETÊ-DVM
# USe pytables to create h5 complex datasets
import os
from pathlib import Path
import joblib
import numpy as np
import cftime
import datetime
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


def cf_date2str(cftime_in):
    return ''.join(cftime_in.strftime("%Y%m%d")[:10].split('-')).strip()


def str2cf_date(string_date, cf_time=True):
    if not cf_time:
        func = datetime.date
    else:
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


def write_h5(out_dir=Path('../outputs'), RUN=0, reclen=0):

    h5_opt = tb.Filters(complevel=1, complib="blosc:blosclz")

    # Filters(complevel=0, complib='zlib', shuffle=True,
    # bitshuffle=False, fletcher32=False,
    # least_significant_digit=None, _new=True)

    postp = os.path.join(Path(out_dir), Path("CAETE.h5"))
    with tb.open_file(postp, mode="w", title="CAETÊ outputs") as h5file:

        group_run = h5file.create_group(
            "/", f'RUN{RUN}', f'CAETÊ outputs tables Run {RUN}')
        exp_rows = 2749 * 365 * 100
        table_g1 = h5file.create_table(
            group_run, 'Outputs_G1', tt.run_g1, "out vars of g1",
            filters=h5_opt, expectedrows=exp_rows)
        table_g2 = h5file.create_table(
            group_run, 'Outputs_G2', tt.run_g2, "out vars of g2",
            filters=h5_opt, expectedrows=exp_rows)
        table_g3 = h5file.create_table(
            group_run, 'Outputs_G3', tt.run_g3, "out vars of g3",
            filters=h5_opt, expectedrows=exp_rows)
        PLS_table = h5file.create_table(
            group_run, 'PLS', tt.PLS_temp, f"PLS table for RUN{RUN}", expectedrows=gp.npls)
        spin_table = h5file.create_table(
            group_run, 'spin_snapshot', tt.spin_snapshots,
            "Area, Nutrient limitation and N/P uptake", expectedrows=2749 * 18, filters=h5_opt)

        # write PLS table
        PLS_row = PLS_table.row

        pls_df = pd.read_csv(os.path.join(Path(out_dir), Path(
            "pls_attrs.csv")), delimiter=",")

        for n in range(gp.npls):
            for key in tt.PLS_head:
                PLS_row[key] = pls_df[key][n]
            PLS_row.append()

        PLS_table.flush()

        # Write outputs//
        # Create filepaths to the raw output data
        cells = []
        grds = os.listdir(out_dir)
        grds = [Path(os.path.join(out_dir, grd)).resolve()
                for grd in grds if Path(os.path.join(out_dir, grd)).is_dir()]
        for grd in grds:
            XY = str(grd).split(os.sep)[-1].split("_")[0][8:].split("-")
            Y = int(XY[0])
            X = int(XY[1])
            files = sorted(os.listdir(grd))
            for f in files:
                filepath = os.path.join(grd, f)
                # if prefix in filepath:
                cells.append((filepath, X, Y))

        # Write table G1
        rec = 0
        for fp, X, Y in cells:
            dt = open_fh(fp)
            date_range = se_dates(dt)
            ndays = dt['emaxm'].size
            g1_row = table_g1.row

            for day in range(ndays):

                g1_row['row_id'] = rec
                g1_row['date'] = cf_date2str(date_range[day].date())
                g1_row['grid_y'] = Y
                g1_row['grid_x'] = X
                # 1 D outputs
                for key in tt.G1_1d:
                    g1_row[key] = dt[key][day]

                # nupt
                g1_row['nupt1'] = dt['nupt'][0, day]
                g1_row['nupt2'] = dt['nupt'][1, day]
                g1_row['pupt1'] = dt['pupt'][0, day]
                g1_row['pupt2'] = dt['pupt'][1, day]
                g1_row['pupt3'] = dt['pupt'][2, day]
                rec += 1
                g1_row.append()
        table_g1.flush()

        # Write table G2
        rec = 0
        for fp, X, Y in cells:
            dt = open_fh(fp)
            date_range = se_dates(dt)
            ndays = dt['emaxm'].size
            g2_row = table_g2.row

            for day in range(ndays):
                g2_row['row_id'] = rec
                g2_row['date'] = cf_date2str(date_range[day].date())
                g2_row['grid_y'] = Y
                g2_row['grid_x'] = X

                # 1 D outputs
                for key in tt.G2_1d:
                    g2_row[key] = dt[key][day]

                g2_row['csoil1'] = dt['csoil'][0, day]
                g2_row['csoil2'] = dt['csoil'][1, day]
                g2_row['csoil3'] = dt['csoil'][2, day]
                g2_row['csoil4'] = dt['csoil'][3, day]

                g2_row['sncN1'] = dt['snc'][0, day]
                g2_row['sncN2'] = dt['snc'][1, day]
                g2_row['sncN3'] = dt['snc'][2, day]
                g2_row['sncN4'] = dt['snc'][3, day]
                g2_row['sncP1'] = dt['snc'][4, day]
                g2_row['sncP2'] = dt['snc'][5, day]
                g2_row['sncP3'] = dt['snc'][6, day]
                g2_row['sncP4'] = dt['snc'][7, day]
                rec += 1
                g2_row.append()
        table_g2.flush()

        # Write table G3
        rec = 0
        for fp, X, Y in cells:
            dt = open_fh(fp)
            date_range = se_dates(dt)
            ndays = dt['emaxm'].size
            g3_row = table_g3.row

            for day in range(ndays):
                g3_row['row_id'] = rec
                g3_row['date'] = cf_date2str(date_range[day].date())
                g3_row['grid_y'] = Y
                g3_row['grid_x'] = X

                for key in tt.G3_1d:
                    g3_row[key] = dt[key][day]

                g3_row['lnc1'] = dt['lnc'][0, day]
                g3_row['lnc2'] = dt['lnc'][1, day]
                g3_row['lnc3'] = dt['lnc'][2, day]
                g3_row['lnc4'] = dt['lnc'][3, day]
                g3_row['lnc5'] = dt['lnc'][4, day]
                g3_row['lnc6'] = dt['lnc'][5, day]

                g3_row['sto1'] = dt['storage_pool'][0, day]
                g3_row['sto2'] = dt['storage_pool'][1, day]
                g3_row['sto3'] = dt['storage_pool'][2, day]
                rec += 1
                g3_row.append()
        table_g3.flush()

        # Write Spin_snapshot
        rec = 0
        spsh_row = spin_table.row
        for fp, X, Y in cells:
            dt = open_fh(fp)
            date_range = se_dates(dt)

            spsh_row['row_id'] = rec
            spsh_row['start_date'] = cf_date2str(date_range[0].date())
            spsh_row['end_date'] = cf_date2str(date_range[-1].date())
            spsh_row['grid_y'] = Y
            spsh_row['grid_x'] = X

            # Process limitation data
            leaf = process_lim(dt['lim_status'][0, :, :],
                               dt['area'][:, -1])
            wood = process_lim(dt['lim_status'][1, :, :],
                               dt['area'][:, -1])
            root = process_lim(dt['lim_status'][2, :, :],
                               dt['area'][:, -1])

            spsh_row['leaf_nolim'] = leaf[0]
            spsh_row['leaf_lim_n'] = leaf[1]
            spsh_row['leaf_lim_p'] = leaf[2]
            spsh_row['leaf_colim_n'] = leaf[3]
            spsh_row['leaf_colim_p'] = leaf[4]
            spsh_row['leaf_colim_np'] = leaf[5]
            spsh_row['wood_nolim'] = wood[0]
            spsh_row['wood_lim_n'] = wood[1]
            spsh_row['wood_lim_p'] = wood[2]
            spsh_row['wood_colim_n'] = wood[3]
            spsh_row['wood_colim_p'] = wood[4]
            spsh_row['wood_colim_np'] = wood[5]
            spsh_row['root_nolim'] = root[0]
            spsh_row['root_lim_n'] = root[1]
            spsh_row['root_lim_p'] = root[2]
            spsh_row['root_colim_n'] = root[3]
            spsh_row['root_colim_p'] = root[4]
            spsh_row['root_colim_np'] = root[5]

            n_strat, p_strat = process_ustrat(dt['u_strat'], dt['area'][:, -1])

            spsh_row['upt_stratN0'] = n_strat[0]
            spsh_row['upt_stratN1'] = n_strat[1]
            spsh_row['upt_stratN2'] = n_strat[2]
            spsh_row['upt_stratN3'] = n_strat[3]
            spsh_row['upt_stratN4'] = n_strat[4]
            spsh_row['upt_stratN6'] = n_strat[5]

            spsh_row['upt_stratP0'] = p_strat[0]
            spsh_row['upt_stratP1'] = p_strat[1]
            spsh_row['upt_stratP2'] = p_strat[2]
            spsh_row['upt_stratP3'] = p_strat[3]
            spsh_row['upt_stratP4'] = p_strat[4]
            spsh_row['upt_stratP5'] = p_strat[5]
            spsh_row['upt_stratP6'] = p_strat[6]
            spsh_row['upt_stratP7'] = p_strat[7]
            spsh_row['upt_stratP8'] = p_strat[8]

            spsh_row['area_0'] = dt['area'][:, 0]
            spsh_row['area_f'] = dt['area'][:, -1]
            rec += 1
            spsh_row.append()
        spin_table.flush
