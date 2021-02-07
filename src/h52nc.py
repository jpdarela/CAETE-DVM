import os
from pathlib import Path

import numpy as np
import tables as tb
from pandas import date_range
import cftime
from netCDF4 import Dataset as dt

from post_processing import cf_date2str, str2cf_date
from caete_module import global_par as gp
from caete import print_progress


def build_strd(strd):
    return f"""(date == b'{strd}')"""


def assemble_layer(ny, nx, var):
    out = np.zeros(shape=(360, 720), dtype=np.float32) - 9999.0

    for i, val in enumerate(var):
        out[ny[i], nx[i]] = val

    return out[160:221, 201:272]


def time_queries(interval):
    sdate = str2cf_date(interval[0])
    edate = str2cf_date(interval[1])

    dates = date_range(sdate, edate)
    query_days = []
    for day in dates:
        query_days.append(build_strd(cf_date2str(day)))
    return query_days


def get_var_metadata(var):

    vunits = {'header': ['long_name', 'unit', 'standart_name', 'ldim'],

              'rsds': ['short_wav_rad_down', 'W m-2', 'rsds'],
              'wind': ['wind_velocity', 'm s-1', 'wind'],
              'ps': ['sur_pressure', 'Pa', 'ps'],
              'tas': ['sur_temperature_2m', 'celcius', 'tas'],
              'tsoil': ['soil_temperature', 'celcius', 'soil_temp'],
              'pr': ['precipitation', 'Kg m-2 month-1', 'pr'],
              'litter_l': ['Litter C flux - leaf', 'g m-2', 'll'],
              'cwd': ['Litter C flux - wood', 'g m-2', 'cwd'],
              'litter_fr': ['Litter C flux fine root', 'g m-2', 'lr'],
              'litter_n': ['Litter Nitrogen Flux', 'g m-2', 'ln'],
              'litter_p': ['Litter phosphorus flux', 'g m-2', 'lp'],
              'sto_c': ['PLant Reserve Carbon', 'g m-2', 'sto_c'],
              'sto_n': ['Pant Reserve Nitrogen', 'g m-2', 'sto_n'],
              'sto_p': ['Plant Reserve Phosphorus', 'g m-2', 'sto_p'],
              'c_cost': ['Carbon costs of Nutrients Uptake', 'g m-2 day-1', 'cc'],
              'wsoil': ['soil_water_content-wsoil', 'kg m-2', 'mrso'],
              'evapm': ['evapotranpiration', 'kg m-2 day-1', 'et'],
              'emaxm': ['potent. evapotranpiration', 'kg m-2 day-1', 'etpot'],
              'runom': ['total_runoff', 'kg m-2 day-1', 'mrro'],
              'aresp': ['autothrophic respiration', 'kg m-2 year-1', 'ar'],
              'photo': ['photosynthesis', 'kg m-2 year-1', 'ph'],
              'npp': ['net primary productivity', 'kg m-2 year-1', 'npp'],
              'lai': ['Leaf Area Index', 'm2 m-2', 'LAI'],
              'rcm': ['stomatal resistence', 's m-1', 'rcm'],
              'hresp': ['Soil heterothrophic respiration', 'g m-2 day-1', 'hr'],
              'nupt': ['Nitrogen uptake', 'g m-2', 'nupt'],
              'pupt': ['Phosphorus uptake', 'g m-2', 'pupt'],
              'csoil': ['Soil Organic Carbon', 'g m-2', 'csoil'],
              'org_n': ['Soil Organic Nitrogen', 'g m-2', 'org_n'],
              'org_p': ['Soil Organic Phosphorus', 'g m-2', 'org_p'],
              'inorg_n': ['Soil Inorganic Nitrogen', 'g m-2', 'inorg_n'],
              'inorg_p': ['Soil Inorganic Phosphorus', 'g m-2', 'inorg_p'],
              'sorbed_p': ['Soil Sorbed Phosphorus', 'g m-2', 'sorbed_p'],
              'nmin': ['Soil Inorganic Nitrogen (solution)', 'g m-2', 'nmin'],
              'pmin': ['Soil Inorganic Phosphorus (solution)', 'g m-2', 'pmin'],
              'rm': ['maintenance respiration', 'kg m-2 year-1', 'rm'],
              'rg': ['growth respiration', 'kg m-2 year-1', 'rg'],
              'wue': ['water use efficiency', 'unitless', 'wue'],
              'vcmax': ['maximum RuBisCo activity', 'mol m-2 s-1', 'vcmax'],
              'sla': ['specfic leaf area', 'm-2 Kg-1', 'SLA'],
              'cue': ['Carbon use efficiency', 'unitless', 'cue'],
              'cawood': ['C in woody tissues', 'kg m-2', 'cawood'],
              'cfroot': ['C in fine roots', 'kg m-2', 'cfroot'],
              'cleaf': ['C in leaves', 'kg m-2', 'cleaf'],
              'cmass': ['total Carbon -Biomass', 'kg m-2', 'cmass']}

    out = {}
    for v in var:
        out[v] = vunits[v]
    return out


def write_daily_output(arr, var, flt_attrs, interval, experiment="TEST RUN HISTORICAL ISIMIP"):

    NO_DATA = [-9999.0, -9999.0]

    time_units = "days since 1979-01-01"
    calendar = "proleptic_gregorian"
    nc_out = Path("../nc_outputs")

    start = cftime.date2num(str2cf_date(interval[0]), time_units, calendar)
    stop = cftime.date2num(str2cf_date(interval[1]), time_units, calendar)

    time_dim = np.arange(start, stop + 1, dtype=np.int32)

    longitude_0 = np.arange(-179.75, 180, 0.5)[201:272]
    latitude_0 = np.arange(89.75, -90, -0.5)[160:221]
    print("\nSaving netCDF4 files")
    print_progress(0, len(var), prefix='Progress:', suffix='Complete')

    for i, v in enumerate(var):
        nc_filename = os.path.join(nc_out, Path(
            f'{v}_{interval[0]}-{interval[1]}.nc4'))
        with dt(nc_filename, mode='w', format='NETCDF4') as rootgrp:
            # dimensions  & variables

            rootgrp.createDimension("latitude", latitude_0.size)
            rootgrp.createDimension("longitude", longitude_0.size)
            rootgrp.createDimension("time", None)

            time = rootgrp.createVariable(varname="time", datatype=np.int32,
                                          dimensions=("time",))
            latitude = rootgrp.createVariable(
                varname="latitude", datatype=np.float32, dimensions=("latitude",))
            longitude = rootgrp.createVariable(
                varname="longitude", datatype=np.float32, dimensions=("longitude",))
            var_ = rootgrp.createVariable(varname=flt_attrs[v][2], datatype=np.float32,
                                          dimensions=(
                                              "time", "latitude", "longitude",),
                                          zlib=True, fill_value=NO_DATA[0], fletcher32=True)

            # attributes
            # rootgrp
            rootgrp.description = flt_attrs[v][0] + " from CAETÊ-CNP OUTPUT"
            rootgrp.source = "CAETE model outputs - darelafilho@gmail.com"
            rootgrp.experiment = experiment

            # time
            time.units = time_units
            time.calendar = calendar
            time.axis = 'T'

            # lat
            latitude.units = u"degrees_north"
            latitude.long_name = u"latitude"
            latitude.standart_name = u"latitude"
            latitude.axis = u'Y'
            # lon
            longitude.units = "degrees_east"
            longitude.long_name = "longitude"
            longitude.standart_name = "longitude"
            longitude.axis = u'X'
            # var
            var_.long_name = flt_attrs[v][0]
            var_.units = flt_attrs[v][1]
            var_.standard_name = flt_attrs[v][2]
            var_.missing_value = NO_DATA[0]

            # WRITING DATA
            longitude[:] = longitude_0
            latitude[:] = latitude_0
            time[:] = time_dim
            var_[:, :, :] = np.fliplr(np.ma.masked_array(
                arr[i], mask=arr[i] == NO_DATA[0]))
            print_progress(i + 1, len(var), prefix='Progress:',
                           suffix='Complete')


def write_area_output(arr, year, experiment="TEST RUN HISTORICAL ISIMIP"):
    NO_DATA = [-9999.0, -9999.0]

    time_units = "days since 1979-01-01"
    calendar = "proleptic_gregorian"
    nc_out = Path("../nc_outputs")

    time_dim = cftime.date2num(str2cf_date(year), time_units, calendar)

    longitude_0 = np.arange(-179.75, 180, 0.5)[201:272]
    latitude_0 = np.arange(-89.75, 90, 0.5)[160:221]
    print("\nSaving netCDF4 area file:", cftime.num2date(
        time_dim, time_units, calendar))
    nc_filename = os.path.join(nc_out, Path(f'ocp_area_{year}.nc4'))
    with dt(nc_filename, mode='w', format='NETCDF4') as rootgrp:
        # dimensions  & variables

        rootgrp.createDimension("latitude", latitude_0.size)
        rootgrp.createDimension("longitude", longitude_0.size)
        rootgrp.createDimension("pls", arr.shape[0])
        rootgrp.createDimension("time", None)

        time = rootgrp.createVariable(varname="time", datatype=np.int32,
                                      dimensions=("time",))

        pls = rootgrp.createVariable(varname="PLS", datatype=np.int16,
                                     dimensions=("pls",))
        latitude = rootgrp.createVariable(
            varname="latitude", datatype=np.float32, dimensions=("latitude",))
        longitude = rootgrp.createVariable(
            varname="longitude", datatype=np.float32, dimensions=("longitude",))
        var_ = rootgrp.createVariable(varname='ocp_area', datatype=np.float32,
                                      dimensions=(
                                          "time", "pls", "latitude", "longitude",),
                                      fill_value=NO_DATA[0])

        # attributes
        # rootgrp
        rootgrp.description = "Ocupation coefficients of Plant Life Strategies" + \
            " from CAETÊ-CNP OUTPUT"
        rootgrp.source = "CAETE model outputs"
        rootgrp.experiment = experiment

        # time
        time.units = time_units
        time.calendar = calendar
        time.axis = u'T'

        # time
        pls.units = u'unitless'
        pls.axis = u'P'

        # lat
        latitude.units = u"degrees_north"
        latitude.long_name = u"latitude"
        latitude.standart_name = u"latitude"
        latitude.axis = u'Y'
        # lon
        longitude.units = "degrees_east"
        longitude.long_name = "longitude"
        longitude.standart_name = "longitude"
        longitude.axis = u'X'
        # var
        var_.long_name = "Occupation coefficients of Plant Life Strategies (Abundance data)"
        var_.units = "unitless"
        var_.standard_name = 'ocp_area'
        var_.missing_value = NO_DATA[0]

        # WRITING DATA
        pls[:] = np.arange(gp.npls, dtype=np.int16)
        longitude[:] = longitude_0
        latitude[:] = latitude_0
        time[:] = time_dim
        var_[0, :, :, :] = np.fliplr(np.ma.masked_array(
            arr, mask=arr == NO_DATA[0]))


def create_ncG1(table, interval):
    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    vars = ['photo', 'aresp', 'npp', 'lai', 'wue', 'cue',
            'vcmax', 'sla', 'nupt', 'pupt']

    dates = time_queries(interval)
    dm1 = len(dates)

    photo = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    aresp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    npp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lai = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    wue = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    cue = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    vcmax = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    specific_la = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    nupt1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    nupt2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    pupt1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    pupt2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    pupt3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)

    print("\nQuerying data from file FOR", end=': ')
    for v in vars:
        print(v, end=", ")
    print("\nInterval: ", interval)
    print_progress(0, len(dates), prefix='Progress:', suffix='Complete')
    for i, day in enumerate(dates):
        out = table.read_where(day)
        photo[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['photo'])
        aresp[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['aresp'])
        npp[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['npp'])
        lai[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lai'])
        wue[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wue'])
        cue[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['cue'])
        vcmax[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['vcmax'])
        specific_la[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['specific_la'])
        nupt1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['nupt1'])
        nupt2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['nupt2'])
        pupt1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['pupt1'])
        pupt2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['pupt2'])
        pupt3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['pupt3'])
        print_progress(i + 1,
                       len(dates),
                       prefix='Progress:',
                       suffix='Complete')
    # write netcdf
    nupt1 = nupt2 + nupt1
    pupt1 = pupt3 + pupt2 + pupt1

    vars = ['photo', 'aresp', 'npp', 'lai', 'wue', 'cue',
            'vcmax', 'sla', 'nupt', 'pupt']

    arr = (photo, aresp, npp, lai, wue, cue, vcmax,
           specific_la, nupt1, pupt1)
    var_attrs = get_var_metadata(vars, dm1)
    write_daily_output(arr, vars, var_attrs, interval)


def create_ncG2(table, interval):
    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    vars = ['csoil', 'total_n', 'total_p', 'org_n', 'org_p',
            'inorg_n', 'inorg_p', 'sorbed_p', 'hresp', 'nmin', 'pmin']

    dates = time_queries(interval)
    dm1 = len(dates)

    csoil1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    csoil2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    csoil3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    csoil4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncN1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncN2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncN3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncN4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncP1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncP2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncP3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sncP4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    inorg_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    inorg_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sorbed_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sorbed_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    hresp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    nmin = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    pmin = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)

    print("\nQuerying data from file FOR", end=': ')
    for v in vars:
        print(v, end=", ")
    print("\nInterval: ", interval)
    print_progress(0, len(dates), prefix='Progress:', suffix='Complete')
    for i, day in enumerate(dates):
        out = table.read_where(day)
        csoil1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["csoil1"])
        csoil2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["csoil2"])
        csoil3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["csoil3"])
        csoil4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["csoil4"])
        sncN1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncN1"])
        sncN2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncN2"])
        sncN3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncN3"])
        sncN4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncN4"])
        sncP1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncP1"])
        sncP2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncP2"])
        sncP3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncP3"])
        sncP4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sncP4"])
        inorg_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["inorg_n"])
        inorg_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["inorg_p"])
        sorbed_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sorbed_n"])
        sorbed_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["sorbed_p"])
        hresp[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["hresp"])
        nmin[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["nmin"])
        pmin[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out["pmin"])
        print_progress(i + 1,
                       len(dates),
                       prefix='Progress:',
                       suffix='Complete')
    # write netcdf
    csoil = csoil1 + csoil2 + csoil3 + csoil4
    org_n = sncN1 + sncN2 + sncN3 + sncN4
    org_p = sncP1 + sncP2 + sncP3 + sncP4
    inorg_n = sorbed_n + inorg_n

    vars = ['csoil', 'org_n', 'org_p', 'inorg_n',
            'inorg_p', 'sorbed_p', 'hresp', 'nmin', 'pmin']
    arr = (csoil, org_n, org_p, inorg_n, inorg_p, sorbed_p, hresp, nmin, pmin)
    var_attrs = get_var_metadata(vars, dm1)
    write_daily_output(arr, vars, var_attrs, interval)


def create_ncG3(table, interval):
    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    vars = ["rcm", "runom", "evapm", "wsoil", "cleaf", "cawood",
            "cfroot", "litter_l", "cwd", "litter_fr", "litter_n",
            "litter_p", "sto_c", "sto_n", "sto_p", "c_cost"]

    dates = time_queries(interval)
    dm1 = len(dates)

    rcm = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    runom = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    evapm = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    wsoil = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    swsoil = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    cleaf = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    cawood = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    cfroot = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    litter_l = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    cwd = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    litter_fr = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc5 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    lnc6 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sto1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sto2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    sto3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)
    c_cost = np.zeros(shape=(dm1, 61, 71), dtype=np.float32)

    print("\nQuerying data from file FOR", end=': ')
    for v in vars:
        print(v, end=", ")
    print("\nInterval: ", interval)
    print_progress(0, len(dates), prefix='Progress:', suffix='Complete')
    for i, day in enumerate(dates):
        out = table.read_where(day)
        rcm[i, :, :] = assemble_layer(out['grid_y'], out['grid_x'], out['rcm'])
        runom[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['runom'])
        evapm[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['evapm'])
        wsoil[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wsoil'])
        swsoil[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['swsoil'])
        cleaf[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['cleaf'])
        cawood[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['cawood'])
        cfroot[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['cfroot'])
        litter_l[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['litter_l'])
        cwd[i, :, :] = assemble_layer(out['grid_y'], out['grid_x'], out['cwd'])
        litter_fr[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['litter_fr'])
        lnc1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc1'])
        lnc2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc2'])
        lnc3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc3'])
        lnc4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc4'])
        lnc5[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc5'])
        lnc6[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['lnc6'])
        sto1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['sto1'])
        sto2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['sto2'])
        sto3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['sto3'])
        c_cost[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['c_cost'])
        print_progress(i + 1,
                       len(dates),
                       prefix='Progress:',
                       suffix='Complete')
    # write netcdf
    litter_n = lnc1 + lnc2 + lnc3
    litter_p = lnc4 + lnc5 + lnc6
    wsoil = swsoil + wsoil

    vars = ["rcm", "runom", "evapm", "wsoil", "cleaf", "cawood",
            "cfroot", "litter_l", "cwd", "litter_fr", "litter_n",
            "litter_p", "sto_c", "sto_n", "sto_p", "c_cost"]

    arr = (rcm, runom, evapm, wsoil, cleaf, cawood, cfroot,
           litter_l, cwd, litter_fr, litter_n, litter_p,
           sto1, sto2, sto3, c_cost)

    var_attrs = get_var_metadata(vars, dm1)
    write_daily_output(arr, vars, var_attrs, interval)


def create_nc_area(table):
    pass


if __name__ == "__main__":

    interval1 = ('19790101', '19790115')
    interval2 = ('19900101', '19991231')
    interval3 = ('20000101', '20151231')

    print("Loading file...")
    with tb.open_file("/d/c1/homes/amazonfaceme/jpdarela/CAETE/out_test/CAETE_100_2.h5", driver="H5FD_CORE") as h5f:
        print('Loaded')

        g1_table = h5f.root.RUN0.Outputs_G1
        g2_table = h5f.root.RUN0.Outputs_G2
        g3_table = h5f.root.RUN0.Outputs_G3

        create_ncG1(g1_table, interval1)
        create_ncG2(g2_table, interval1)
        create_ncG3(g3_table, interval1)
