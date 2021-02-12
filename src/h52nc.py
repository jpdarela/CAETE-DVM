import os
from pathlib import Path

import numpy as np
import tables as tb
from pandas import date_range
import cftime
from netCDF4 import Dataset as dt

from post_processing import cf_date2str, str2cf_date
from caete_module import global_par as gp
from caete import NO_DATA, print_progress, run_breaks

TIME_UNITS = u"days since 1979-01-01"
CALENDAR = u"proleptic_gregorian"


def build_strd(strd):
    return f"""(date == b'{strd}')"""


def build_strds(strd):
    return f"""(start_date == b'{strd}')"""


def assemble_layer(ny, nx, var):
    out = np.zeros(shape=(360, 720), dtype=np.float32) - 9999.0

    for i, val in enumerate(var):
        out[ny[i], nx[i]] = val

    return out[160:221, 201:272]


def assemble_cwm(ny, nx, area, pls_arr):
    out = np.zeros(shape=(360, 720), dtype=np.float32) - 9999.0

    for x in range(area.shape[0]):
        out[ny[x], nx[x]] = np.sum(area[x, :] * pls_arr)
    return out[160:221, 201:272]


def assemble_layer_area(ny, nx, var):
    out = np.zeros(shape=(gp.npls, 360, 720), dtype=np.float32) - 9999.0
    # var shape = ny*nx , npls
    for x in range(var.shape[0]):
        out[:, ny[x], nx[x]] = var[x, :]
    return out[:, 160:221, 201:272]


def time_queries(interval):
    sdate = str2cf_date(interval[0])
    edate = str2cf_date(interval[1])

    dates = date_range(sdate, edate)
    query_days = []
    for day in dates:
        query_days.append(build_strd(cf_date2str(day)))
    return query_days


def get_var_metadata(var):

    vunits = {'header': ['long_name', 'unit', 'standart_name'],

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
              'sla': ['specfic leaf area', 'm-2 g-1', 'SLA'],
              'cue': ['Carbon use efficiency', 'unitless', 'cue'],
              'cawood': ['C in woody tissues', 'kg m-2', 'cawood'],
              'cfroot': ['C in fine roots', 'kg m-2', 'cfroot'],
              'cleaf': ['C in leaves', 'kg m-2', 'cleaf'],
              'cmass': ['total Carbon -Biomass', 'kg m-2', 'cmass'],
              'leaf_nolim': ['no lim. in leaf growth', 'Time fraction', 'leaf_nolim'],
              'leaf_lim_n': ['N lim. growth L', 'Time fraction', 'leaf_lim_n'],
              'leaf_lim_p': ['P lim. growth L', 'Time fraction', 'leaf_lim_p'],
              'leaf_colim_n': ['colim N L', 'Time fraction', 'leaf_colim_n'],
              'leaf_colim_p': ['colim P L', 'Time fraction', 'leaf_colim_p'],
              'leaf_colim_np': ['colim NP L', 'Time fraction', 'leaf_colim_np'],
              'wood_nolim': ['no lim. in wood growth', 'Time fraction', 'wood_nolim'],
              'wood_lim_n': ['N lim. growth W', 'Time fraction', 'wood_lim_n'],
              'wood_lim_p': ['P lim. growth W', 'Time fraction', 'wood_lim_p'],
              'wood_colim_n': ['colim N W', 'Time fraction', 'wood_colim_n'],
              'wood_colim_p': ['colim P W', 'Time fraction', 'wood_colim_p'],
              'wood_colim_np': ['colim NP W', 'Time fraction', 'wood_colim_np'],
              'root_nolim': ['no lim. in root growth', 'Time fraction', 'root_nolim'],
              'root_lim_n': ['N lim. growth R', 'Time fraction', 'root_lim_n'],
              'root_lim_p': ['P lim. growth R', 'Time fraction', 'root_lim_p'],
              'root_colim_n': ['colim N R', 'Time fraction', 'root_colim_n'],
              'root_colim_p': ['colim P R', 'Time fraction', 'root_colim_p'],
              'root_colim_np': ['colim NP R', 'Time fraction', 'root_colim_np'],
              'upt_stratN0': ['passive N uptake', 'Time fraction', 'upt_stratN0'],
              'upt_stratN1': ['nma', 'Time fraction', 'upt_stratN1'],
              'upt_stratN2': ['nme', 'Time fraction', 'upt_stratN2'],
              'upt_stratN3': ['am', 'Time fraction', 'upt_stratN3'],
              'upt_stratN4': ['em', 'Time fraction', 'upt_stratN4'],
              'upt_stratN6': ['am0', 'Time fraction', 'upt_stratN6'],
              'upt_stratP0': ['passive P uptake', 'Time fraction', 'upt_stratP0'],
              'upt_stratP1': ['nma', 'Time fraction', 'upt_stratP1'],
              'upt_stratP2': ['nme', 'Time fraction', 'upt_stratP2'],
              'upt_stratP3': ['am', 'Time fraction', 'upt_stratP3'],
              'upt_stratP4': ['em', 'Time fraction', 'upt_stratP4'],
              'upt_stratP5': ['ram-ap', 'Time fraction', 'upt_stratP5'],
              'upt_stratP6': ['rem-ap', 'Time fraction', 'upt_stratP6'],
              'upt_stratP7': ['amap', 'Time fraction', 'upt_stratP7'],
              'upt_stratP8': ['em0x', 'Time fraction', 'upt_stratP8'],
              'g1': ['CWM- G1 param for Stomat.Resist  model', 'unitless', 'g1'],
              'resopfrac': ['CWM- Leaf resorpton fractio', '%', 'resopfrac'],
              'tleaf': ['CWM- leaf turnover time', 'years', 'tleaf'],
              'twood': ['CWM- wood turnover time', 'years', 'twood'],
              'troot': ['CWM- fine root turnover time', 'years', 'troot'],
              'aleaf': ['CWM- allocation coefficients for leaf', 'unitless', 'aleaf'],
              'awood': ['CWM- allocation coefficients for wood', 'unitless', 'awood'],
              'aroot': ['CWM- allocation coefficients for root', 'unitless', 'aroot'],
              'c4': ['CWM- c4 photosynthesis pathway', 'unitless', 'c4'],
              'leaf_n2c': ['CWM- leaf N:C', 'g g-1', 'leaf_n2c'],
              'awood_n2c': ['CWM- wood tissues N:C', 'g g-1', 'awood_n2c'],
              'froot_n2c': ['CWM- fine root N:C', 'g g-1', 'froot_n2c'],
              'leaf_p2c': ['CWM- leaf P:C', 'g g-1', 'leaf_p2c'],
              'awood_p2c': ['CWM- wood tissues P:C', 'g g-1', 'awood_p2c'],
              'froot_p2c': ['CWM- fine root P:C', 'g g-1', 'froot_p2c'],
              'amp': ['CWM- Percentage of fine root colonized by AM', '%', 'amp'],
              'pdia': ['CWM- NPP aloated to N fixers', 'fraction_of_npp', 'pdia']}

    out = {}
    for v in var:
        out[v] = vunits[v]
    return out


def write_daily_output(arr, var, flt_attrs, interval, experiment="TEST RUN HISTORICAL ISIMIP"):

    NO_DATA = [-9999.0, -9999.0]

    time_units = TIME_UNITS
    calendar = CALENDAR
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
            var_[:, :, :] = np.ma.masked_array(
                arr[i], mask=arr[i] == NO_DATA[0])
            print_progress(i + 1, len(var), prefix='Progress:',
                           suffix='Complete')


def write_snap_output(arr, var, flt_attrs, time_index, experiment="TEST RUN HISTORICAL ISIMIP"):

    NO_DATA = [-9999.0, -9999.0]

    time_units = TIME_UNITS
    calendar = CALENDAR
    nc_out = Path("../nc_outputs")

    time_dim = time_index

    longitude_0 = np.arange(-179.75, 180, 0.5)[201:272]
    latitude_0 = np.arange(89.75, -90, -0.5)[160:221]
    print("\nSaving netCDF4 files")
    print_progress(0, len(var), prefix='Progress:', suffix='Complete')

    for i, v in enumerate(var):
        nc_filename = os.path.join(nc_out, Path(f'{v}.nc4'))
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
            var_[:, :, :] = np.ma.masked_array(
                arr[i], mask=arr[i] == NO_DATA[0])
            print_progress(i + 1, len(var), prefix='Progress:',
                           suffix='Complete')


def write_area_output(arr, time_index, experiment="TEST RUN HISTORICAL ISIMIP"):
    NO_DATA = [-9999.0, -9999.0]

    time_units = TIME_UNITS
    calendar = CALENDAR
    nc_out = Path("../nc_outputs")

    time_dim = time_index

    longitude_0 = np.arange(-179.75, 180, 0.5)[201:272]
    latitude_0 = np.arange(89.75, -90, -0.5)[160:221]

    nc_filename = os.path.join(nc_out, Path(f'ocp_area.nc4'))
    with dt(nc_filename, mode='w', format='NETCDF4') as rootgrp:
        # dimensions  & variables

        rootgrp.createDimension("latitude", latitude_0.size)
        rootgrp.createDimension("longitude", longitude_0.size)
        rootgrp.createDimension("pls", arr.shape[1])
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
        var_[:, :, :, :] = np.ma.masked_array(arr, mask=arr == NO_DATA[0])


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

    photo = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    aresp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    npp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lai = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wue = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    cue = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    vcmax = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    specific_la = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    nupt1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    nupt2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    pupt1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    pupt2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    pupt3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0

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
    np.place(nupt1, mask=nupt2 == -9999.0, vals=NO_DATA)
    pupt1 = pupt3 + pupt2 + pupt1
    np.place(pupt1, mask=pupt2 == -9999.0, vals=NO_DATA)

    vars = ['photo', 'aresp', 'npp', 'lai', 'wue', 'cue',
            'vcmax', 'sla', 'nupt', 'pupt']

    arr = (photo, aresp, npp, lai, wue, cue, vcmax,
           specific_la, nupt1, pupt1)
    var_attrs = get_var_metadata(vars)
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

    csoil1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    csoil2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    csoil3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    csoil4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncN1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncN2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncN3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncN4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncP1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncP2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncP3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sncP4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    inorg_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    inorg_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sorbed_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sorbed_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    hresp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    nmin = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    pmin = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0

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
    np.place(csoil, mask=csoil1 == -9999.0, vals=NO_DATA)
    org_n = sncN1 + sncN2 + sncN3 + sncN4
    np.place(org_n, mask=sncN1 == -9999.0, vals=NO_DATA)
    org_p = sncP1 + sncP2 + sncP3 + sncP4
    np.place(org_p, mask=sncP1 == -9999.0, vals=NO_DATA)
    inorg_n = sorbed_n + inorg_n
    np.place(inorg_n, mask=sorbed_n == -9999.0, vals=NO_DATA)

    vars = ['csoil', 'org_n', 'org_p', 'inorg_n',
            'inorg_p', 'sorbed_p', 'hresp', 'nmin', 'pmin']
    arr = (csoil, org_n, org_p, inorg_n, inorg_p, sorbed_p, hresp, nmin, pmin)
    var_attrs = get_var_metadata(vars)
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

    rcm = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    runom = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    evapm = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wsoil = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    swsoil = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    cleaf = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    cawood = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    cfroot = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    litter_l = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    cwd = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    litter_fr = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc5 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    lnc6 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sto1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sto2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    sto3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    c_cost = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0

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
    np.place(litter_n, mask=lnc1 == -9999.0, vals=NO_DATA)
    litter_p = lnc4 + lnc5 + lnc6
    np.place(litter_p, mask=lnc4 == -9999.0, vals=NO_DATA)
    wsoil = swsoil + wsoil
    np.place(wsoil, mask=swsoil == -9999.0, vals=NO_DATA)

    vars = ["rcm", "runom", "evapm", "wsoil", "cleaf", "cawood",
            "cfroot", "litter_l", "cwd", "litter_fr", "litter_n",
            "litter_p", "sto_c", "sto_n", "sto_p", "c_cost"]

    arr = (rcm, runom, evapm, wsoil, cleaf, cawood, cfroot,
           litter_l, cwd, litter_fr, litter_n, litter_p,
           sto1, sto2, sto3, c_cost)

    var_attrs = get_var_metadata(vars)
    write_daily_output(arr, vars, var_attrs, interval)


def lim_data(table):

    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    dm1 = len(run_breaks)

    leaf_nolim = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_lim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_lim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_colim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_colim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_colim_np = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_nolim = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_lim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_lim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_colim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_colim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    wood_colim_np = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_nolim = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_lim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_lim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_colim_n = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_colim_p = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    root_colim_np = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    time_index = []

    for i, interval in enumerate(run_breaks):
        sdate = str2cf_date(interval[1])
        time_index.append(int(cftime.date2num(sdate, TIME_UNITS, CALENDAR)))

        out = table.read_where(build_strds(interval[0]))
        leaf_nolim[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_nolim'])
        leaf_lim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_lim_n'])
        leaf_lim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_lim_p'])
        leaf_colim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_colim_n'])
        leaf_colim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_colim_p'])
        leaf_colim_np[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['leaf_colim_np'])
        wood_nolim[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_nolim'])
        wood_lim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_lim_n'])
        wood_lim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_lim_p'])
        wood_colim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_colim_n'])
        wood_colim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_colim_p'])
        wood_colim_np[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['wood_colim_np'])
        root_nolim[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_nolim'])
        root_lim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_lim_n'])
        root_lim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_lim_p'])
        root_colim_n[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_colim_n'])
        root_colim_p[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_colim_p'])
        root_colim_np[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['root_colim_np'])

    vars = ['leaf_nolim',
            'leaf_lim_n',
            'leaf_lim_p',
            'leaf_colim_n',
            'leaf_colim_p',
            'leaf_colim_np',
            'wood_nolim',
            'wood_lim_n',
            'wood_lim_p',
            'wood_colim_n',
            'wood_colim_p',
            'wood_colim_np',
            'root_nolim',
            'root_lim_n',
            'root_lim_p',
            'root_colim_n',
            'root_colim_p',
            'root_colim_np']

    arr = [leaf_nolim,
           leaf_lim_n,
           leaf_lim_p,
           leaf_colim_n,
           leaf_colim_p,
           leaf_colim_np,
           wood_nolim,
           wood_lim_n,
           wood_lim_p,
           wood_colim_n,
           wood_colim_p,
           wood_colim_np,
           root_nolim,
           root_lim_n,
           root_lim_p,
           root_colim_n,
           root_colim_p,
           root_colim_np]

    flt_attrs = get_var_metadata(vars)
    print("Saving growth limitation outputs")
    write_snap_output(arr, vars, flt_attrs, time_index)


def ustrat_data(table):
    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    dm1 = len(run_breaks)

    upt_stratN0 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratN1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratN2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratN3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratN4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratN6 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP0 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP2 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP3 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP5 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP6 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP7 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    upt_stratP8 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    time_index = []

    for i, interval in enumerate(run_breaks):
        sdate = str2cf_date(interval[1])
        time_index.append(int(cftime.date2num(sdate, TIME_UNITS, CALENDAR)))

        out = table.read_where(build_strds(interval[0]))
        upt_stratN0[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN0'])
        upt_stratN1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN1'])
        upt_stratN2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN2'])
        upt_stratN3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN3'])
        upt_stratN4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN4'])
        upt_stratN6[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratN6'])
        upt_stratP0[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP0'])
        upt_stratP1[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP1'])
        upt_stratP2[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP2'])
        upt_stratP3[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP3'])
        upt_stratP4[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP4'])
        upt_stratP5[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP5'])
        upt_stratP6[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP6'])
        upt_stratP7[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP7'])
        upt_stratP8[i, :, :] = assemble_layer(
            out['grid_y'], out['grid_x'], out['upt_stratP8'])

    vars = ['upt_stratN0',
            'upt_stratN1',
            'upt_stratN2',
            'upt_stratN3',
            'upt_stratN4',
            'upt_stratN6',
            'upt_stratP0',
            'upt_stratP1',
            'upt_stratP2',
            'upt_stratP3',
            'upt_stratP4',
            'upt_stratP5',
            'upt_stratP6',
            'upt_stratP7',
            'upt_stratP8']

    arr = [upt_stratN0,
           upt_stratN1,
           upt_stratN2,
           upt_stratN3,
           upt_stratN4,
           upt_stratN6,
           upt_stratP0,
           upt_stratP1,
           upt_stratP2,
           upt_stratP3,
           upt_stratP4,
           upt_stratP5,
           upt_stratP6,
           upt_stratP7,
           upt_stratP8]

    flt_attrs = get_var_metadata(vars)
    print("Saving uptake strategy outputs")
    write_snap_output(arr, vars, flt_attrs, time_index)


def ccc(table, pls_table):

    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    dm1 = len(run_breaks) + 1
    g1 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    resopfrac = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    tleaf = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    twood = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    troot = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    aleaf = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    awood = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    aroot = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    c4 = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_n2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    awood_n2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    froot_n2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    leaf_p2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    awood_p2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    froot_p2c = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    amp = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0
    pdia = np.zeros(shape=(dm1, 61, 71), dtype=np.float32) - 9999.0

    time_index = []
    pls_array = pls_table.read_where("PLS_id >= 0")
    for i, interval in enumerate(run_breaks):
        out = table.read_where(build_strds(interval[0]))
        if i == 0:
            a0date = str2cf_date(interval[0])
            time_index.append(
                int(cftime.date2num(a0date, TIME_UNITS, CALENDAR)))
            afdate = str2cf_date(interval[1])
            time_index.append(
                int(cftime.date2num(afdate, TIME_UNITS, CALENDAR)))
            g1[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['g1'])
            resopfrac[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['resopfrac'])
            tleaf[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['tleaf'])
            twood[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['twood'])
            troot[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['troot'])
            aleaf[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['aleaf'])
            awood[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['awood'])
            aroot[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['aroot'])
            c4[i, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                       out['area_0'], pls_array['c4'])
            leaf_n2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['leaf_n2c'])
            awood_n2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['awood_n2c'])
            froot_n2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['froot_n2c'])
            leaf_p2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['leaf_p2c'])
            awood_p2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['awood_p2c'])
            froot_p2c[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['froot_p2c'])
            amp[i, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                        out['area_0'], pls_array['amp'])
            pdia[i, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_0'], pls_array['pdia'])

            g1[i + 1, :, :] = assemble_cwm(out['grid_y'],
                                           out['grid_x'], out['area_f'], pls_array['g1'])
            resopfrac[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['resopfrac'])
            tleaf[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['tleaf'])
            twood[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['twood'])
            troot[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['troot'])
            aleaf[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['aleaf'])
            awood[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood'])
            aroot[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['aroot'])
            c4[i + 1, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                           out['area_f'], pls_array['c4'])
            leaf_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['leaf_n2c'])
            awood_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood_n2c'])
            froot_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['froot_n2c'])
            leaf_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['leaf_p2c'])
            awood_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood_p2c'])
            froot_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['froot_p2c'])
            amp[i + 1, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                            out['area_f'], pls_array['amp'])
            pdia[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['pdia'])
        else:
            afdate = str2cf_date(interval[1])
            time_index.append(
                int(cftime.date2num(afdate, TIME_UNITS, CALENDAR)))
            g1[i + 1, :, :] = assemble_cwm(out['grid_y'],
                                           out['grid_x'], out['area_f'], pls_array['g1'])
            resopfrac[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['resopfrac'])
            tleaf[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['tleaf'])
            twood[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['twood'])
            troot[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['troot'])
            aleaf[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['aleaf'])
            awood[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood'])
            aroot[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['aroot'])
            c4[i + 1, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                           out['area_f'], pls_array['c4'])
            leaf_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['leaf_n2c'])
            awood_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood_n2c'])
            froot_n2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['froot_n2c'])
            leaf_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['leaf_p2c'])
            awood_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['awood_p2c'])
            froot_p2c[i + 1, :, :] = assemble_cwm(
                out['grid_y'], out['grid_x'], out['area_f'], pls_array['froot_p2c'])
            amp[i + 1, :, :] = assemble_cwm(out['grid_y'], out['grid_x'],
                                            out['area_f'], pls_array['amp'])

    arr = [g1,
           resopfrac,
           tleaf,
           twood,
           troot,
           aleaf,
           awood,
           aroot,
           c4,
           leaf_n2c,
           awood_n2c,
           froot_n2c,
           leaf_p2c,
           awood_p2c,
           froot_p2c,
           amp,
           pdia]

    vars = ['g1',
            'resopfrac',
            'tleaf',
            'twood',
            'troot',
            'aleaf',
            'awood',
            'aroot',
            'c4',
            'leaf_n2c',
            'awood_n2c',
            'froot_n2c',
            'leaf_p2c',
            'awood_p2c',
            'froot_p2c',
            'amp',
            'pdia']

    flt_attrs = get_var_metadata(vars)
    write_snap_output(arr, vars, flt_attrs, time_index)


def create_nc_area(table):

    nc_out = Path("../nc_outputs")
    out_data = True if nc_out.exists() else os.mkdir(nc_out)
    if out_data is None:
        print(f"Creating output folder at{nc_out.resolve()}")
    elif out_data:
        print(f"Saving outputs in {nc_out.resolve()}")

    dm1 = len(run_breaks) + 1

    area = np.zeros(shape=(dm1, gp.npls, 61, 71), dtype=np.float32) - 9999.0

    time_index = []

    for i, interval in enumerate(run_breaks):
        out = table.read_where(build_strds(interval[0]))
        if i == 0:
            a0date = str2cf_date(interval[0])
            time_index.append(
                int(cftime.date2num(a0date, TIME_UNITS, CALENDAR)))
            afdate = str2cf_date(interval[1])
            time_index.append(
                int(cftime.date2num(afdate, TIME_UNITS, CALENDAR)))
            area[i, :, :, :] = assemble_layer_area(
                out['grid_y'], out['grid_x'], out['area_0'])
            area[i + 1, :, :,
                 :] = assemble_layer_area(out['grid_y'], out['grid_x'], out['area_f'])
        else:
            afdate = str2cf_date(interval[1])
            time_index.append(
                int(cftime.date2num(afdate, TIME_UNITS, CALENDAR)))
            area[i + 1, :, :,
                 :] = assemble_layer_area(out['grid_y'], out['grid_x'], out['area_f'])

    write_area_output(area, time_index)


def h52nc(input_file):

    import time

    interval1 = (run_breaks[0][0], '19891231')
    interval2 = ('19900101', '19991231')
    interval3 = ('20000101', run_breaks[-1][-1])


    print("Loading file...")
    h5f = tb.open_file(input_file, mode='a', driver="H5FD_CORE")
    print('Loaded')

    g1_table = h5f.root.RUN0.Outputs_G1
    print('Creating Sorted table for g1', time.ctime())
    index_dt = g1_table.cols.date.create_csindex()
    t1d = g1_table.copy(newname='indexedT1date', sortby=g1_table.cols.date)
    # t1d = h5f.root.RUN0.indexedT1date

    g2_table = h5f.root.RUN0.Outputs_G2
    print('Creating Sorted table for g2', time.ctime())
    index_dt = g2_table.cols.date.create_csindex()
    t2d = g2_table.copy(newname='indexedT2date', sortby=g2_table.cols.date)
    # t2d = h5f.root.RUN0.indexedT2date

    g3_table = h5f.root.RUN0.Outputs_G3
    print('Creating Sorted table for g3', time.ctime())
    index_dt = g3_table.cols.date.create_csindex()
    t3d = g3_table.copy(newname='indexedT3date', sortby=g3_table.cols.date)
    # t3d = h5f.root.RUN0.indexedT3date


    create_ncG1(t1d, interval1)
    create_ncG1(t1d, interval2)
    create_ncG1(t1d, interval3)
    create_ncG2(t2d, interval1)
    create_ncG2(t2d, interval2)
    create_ncG2(t2d, interval3)
    create_ncG3(t3d, interval1)
    create_ncG3(t3d, interval2)
    create_ncG3(t3d, interval3)

    snap_table = h5f.root.RUN0.spin_snapshot
    lim_data(snap_table)
    ustrat_data(snap_table)
    create_nc_area(snap_table)

    pls_table = h5f.root.RUN0.PLS
    ccc(snap_table, pls_table)

    h5f.close()
