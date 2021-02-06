import sys
import numpy as np
import tables as tb
from pandas import date_range

from post_processing import cf_date2str, str2cf_date
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


def create_npyG1(table, interval):

    vars = ['photo', 'aresp', 'npp', 'lai', 'wue', 'cue', 'vcmax', 'sla']

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
        print_progress(i + 1,
                       len(dates),
                       prefix='Progress:',
                       suffix='Complete')
    # write netcdf TODO
    # f
    return dict(zip(vars, (photo, aresp, npp, lai, wue, cue, vcmax, specific_la)))


interval1 = ('19790101', '19791231')
interval2 = ('19900101', '19991231')
interval3 = ('20000101', '20151231')

longitude = np.arange(-179.75, 180, 0.5)[201:272]
latitude = np.arange(-89.75, 90, 0.5)[160:221]

time_units = "days since 1979-01-01"
calendar = 'proleptic_gregorian'
nt = 200
npls = 2342324522345

vunits = {'header': ['long_name', 'unit', 'standart_name', 'ldim'],

          'rsds': ['short_wav_rad_down', 'W m-2', 'rsds', nt],
          'wind': ['wind_velocity', 'm s-1', 'wind', nt],
          'ps': ['sur_pressure', 'Pa', 'ps', nt],
          'tas': ['sur_temperature_2m', 'celcius', 'tas', nt],
          'tsoil': ['soil_temperature', 'celcius', 'soil_temp', nt],
          'pr': ['precipitation', 'Kg m-2 month-1', 'pr', nt],
          'wsoil': ['soil_water_content-wsoil', 'kg m-2', 'mrso', nt],
          'evapm': ['evapotranpiration', 'kg m-2 day-1', 'et', nt],
          'emaxm': ['potent. evapotranpiration', 'kg m-2 day-1', 'etpot', nt],
          'runom': ['total_runoff', 'kg m-2 day-1', 'mrro', nt],
          'aresp': ['autothrophic respiration', 'kg m-2 year-1', 'ar', nt],
          'photo': ['photosynthesis', 'kg m-2 year-1', 'ph', nt],
          'npp': ['net primary productivity', 'kg m-2 year-1', 'npp', nt],
          'lai': ['Leaf Area Index', 'm2 m-2', 'LAI', nt],
          'rcm': ['stomatal resistence', 's m-1', 'rcm', nt],
          'hresp': ['heterothrophic respiration', 'kg m-2 year-1', 'hr', nt],
          'clit': ['Litter Carbon', 'Kg m-2', 'clit', nt],
          'csoil': ['Soil Carbon', 'Kg m-2', 'csoil', nt],
          'rm': ['maintenance respiration', 'kg m-2 year-1', 'rm', nt],
          'rg': ['growth respiration', 'kg m-2 year-1', 'rg', nt],
          'wue': ['water use efficiency', 'ad', 'wue', nt],
          'cue': ['Carbon use efficiency', 'ad', 'cue', nt],
          'cawood': ['C in abovewgrownd wood', 'kg m-2', 'cawood', npls],
          'cfroot': ['C in fine roots', 'kg m-2', 'cfroot', npls],
          'cleaf': ['C in leaves', 'kg m-2', 'cleaf', npls],
          'area': ['occupation coefficient', '%', 'area', npls],
          'area0': ['occupation coefficient', '%', 'init_area', npls],
          'cwin': ['init C in abovewgrownd wood', 'kg m-2', 'cawood', npls],
          'cfin': ['init C in fine roots', 'kg m-2', 'cfroot', npls],
          'cmass': ['total Carbon -Biomass', 'kg m-2', 'cmass', npls],
          'clin': ['init C in leaves', 'kg m-2 ', 'cleaf', npls]}

# print("Loading file...")
# with tb.open_file("/d/c1/homes/amazonfaceme/jpdarela/CAETE/out_test/CAETE_100_2.h5", driver="H5FD_CORE") as h5f:
#     print('Loaded')

#     g1_table = h5f.root.RUN0.Outputs_G1

#     out_data = create_npyG1(g1_table, interval1)

#     # np.save('npp19790101-19891231.npy', out_data)
