import os
import datetime
from pathlib import Path
import pickle
import bz2

import cftime
import numpy as np
import pandas



# historical_simulation_start = datetime.datetime(1850, 1, 1, 0, 0)
# historical_simulation_end = datetime.datetime(2014, 12, 31, 0, 0)

# future_simulation_start = datetime.datetime(2015, 1, 1, 0, 0)
# future_simulation_end = datetime.datetime(2100, 12, 31, 0, 0)


MODS = ["MPI-ESM1-2-HR",]
SCEN = ["historical", "ssp370", "ssp585"]

# Convert prec to kg m-2 day-1
precConvFactor = 8.64e4

coord = pandas.read_csv(
    "gridlist_with_idx_7e1f255d9531480cb93f2cc70edbe5b6.csv",
    index_col="name")

MUNICIPIOS = list(coord.index)

# Output data
outputPath = Path("./rclimdex/")
os.makedirs(outputPath, exist_ok=True)


def strDate(dtobject: cftime.real_datetime)->str:
    return dtobject.strftime("%Y%m%d")


def openBZ2(filepath:Path)-> dict:
    with bz2.BZ2File(filepath, mode='r') as fh:
        return pickle.load(fh)


def openFile(filepath:Path) -> tuple:
    dt = openBZ2(filepath)
    tasmax = dt['tas'] + 2.0
    tasmin = dt['tas'] - 2.0
    pr = dt['pr']

    return (pr, tasmax, tasmin)


def main(model, scen):

    # iter over models
    data_path = Path(f"./{model}").resolve()
    scen_path = data_path / scen
    print(scen_path.resolve())
    metadata = scen_path / "ISIMIP_HISTORICAL_METADATA.pbz2"
    print(metadata.resolve().exists())
    with bz2.BZ2File(metadata, mode='r') as fh:
        ancillData = pickle.load(fh)

    calendar = ancillData[0]['calendar']
    time_unit = ancillData[0]['units']
    ndays = ancillData[0]['time_index'][:].size
    range_days = np.arange(ndays)
    tindex = cftime.num2date(range_days, time_unit, calendar)
    start = tindex[0]
    end = tindex[-1]
    idx = pandas.date_range(strDate(start), strDate(end), freq="D")

    # return idx, start, end
    # Iterate over gridpoints
    for muni in MUNICIPIOS:
        # Read Climatic Data
        nx = int(coord.loc(0)[muni]['x'])
        ny = int(coord.loc(0)[muni]['y'])
        filename = scen_path/Path(f"input_data_{ny}-{nx}.pbz2")
        pr, tasmax, tasmin = openFile(filename)

        # Write a dict with the available range
        dataDict = {"YEAR" : idx.year,
                    "MONTH": idx.month,
                    "DAY"  : idx.day,
                    "PRCP" : pandas.Series(pr, index=idx) * precConvFactor,
                    "TMAX" : pandas.Series(tasmax, index=idx) - 273.15,
                    "TMIN" : pandas.Series(tasmin, index=idx) - 273.15}

        df = pandas.DataFrame(dataDict)

        # Set data ranges 2
        Start = datetime.datetime(1961, 1, 1, 0, 0) \
            if scen == "historical" \
                else datetime.datetime(2041, 1, 1, 0, 0)

        End = datetime.datetime(1990, 12, 31, 0, 0) \
            if scen == "historical" else \
                datetime.datetime(2070, 12, 31, 0, 0)

        # The final dict
        to_write = df[Start:End]

        final_filename = Path(os.path.join(outputPath.resolve(), f"RClimDex-DATA_{muni}_{model}-{scen}.csv"))
        to_write.to_csv(final_filename, header=False, index=False)


if __name__ == "__main__":
    for model in MODS:
        for scen in SCEN:
            print(model, scen)
            main(model, scen)
