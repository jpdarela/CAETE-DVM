import os
import datetime
from pathlib import Path
import pickle
import pandas
import bz2
import cftime


# periods 1961-1990; 2041-2070
# models

MODS = ["GFDL-ESM2M", "HadGEM2-ES", "IPSL-CM5A-LR", "MIROC5"]
SCEN = ["historical", "rcp26", "rcp60", "rcp85"]

# Convert prec to kg m-2 day-1
precConvFactor = 8.64e4

coord = pandas.read_csv(
    "/home/joao/Desktop/CAETE-DVM/input/task5/task5_coordinates.csv",
    index_col="NM_MUNICIP")

MUNICIPIOS = list(coord.index)

# Output data
outputPath = Path("/home/joao/Desktop/CAETE-DVM/input/task5/RClimDex")
os.makedirs(outputPath, exist_ok=True)

cwd = os.getcwd
ROOT = cwd()

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

for model in MODS:
    # iter over models
    data_path = Path(f"../task5/CMIP5_ISIMIP2b/{model}")
    print(data_path)
    for scen in SCEN:
        # iter over scenarios
        scen_path = Path(os.path.join(data_path, Path(f"{model}_{scen}")))
        
        # go to scen folder        
        os.chdir(scen_path)
        
        # read time metadata
        metadata = Path(f"{model}-{scen}_METADATA.pbz2")
        with bz2.BZ2File(metadata, mode='r') as fh:
            ancillData = pickle.load(fh)
        calendar = ancillData[0]['calendar']
        time_unit = ancillData[0]['units']
        tindex = ancillData[0]['time_index'][:]

        # Set date ranges 1
        start = cftime.num2pydate(tindex[0], time_unit, calendar)
        end = cftime.num2pydate(tindex[-1], time_unit, calendar)            
        idx = pandas.date_range(strDate(start), strDate(end), freq="D")        

        # Iterate over gridpoints
        for muni in MUNICIPIOS:
            # Read Climatic Data
            nx = int(coord.loc(0)[muni]['xindex'])
            ny = int(coord.loc(0)[muni]['yindex'])
            filename = Path(f"input_data_{ny}-{nx}.pbz2")
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
            
            # Write to file
            final_filename = Path(os.path.join(outputPath.resolve(), f"RClimDex-DATA_{muni}_{model}-{scen}.csv"))
            to_write.to_csv(final_filename, header=False, index=False)
            print(final_filename)
            print(outputPath.exists())
        # Return to ROOT
        os.chdir(ROOT)
