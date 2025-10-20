from pathlib import Path
import sys

sys.path.append('../src')


import _geos as geos  # type: ignore
from caete import read_bz2_file # type: ignore
from config import fetch_config # type: ignore

CFG = fetch_config()

def find_file(input_folder: str | Path, lat:float, lon:float) -> str:
    """Finds the appropriate input file for the given latitude and longitude.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.

    Returns:
        str: The filename corresponding to the given coordinates.
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    try:
        idx = geos.find_indices_xy(N=lat, W=lon, res_y = CFG.crs.yres, res_x = CFG.crs.xres)
    except Exception as e:
        raise ValueError(f"Error finding indices for lat {lat}, lon {lon}: {e}")

    f =  input_path / f"input_data_{idx[0]}-{idx[1]}.pbz2"

    if not f.exists():
        raise FileNotFoundError(f"Input file {f} does not exist for lat {lat}, lon {lon}.")
    
    return f


def read_file(file_path: str | Path) -> dict:
    """Reads the input file and returns its contents as a dictionary.
    Args:
        file_path (str | Path): Path to the input file.
    Returns:
        dict: Contents of the input file.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {file_path} does not exist.")

    return read_bz2_file(input_path)