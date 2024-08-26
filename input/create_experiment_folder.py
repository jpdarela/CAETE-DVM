
# REad the gridlist_cities.csv and use the lat and lon to find the indices of the grid cells
# Then copy the files from the main directories to the new directories based on the indices
# A new gridlist_cities_idx.csv file is created with the indices of the grid cells
# This file can identify the outputs of the model with the location of the cities
# The input files are copied to the new directories based on the indices

from typing import Union, List, Dict
from pathlib import Path
import csv
import shutil
import sys
import uuid

# Add the src directory to the path
sys.path.append('../src')

from _geos import find_indices # type: ignore
from caete import str_or_path  # type: ignore


def get_location_data(filename: Union[Path, str]) -> List[Dict[str, Union[float, str]]]:
    """Extracts location data from a delimited text file with lon, lat, and name columns.

    Args:
        filename (Union[Path, str]): File name or path to the location file.

    Returns:
        List[Dict[str, Union[float, str]]]: List of dictionaries with the location data.
    """
    fname = str_or_path(filename, check_is_file=True)
    with open(fname, 'r') as file:
        # Use the Sniffer class to detect the dialect
        sample = file.read(1024)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        file.seek(0)
        reader = csv.DictReader(file, dialect=dialect)

        # Ensure the file has the required headers
        if not {'lon', 'lat', 'name'}.issubset(reader.fieldnames):
            raise ValueError("File must contain 'lon', 'lat', and 'name' columns")

        data = []
        for row in reader:
            data.append({
                'lon': float(row['lon']),
                'lat': float(row['lat']),
                'name': row['name']
            })

    return data

def get_indices(filename: Union[Path, str] = './gridlist_cities.csv') -> None:
    gridlist = str_or_path(filename, check_is_file=True)
    locations = get_location_data(gridlist)
    # mask = np.ones((360, 720), dtype=bool)
    idx = []
    for loc in locations:
        # print(loc)
        idx.append(find_indices(loc['lat'], loc['lon']))
        # mask[indices] = False
    return idx


def create_experiment_input2(
    filename: Union[Path, str] = './gridlist_cities.csv',
    src_folders: List[Union[Path, str]] = None,
    dest_folders: List[Union[Path, str]] = None
) -> None:
    if src_folders is None or dest_folders is None:
        raise ValueError("Source and destination folders must be provided")

    if len(src_folders) != len(dest_folders):
        raise ValueError("Source and destination folders lists must have the same length")

    # check if all the folders exist
    for src, dest in zip(src_folders, dest_folders):
        src_path = Path(src)
        dest_path = Path(dest)
        if not src_path.exists():
            raise FileNotFoundError(f"Source folder {src} does not exist")
        if not dest_path.exists():
            dest_path.mkdir(exist_ok=True, parents=True)

    idx = get_indices(filename)

    # Ensure destination directories exist
    for dest in dest_folders:
        dest_path = Path(dest)
        dest_path.mkdir(exist_ok=True, parents=True)

    # Based on index, find the required files and copy them
    for indices in idx:
        y, x = indices
        file_name = Path(f"input_data_{y}-{x}.pbz2")
        metadata_file = Path(f"ISIMIP_HISTORICAL_METADATA.pbz2")

        for src, dest in zip(src_folders, dest_folders):
            src_path = Path(src) / file_name
            src_path_metadata = Path(src) / metadata_file
            dest_path = Path(dest) / file_name
            dest_path_metadata = Path(dest) / metadata_file
            shutil.copy(src_path, dest_path)
            shutil.copy(src_path_metadata, dest_path_metadata)


if __name__ == '__main__':

# # Example usage
# Source and destination folders for an experiment using the
# gridlist_cities.csv file to get the gridcells at the centroid of the municipalities

    # give a string with the name of the new folders
    # Change these variables to match you needs

    _append = '_cities'
    _gridlist = 'gridlist_cities.csv'
    write_gridlist_with_indices = True


    # _append = '_test'
    # _gridlist = 'gridlist_test.csv'
    # write_gridlist_with_indices = True

    # Define source and destination folders
    # order is important here. The source folders must be in the same order as the destination folders

    src_folders = [
        './20CRv3-ERA5/counterclim',
        './20CRv3-ERA5/obsclim',
        './20CRv3-ERA5/spinclim',
        './20CRv3-ERA5/transclim',
        './MPI-ESM1-2-HR/piControl',
        './MPI-ESM1-2-HR/historical',
        './MPI-ESM1-2-HR/ssp370',
        './MPI-ESM1-2-HR/ssp585'
    ]


    dest_folders = [
        f'./20CRv3-ERA5/counterclim{_append}',
        f'./20CRv3-ERA5/obsclim{_append}',
        f'./20CRv3-ERA5/spinclim{_append}',
        f'./20CRv3-ERA5/transclim{_append}',
        f'./MPI-ESM1-2-HR/piControl{_append}',
        f'./MPI-ESM1-2-HR/historical{_append}',
        f'./MPI-ESM1-2-HR/ssp370{_append}',
        f'./MPI-ESM1-2-HR/ssp585{_append}'
    ]


    create_experiment_input2(
        filename=_gridlist,
        src_folders=src_folders,
        dest_folders=dest_folders
    )
    # Create a new file (gridlist) with the two extra columns with indices of the grid cells
    if write_gridlist_with_indices:
        locations = get_location_data(_gridlist)
        # Use the uuid module to create a unique identifier for each new gridlist
        # Will be appende to the file name
        id1 = uuid.uuid4().hex
        with open(f'gridlist_with_idx_{id1}.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['lon', 'lat', 'name', 'y', 'x'])
            writer.writeheader()
            for loc in locations:
                indices = find_indices(loc['lat'], loc['lon'])
                loc['y'] = indices[0]
                loc['x'] = indices[1]
                writer.writerow(loc)
