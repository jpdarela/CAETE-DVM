import csv
import numpy as np
import argparse
import os

import sys
sys.path.append('../src')

from _geos import find_coordinates_xy

def create_gridlist_from_mask(mask, output_file, res_x=0.5, res_y=0.5):
    """
    Create a gridlist file from a given mask.

    Args:
        mask (np.ndarray): 2D numpy array where 1 indicates valid grid cells and 0 indicates invalid cells.
        output_file (str): Path to the output CSV file.
        res_x (float): Grid resolution in the x-direction (longitude).
        res_y (float): Grid resolution in the y-direction (latitude).
    """
    rows, cols = mask.shape

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['station_id','lat', 'lon', 'station_name', 'global_y', 'global_x'])
        k = 0
        for y in range(rows):
            for x in range(cols):
                if mask[y, x] == False:  # Valid grid cell mask is False
                    lat, lon = find_coordinates_xy(y, x, res_y, res_x)
                    writer.writerow([k, lat, lon, f"station_{y}-{x}", y, x])
                    k += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a gridlist file from a mask.")
    parser.add_argument("mask_file", type=str, help="Path to the mask file (numpy .npy format).")
    args = parser.parse_args()

    # Load the mask from the provided file
    mask = np.load(args.mask_file)

    # Generate the output filename based on the mask file name
    mask_filename = os.path.basename(args.mask_file)
    output_file = f"gridlist_{os.path.splitext(mask_filename)[0]}.csv"

    create_gridlist_from_mask(mask, output_file)
    print(f"Gridlist file created: {output_file}")