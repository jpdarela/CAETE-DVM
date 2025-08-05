"""This script read as netCDF file and extracts the boolean mask from the variable mask.
    There is an example netCDF4 file in the mask folder called pan_amazon_05d.nc which exemplifies
    how the netCDF file should be structured.
"""
import argparse
from netCDF4 import Dataset # type: ignore
import numpy as np

parser = argparse.ArgumentParser(
    description='Extracts the boolean mask from the variable mask in a netCDF file.',
    epilog='Example usage: python make_mask.py -m pan_amazon_05d.nc -o mask_array.npy'
)
# add argument
parser.add_argument('-m', '--mask_netcdf', type=str, help='Must be a valid filename\
                                                           or path for a existing well formated netCDF4 file', required=True)
# parse the argument -o --output_mask of type string
parser.add_argument('-o', '--output_mask', type=str, required=True, help='Output mask file, must be a valid filename or path')
#parse the argument -v --variable of type string with default value of "mask"
parser.add_argument('-v', '--variable', type=str, default='mask', help='Variable name in the netCDF file, defaults to "mask"')
parser.add_argument('-f', '--flip', action='store_true', help='If set, the mask will be flipped in the 0 axis (up/down). This is useful for some netCDF files that have the y-axis inverted.')

# parse the arguments
args = parser.parse_args()

with Dataset(args.mask_netcdf, 'r') as mask_nc:
    mask = mask_nc.variables[args.variable][:].mask
if args.flip:
    mask = np.flipud(mask)  # Flip the mask vertically if the flip argument is set
np.save(args.output_mask, mask)
