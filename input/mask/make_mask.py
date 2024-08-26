"""This script read as netCDF file and extracts the boolean mask from the variable mask.
    There is an example netCDF4 file in the mask folder called pan_amazon_05d.nc that exemplifies
    how the netCDF file should be structured.
"""
import argparse
from netCDF4 import Dataset
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

# parse the arguments
args = parser.parse_args()

with Dataset(args.mask_netcdf, 'r') as mask_nc:
    mask = mask_nc.variables[args.variable][:].mask

np.save(args.output_mask, mask)



