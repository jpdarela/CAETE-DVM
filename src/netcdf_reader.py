#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel NetCDF Reader using MPI4py
-----------------------------------
A standalone script to read NetCDF variables in parallel using MPI.
Called as a subprocess from the netcdf_handler class.

Usage:
    mpiexec -n <num_processes> python netcdf_reader.py <nc_file> <indices> <var1> [var2] [var3] ...

Arguments:
    nc_file: Path to NetCDF file
    indices: Comma-separated list of station indices to read
    var1, var2, ...: Variable names to read

Output:
    Pickled dictionary written to stdout containing variable data
"""

import sys
import pickle
import numpy as np
import netCDF4 as nc
from mpi4py import MPI
import argparse

# Check if the netCDF4 module has parallel support for NetCDF4 via HDF5 parallel I/O.
# This is necessary for parallel reading of NetCDF files. PnetCDF not yet supported.
if nc.__has_parallel4_support__ :
    pass
else:
    print("ERROR: NetCDF4 parallel support is not enabled. Please install the parallel version of netCDF4.", file=sys.stderr)
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parallel NetCDF variable reader. Output is a pickled dictionary of variable data to stdout.')
    parser.add_argument('nc_file', help='Path to NetCDF4-HDF5 file')
    parser.add_argument('indices', help='Comma-separated list of station indices')
    parser.add_argument('variables', nargs='+', help='Variable names to read')
    parser.add_argument('--var-type', choices=['time', 'station'], default='time',
                       help='Type of variables (time-varying or station-specific)')

    return parser.parse_args()


def read_variable_chunk(nc_file, var_name, indices, var_type):
    """
    Read a variable using MPI process rank assignment.

    Args:
        nc_file (str): Path to NetCDF file
        var_name (str): Variable name to read
        indices (np.array): Station indices to extract
        var_type (str): 'time' or 'station'
        rank (int): MPI process rank
        size (int): Total number of MPI processes

    Returns:
        tuple: (var_name, var_data) or (var_name, None) if not assigned
    """
    try:
        # Open NetCDF file with this process
        with nc.Dataset(nc_file, 'r') as dataset:
            if var_name not in dataset.variables:
                return var_name, None

            # Read variable data based on type
            if var_type == 'time':
                # Time-varying variables: shape (time, stations)
                var_data = dataset.variables[var_name][:, indices].astype(np.float32)
            else:
                # Station variables: shape (stations,)
                var_data = dataset.variables[var_name][indices].astype(np.float32)

            # Handle masked arrays
            if isinstance(var_data, np.ma.MaskedArray):
                fill_value = var_data.mean() if var_data.size > 0 else 0.0
                var_data = var_data.filled(fill_value=fill_value)

            return var_name, var_data

    except Exception as e:
        return var_name, f"ERROR: {str(e)}"


def main():
    """Main function to coordinate MPI reading."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse arguments (only rank 0 needs to do this initially)
    if rank == 0:
        args = parse_arguments()

        # Parse indices
        try:
            indices = np.array([int(x.strip()) for x in args.indices.split(',')])
        except ValueError:
            print("ERROR: Invalid indices format", file=sys.stderr)
            comm.Abort(1)

        # Prepare data to broadcast
        broadcast_data = {
            'nc_file': args.nc_file,
            'indices': indices,
            'variables': args.variables,
            'var_type': args.var_type
        }
    else:
        broadcast_data = None

    # Broadcast data to all processes
    broadcast_data = comm.bcast(broadcast_data, root=0)

    nc_file = broadcast_data['nc_file']
    indices = broadcast_data['indices']
    variables = broadcast_data['variables']
    var_type = broadcast_data['var_type']

    # Distribute variables among processes
    vars_per_process = len(variables) // size
    extra_vars = len(variables) % size

    # Calculate which variables this process should handle
    if rank < extra_vars:
        start_var = rank * (vars_per_process + 1)
        num_vars = vars_per_process + 1
    else:
        start_var = rank * vars_per_process + extra_vars
        num_vars = vars_per_process

    # Get assigned variables for this process
    my_variables = variables[start_var:start_var + num_vars]

    # Read assigned variables
    my_results = {}
    for var_name in my_variables:
        var_name_result, var_data = read_variable_chunk(
            nc_file, var_name, indices, var_type,
        )

        if var_data is not None and not isinstance(var_data, str):
            my_results[var_name_result] = var_data
        elif isinstance(var_data, str):
            # Error occurred
            if rank == 0:
                print(f"Error reading {var_name}: {var_data}", file=sys.stderr)

    # Gather results from all processes
    all_results = comm.gather(my_results, root=0)

    # Combine results and output (only rank 0)
    if rank == 0:
        final_results = {}
        for result_dict in all_results:
            final_results.update(result_dict)

        # Validate that we got all requested variables
        missing_vars = set(variables) - set(final_results.keys())
        if missing_vars:
            print(f"WARNING: Missing variables: {missing_vars}", file=sys.stderr)

        # Output results as pickled data to stdout
        try:
            # Use binary mode for pickle output
            pickle.dump(final_results, sys.stdout.buffer)
        except Exception as e:
            print(f"ERROR: Failed to output results: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()