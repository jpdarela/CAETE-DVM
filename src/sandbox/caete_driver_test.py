# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path
from polars import read_csv
import time

from mpi4py.futures import MPIPoolExecutor

from metacommunity import pls_table
from parameters import hsoil, ssoil, tsoil
from region import region
from worker import worker

# This is a script that exemplify the usage of the new implementation of the CAETÊ model.
# Please, refer to the summary section in caete.py for more information.

# Name of the region. This name will be used to create the output folder.
region_name = "test"
obsclim_files = "../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc"
spinclim_files = "../input/20CRv3-ERA5/spinclim/caete_input_20CRv3-ERA5_spinclim.nc"
gridlist = read_csv("../grd/gridlist_test.csv")
co2_path = Path("../input/co2/historical_CO2_annual_1765-2024.csv")



if __name__ == "__main__":
    
    
    soil_tuple = tsoil, ssoil, hsoil

    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-9999.csv"))

    print("creating region with spinclim files")
    r = region(region_name,
            spinclim_files,
            soil_tuple,
            co2_path,
            main_table,
            gridlist=gridlist)

    print(f"Region {region_name} created with {r.region_size} gridcells")
    r.set_gridcells()
        
    fn = worker()
    time_start = time.perf_counter()
    print("Starting the model run")
    with MPIPoolExecutor() as executor:
        executor.map(fn.test_run, r.gridcells, chunksize=16, buffersize=16)

    print("Model run finished")
    time_end = time.perf_counter()
    print(f"Total time: {time_end - time_start:.2f} seconds")