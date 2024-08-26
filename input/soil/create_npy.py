
from netCDF4 import Dataset
import numpy as np

# netCFD files with soil Phoisphorus data
p_files = ["./total_p_area_density_g-per-m2.nc",
           "./avail_p_area_density_g-per-m2.nc",
           "./inorg_p_area_density_g-per-m2.nc",
           "./org_p_area_density_g-per-m2.nc",
           "./mineral_p_area_density_g-per-m2.nc",
           "./occ_p_area_density_g-per-m2.nc",
           "./total_n_SoilGrids_g_per_m2.nc",]

variables = ["total_p", "avail_p", "inorg_p", "org_p", "mineral_p", "occ_p", "total_n"]

for p_file, varname  in zip(p_files, variables):
    with Dataset(p_file, 'r') as p_nc:
        p = p_nc.variables[varname][:]
        p[p < 0] = 0 # set negative values to 0
        p[p.mask] = -9999.0
        np.save(f"{varname}.npy", p.data)

