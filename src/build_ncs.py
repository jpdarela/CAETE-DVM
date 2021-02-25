# experiments.py
import os
import joblib
from post_processing import write_h5
from h52nc import h52nc


# # antes de rodar o experimento:

# # 1 limpar a pasta outputs deixando apenas o arquivo dos PLSs
# # 2 mover a pasta nc_outputs para outro local pois o programa pode sobrescrever os arquivos

# with open("RUN0.pkz", 'rb') as fh:
#     gridcells = joblib.load(fh)


# def fun1(grid):
#     os.mkdir(grid.out_dir)
#     grid.run_caete("19790101", "19991231")


# def fun2(grid):
#     grid.run_caete("20000101", "20150101", fix_co2=600.0)


# for gridcell in gridcells:
#     fun1(gridcell)

# for gridcell in gridcells:
#     fun2(gridcell)

# print("Saving db")
# write_h5()
print("\n\nSaving netCDF4 files")
h52nc("../outputs/CAETE.h5")
