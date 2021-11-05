
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.)

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# AUTHOR: JP Darela

# sensi_test.py
import os
import _pickle as pkl
import bz2
import shutil
import copy
import multiprocessing as mp
from pathlib import Path
import joblib
from post_processing import write_h5
import h52nc


model = "GFDL-ESM2M"
rcp = "rcp26"

base_run = f"/home/amazonfaceme/jpdarela/CAETE/CAETE-DVM/outputs/{model}_historical"


run_path = Path(os.path.join(base_run, Path(f"RUN_{model}_historical_.pkz")))

pls_path = Path(os.path.join(base_run, Path("pls_attrs.csv")))


# RCP X.X

# new outputs folder
dump_folder = Path(f"{model}_{rcp}")

s_data = Path(f"/home/amazonfaceme/shared_data/{model}").resolve()
input_fpath = Path(os.path.join(s_data, Path(rcp)))

metadata = Path(os.path.join(
    input_fpath, Path(f"{model}-{rcp}_METADATA.pbz2")))
co2p = Path(os.path.join(s_data, Path(f"co2-{model}-{rcp}.txt")))

# READ METADATA
with bz2.BZ2File(metadata, mode='r') as fh:
    clim_metadata = pkl.load(fh)

# READ CO2
with open(co2p) as fh:
    co2_data = fh.readlines()

stime = copy.deepcopy(clim_metadata[0])
del clim_metadata

with open(run_path, 'rb') as fh:
    init_conditions = joblib.load(fh)

for gridcell in init_conditions:
    gridcell.clean_run(dump_folder, "init_cond")
    gridcell.change_clim_input(input_fpath, stime, co2_data)

print(f"Calendar is {h52nc.CALENDAR}")
print(f"Time_UNITS  are {h52nc.TIME_UNITS}")
print(f"EXPERIMENT is {h52nc.EXPERIMENT}")

h52nc.CALENDAR = stime['calendar']
h52nc.TIME_UNITS = stime['units']
h52nc.EXPERIMENT = f"{model}-{rcp}"
from caete import run_breaks_CMIP5_proj as rb
h52nc.custom_rbrk(rb)

print(f"Calendar is {h52nc.CALENDAR}")
print(f"Time_UNITS  are {h52nc.TIME_UNITS}")
print(f"EXPERIMENT is {h52nc.EXPERIMENT}")


def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res


def apply_funX(grid, brk):
    grid.run_caete(brk[0], brk[1])
    return grid


n_proc = mp.cpu_count() // 2

for i, brk in enumerate(rb):
    print(f"Applying model to the interval {brk[0]}-{brk[1]}")
    init_conditions = zip_gridtime(init_conditions, (brk,))
    with mp.Pool(processes=n_proc) as p:
        init_conditions = p.starmap(apply_funX, init_conditions)

to_write = Path(os.path.join(Path("../outputs"), dump_folder)).resolve()
attrs = Path(os.path.join(to_write, Path("pls_attrs.csv"))).resolve()
h5path = Path(os.path.join(to_write, Path('CAETE.h5'))).resolve()
nc_outputs = Path(os.path.join(to_write, Path('nc_outputs')))

shutil.copy(pls_path, attrs)
write_h5(to_write)
h52nc.h52nc(h5path, nc_outputs)
