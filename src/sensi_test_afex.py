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
import shutil
import multiprocessing as mp
from pathlib import Path
import joblib
from post_processing import write_h5
import h52nc

while True:
    afex_treat = input("N, P ou NP: ")
    afex_treat = afex_treat.upper()
    if afex_treat in ['N', 'P', 'NP']:
        break

with open("afex.cfg", mode='w') as fh:
    fh.writelines([f"{afex_treat}\n", ])

while True:
    run = input("Select the base run (h for help): ")
    av_run = os.listdir("/home/amazonfaceme/jpdarela/CAETE/CAETE-DVM/outputs/")

    if run in av_run:
        break

    if run.lower() == 'h':
        print("\nAvailable base runs are: ")
        for r in av_run:
            print(f"\tRun name: {r}")
        print("\n")

run_path = Path(
    f"/home/amazonfaceme/jpdarela/CAETE/CAETE-DVM/outputs/{run}/RUN_{run}_.pkz")
pls_path = Path(
    f"/home/amazonfaceme/jpdarela/CAETE/CAETE-DVM/outputs/{run}/pls_attrs.csv")

# Experiment - AFEX X

# new outputs folder
dump_folder = Path(f"{run}_afex_{afex_treat}")

with open(run_path, 'rb') as fh:
    init_conditions = joblib.load(fh)

for gridcell in init_conditions:
    gridcell.clean_run(dump_folder, "init_cond")

h52nc.set_historical_stime(new_descr=False)
h52nc.EXPERIMENT = f"AFEX_{afex_treat}"
from caete import run_breaks_hist as rb
# h52nc.custom_rbrk(rb)

print(f"Calendar is {h52nc.CALENDAR}")
print(f"Time_UNITS  are {h52nc.TIME_UNITS}")
print(f"EXPERIMENT is {h52nc.EXPERIMENT}")


def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res


def apply_funX(grid, brk):
    grid.run_caete(brk[0], brk[1], afex=True)
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
