# co2_experiment.py
# sensi_test.py
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import joblib
from post_processing import write_h5
from h52nc import h52nc

run_path = Path("barbara@sombrero:/d/c1/homes/amazonfaceme/barbara/caete/CAETE-DVM/outputs/run1_1k/RUN_run1_1k_.pkz")
pls_path = Path("barbara@sombrero:/d/c1/homes/amazonfaceme/barbara/caete/CAETE-DVM/outputs/run1_1k/pls_attrs.csv")

# Experiment - No eCO2 - HISTORICAL

# new outputs folder
dump_folder = Path("CO2_EXPERIMENT")

with open(run_path, 'rb') as fh:
    init_conditions = joblib.load(fh)

for gridcell in init_conditions:
    gridcell.clean_run(dump_folder, "init_cond")


def fun(gridcell):
    gridcell.run_caete("19790101", "20161231", fix_co2=200)


n_proc = mp.cpu_count() // 2

with mp.Pool(processes=n_proc) as p:
    result = p.map(fun, init_conditions)

to_write = Path(os.path.join(Path("../outputs"), dump_folder)).resolve()
attrs = Path(os.path.join(to_write, Path("pls_attrs.csv"))).resolve()
h5path = Path(os.path.join(to_write, Path('CAETE.h5'))).resolve()
nc_outputs = Path(os.path.join(to_write, Path('nc_outputs')))

shutil.copy(pls_path, attrs)
write_h5(to_write)
h52nc(h5path, nc_outputs)


# LOAD G0 run
# UPDATE DUMP FOLDER
# run the experiment
# process outputs
