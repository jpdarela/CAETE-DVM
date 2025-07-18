from pathlib import Path
import sys

import numpy as np

sys.path.append("../src")

from caete import read_bz2_file

picontrol = Path("./MPI-ESM1-2-HR/piControl")

metadata = read_bz2_file(picontrol / "METADATA.pbz2")
input_files = list(picontrol.glob("input_data*"))

time_index_len = metadata[0]["time_index"].shape[0]

data_len = []
nf = len(input_files)
k = ['hurs', 'tas', 'pr', 'ps', 'rsds', 'sfcwind']
nk = len(k)

out = np.zeros((nk, nf), dtype=int)

for i, input_file in enumerate(input_files):
    data = read_bz2_file(input_file)
    for j, var in enumerate(k):
        out[j, i] = data[var].size

t = np.all(out == time_index_len)
if t:
    print(f"All input files have the same time index length: {time_index_len}")