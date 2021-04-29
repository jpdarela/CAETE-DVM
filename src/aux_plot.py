
import joblib
import numpy as np
import matplotlib.pyplot as plt


def get_var(grd, var, spin=(1, 5)):
    assert spin[0] > 0 and spin[1] > spin[0]

    k = sorted(list(grd.outputs.keys()))[spin[0] - 1:spin[1]]
    # find var dim
    with open(grd.outputs[k[-1]], 'rb') as fh:
        dt = joblib.load(fh)
        dt = dt[var]
        dim = dt.shape
    if len(dim) == 1:
        output = np.zeros(0, dtype=dt.dtype)
        for fname in k:
            with open(grd.outputs[fname], mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    elif len(dim) == 2:
        day_len = dim[-1] * int(len(k))
        s = (dim[0], 0)
        output = np.zeros(s, dtype=dt.dtype)
        for fname in k:
            with open(grd.outputs[fname], mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    elif len(dim) == 3:
        s = (dim[0], dim[1], 0)
        output = np.zeros(s, dtype=dt.dtype)
        for fname in k:
            with open(grd.outputs[fname], mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    else:
        output = 0
    return output
