
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tables as tb


def get_var(grd, var, spin=(1, 5)):
    assert spin[0] > 0 and spin[1] > spin[0] and spin[1] <= 70

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


class test(tb.IsDescription):

    v1 = tb.Float64Col(dflt=0.0, pos=0)
    v2 = tb.Float64Col(dflt=0.0, pos=1)
    v3 = tb.Float64Col(shape=(100,), pos=0)


a = np.zeros(100,)

h5file = tb.open_file("test1.h5", mode="w", title="Test file")
group = h5file.create_group("/", 'test1', 'Test nested dtypes')

table = h5file.create_table(group, 'readout', test, "Nested datatype example")

obs = table.row

for x in range(200):
    obs['v1'] = np.random.normal()
    obs['v2'] = np.random.normal()
    obs['v3'] = np.random.normal(100,)
    obs.append()

table.flush()
