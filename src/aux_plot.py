import _pickle as pkl
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    else:
        output = 0
    return output


def get_data(grd, spin=(1, 5)):
    assert spin[0] > 0 and spin[1] > spin[0] and spin[1] <= 70

    vars = ['emaxm', 'tsoil', 'photo', 'aresp',
            'npp', 'sind', 'eind']  # ['lai','inorg_n','inorg_p',
    #  'sorbed_n','sorbed_p','hresp','rcm','f5','runom','evapm','wsoil',
    #  'swsoil','rm','rg','cleaf','cawood','cfroot','wue','cue','cdef',
    #  'nmin','pmin','vcmax','specific_la','litter_l','cwd',
    #  'litter_fr','ls','c_cost']

    k = sorted(list(grd.outputs.keys()))[spin[0] - 1:spin[1] + 1]
    # find var dim
    out = []
    print(k)
    with open(grd.outputs[k[-1]], 'rb') as fh:
        dt1 = joblib.load(fh)
        for key, var in dt1.items():
            if key in vars:
                out.append(var)
    return dict(zip(vars, out))


def create_panel(grd):
    # hdf = pd.HDFStore('db.h5')
    a = grd.outputs
    return a
    # b = np.array([])
    # names = []
    # arrays = []

    # c = grd.outputs
    # dt_shape = c['spin1'][var].shape
    # test = len(dt_shape) > 1

    # if not test:
    #     for i, spin in enumerate(a):
    #         data = c[spin][var]
    #         b = np.hstack((b, data))
    #     return b,
    # else:
    #     arrays = [np.array([]) for x in np.arange(dt_shape[0])]
    #     for i, spin in enumerate(a):
    #         data = c[spin][var]
    #         for z in range(data.shape[0]):
    #             dt2 = data[z, :]
    #             arrays[z] = np.hstack((arrays[z], dt2))
    #     for z in range(dt_shape[0]):
    #         names.append("{}{}".format(var, z + 1))
    #     return arrays, names


# def plot_var(grd, var):
#     to_plot = int_tspan(grd, var)

#     if len(to_plot) == 1:
#         plt.plot(to_plot[0])
#         plt.legend([var, ])
#         plt.show()
#     else:
#         data, names = to_plot
#         for array in data:
#             plt.plot(array)
#         plt.legend(names)
#         plt.show()
