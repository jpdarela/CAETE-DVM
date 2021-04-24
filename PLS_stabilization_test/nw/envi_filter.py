import os
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt


def get_var(grd, var, spin=(1, 5)):
    assert spin[0] > 0 and spin[1] > spin[0]

    files = os.listdir(grd)
    # files = sorted(files)

    ROOT = os.getcwd()
    os.chdir(grd)

    k = sorted(files)[spin[0] - 1:spin[1]]
    # find var dim
    with open(k[-1], 'rb') as fh:
        dt = joblib.load(fh)
        dt = dt[var]
        dim = dt.shape
    if len(dim) == 1:
        output = np.zeros(0, dtype=dt.dtype)
        for fname in k:
            with open(fname, mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    elif len(dim) == 2:
        day_len = dim[-1] * int(len(k))
        s = (dim[0], 0)
        output = np.zeros(s, dtype=dt.dtype)
        for fname in k:
            with open(fname, mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    elif len(dim) == 3:
        s = (dim[0], dim[1], 0)
        output = np.zeros(s, dtype=dt.dtype)
        for fname in k:
            with open(fname, mode='rb') as fh:
                dt1 = joblib.load(fh)
                dt1 = dt1[var]
                output = np.hstack((output, dt1))
    else:
        output = 0
    os.chdir(ROOT)
    return output


dirs = glob.glob1("./", "gridcell*")

div = len(dirs) - 2

template = get_var(dirs[0], 'area', (1, 37))

template = 0.0

for d in dirs:
    print("Extracting from ", d, end='\r')
    template += get_var(d, 'area', (1, 37)) / div
print("")
plt.plot(template.T)
plt.title("Filtragem das estratégias de vida durante o spinup")
plt.ylabel("Abundância Normalizada")
plt.xticks(np.array([]))
plt.xlabel("389 anos")
plt.savefig("SPINUP_ABUND_SPINUP.png", dpi=600)
plt.show()

# __
template = get_var(dirs[0], 'csoil', (1, 37))

template[...] = 0.0


for d in dirs:
    template = get_var(d, 'csoil', (1, 37)).sum(axis=0,)
    plt.plot(template, 'tab:brown', alpha=0.3)
plt.title("Estabilização do Carbono orgânico no solo durante o spinup")
plt.ylabel("Carbono organico no solo g(C)m⁻²")
plt.xticks(np.array([]))
plt.xlabel("389 anos")
plt.savefig("SPINUP_CSOIL_SPINUP.png", dpi=600)
plt.show()

template_cl = get_var(dirs[0], 'cleaf', (1, 37))
template_cf = get_var(dirs[0], 'cfroot', (1, 37))
template_cw = get_var(dirs[0], 'cawood', (1, 37))
template_st = get_var(dirs[0], 'storage_pool', (1, 37))[0, :]

template_cl[...] = 0.0
template_cf[...] = 0.0
template_cw[...] = 0.0
template_st[...] = 0.0

out = template_cl + template_cf + template_cw + template_st

for d in dirs:
    print("CVEG -", d, end='\r')
    template_cl = get_var(d, 'cleaf', (1, 37))
    template_cf = get_var(d, 'cfroot', (1, 37))
    template_cw = get_var(d, 'cawood', (1, 37))
    template_st = get_var(d, 'storage_pool', (1, 37))[0, :] / 1e3
    out = template_cl + template_cf + template_cw + template_st
    if out[-1] > 0.0:
        plt.plot(out, 'tab:green', alpha=0.3)

plt.title("Estabilização do Carbono na vegetação durante o spinup")
plt.ylabel("Biomassa seca  g(C)m⁻²")
plt.xticks(np.array([]))
plt.xlabel("389 anos")

plt.savefig("SPINUP_CVEG_SPINUP.png", dpi=600)
print("")
plt.show()
