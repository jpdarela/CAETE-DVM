# cc.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cc import carbon_costs as cc

print(cc.active_cost.__doc__)



# Simulation of cc to nutrients uptake for CAETÊ-DVM
# TRAITS
pls_table = pd.read_csv('./pls_attrs.csv')


# fraction of NPP dedicated to Mycorrhizi symbionts
pmyco = pls_table.iloc(1)[15]

# fraction of Symbionts that are AM
amp = pls_table.iloc(1)[16]

# fraction of NPP dedicated to Diazotrophic organisms
pdia = pls_table.iloc(1)[17]

pls = 20


nmin = 0.1  # gm-2
plab = 0.01
orgp = 5.0

croot = 0.85  # g m-2

nupt = 0.4   # all g m-2 day-1
pupt = 0.003
npp = 1.25

tsoil = 22.5
wsoil = 345.0  # mm
e = 5e-4 * 86400.0   # mm/s to mm/day

# calc N fixed
fixed_n = cc.fixed_n(npp * pdia[pls], tsoil)

# #Passive uptake
#    subroutine passive_uptake (w, av_n, av_p, nupt, pupt,&
#                 & e, topay_upt, to_storage)

to_pay, to_sto = cc.passive_uptake(wsoil, nmin + fixed_n, plab, nupt, pupt, e)


# Calc costs of active uptake

cc_active = cc.active_cost(amp[pls], nmin, plab, croot)


def print1():
    out = np.zeros(shape=(2, 4, 1000))
    phosp = np.linspace(0.0003, 3.0, 1000)
    for i, p in enumerate(phosp):
        print(i, p)
        out[:, :, i] = cc.active_cost(0.1, p, 0.01, 0.95)

    legend = ['CaquN NAM', 'CaquN NEM', 'CaquN_AM', 'CaquN_EM']
    legend += ['CaquP NAM', 'CaquP NEM', 'CaquP_AM', 'CaquP_EM']
    colors = ['r', 'm', 'b', 'g', 'r', 'c', 'y', 'orange']

    count = 0
    for mode in range(4):
        for nut in range(1, 2):
            # print(out[nut, mode, :])
            plt.plot(phosp, out[nut, mode, :], colors[count])
            count += 1
    plt.xlabel('Nutrient gm⁻²')
    plt.ylabel('g(C) g(Nutrient)⁻¹')
    plt.legend(legend[:4])
    plt.show()

# count = 0
# for mode in range(4):
#     for nut in range(1, 2):
#         plt.plot(phosp, out[nut, mode, :], colors[count])
#         count += 1

# plt.legend(legend[4:])
# plt.show()
