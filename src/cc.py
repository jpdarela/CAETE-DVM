# cc.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from cc import carbon_costs as cc

print(cc.active_cost.__doc__)


# Simulation of cc to nutrients uptake for CAETÊ-DVM
# TRAITS
pls_table = pd.read_csv('./pls_attrs.csv')


# fraction of Symbionts that are AM
amp = pls_table.iloc(1)[15]

# fraction of NPP dedicated to Diazotrophic organisms
pdia = pls_table.iloc(1)[16]

pls = 26


nmin = 0.01  # gm-2
plab = 0.34

croot = 0.35 * 1000  # kg m-2 to g m-2

nupt = 0.04   # all g m-2 day-1
pupt = 0.03
npp = 1.25

tsoil = 22.5
wsoil = 145.0  # mm
e = 5e-4 * 86400.0   # mm/s to mm/day

# calc N fixed
# Calculate before all to subtract npp
fixed_n = cc.fixed_n(npp * pdia[pls], tsoil)

npp -= npp * pdia[pls]


# #Passive uptake
#    subroutine passive_uptake (w, av_n, av_p, nupt, pupt,&
#                 & e, topay_upt, to_storage)

to_pay, to_sto = cc.passive_uptake(wsoil, nmin + fixed_n, plab, nupt, pupt, e)


# Calc costs of active uptake
args = (amp[pls], nmin, plab, croot)
cc_active = cc.active_cost(*args)

ccost, strategy = cc.select_active_strategy(*args)

total_c_cost = to_pay * ccost

if to_pay[1] > 0.0:
    # Calculate the N costs of P uptake
    # subroutine ap_actvity1(c_xm, nut, strat, cc_array, ezc_ap)
    # C that is used in enzymes
    ezc_ap = cc.ap_actvity1(total_c_cost[1], strategy[1], cc_active)

    # Calculate the enzyme concentration
    # ezc_prod(c_ezc, nut, strat, cc_array, enzyme_conc)
    enzyme_conc = cc.ezc_prod(ezc_ap, strategy[1], cc_active)

    # Nutrient gains with enzymes
    nut_out = cc.active_nutri_gain(enzyme_conc)

    # N investido em P
    n_invest_p = cc.n_invest_p(ezc_ap)


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
