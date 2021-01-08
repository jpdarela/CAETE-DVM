# cc.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.arrays import sparse
import plsgen

from cc import carbon_costs as cc

# print(cc.active_cost.__doc__)


# # Simulation of cc to nutrients uptake for CAETÊ-DVM
# TRAITS
pls_table = pd.read_csv('./pls_attrs.csv')
# pls_table = plsgen.table_gen(1000)


# fraction of Symbionts that are AM
amp = pls_table.iloc(1)[15]
# amp = pls_table[15, :]
# fraction of NPP dedicated to Diazotrophic organisms
pdia = pls_table.iloc(1)[16]
# pdia = pls_table[16, :]
rs = pls_table.iloc(1)[1]

croot = 950.0  # g m-2

nupt = 0.03   # all g m-2
pupt = 0.0039
npp = 2.25


tsoil = 25.5
wsoil = 245.0  # mm
e = 3e-4 * 86400.0   # mm/s to mm/day


pls = np.random.randint(0, 999)
# calc N fixed
# Calculate before all to subtract npp

fixed_n = cc.fixed_n(npp * pdia[pls], tsoil)
# In CAETÊ the fixed N or a fraction of it are available to allocate C
# I is counted as available N and non used mass remains in storage
# Nutrients in soil. In this way the N STORAGE will downregulate N uptake
nmin = 1.2 * 0.0045  # gm-2
on = 1.0 * 0.0045
plab = 1.0 * 0.003
sop = 1.0 * 0.003
op = 3.0 * 0.003

# discount c_fix before  allocation
npp -= npp * pdia[pls]

littern = 0.08  # g m-2
rsn = rs[pls] * littern
litterp = 0.03
rsp = rs[pls] * litterp

# #Passive uptake
# Estimate passive uptake and subtract from realized uptake

to_pay, to_sto, pu = cc.passive_uptake(
    wsoil, nmin, plab, nupt, pupt, e)

a = 0
if pu[0] > nmin:
    pu[0] = nmin
    nmin = 0.0
    print("PASSIVE n CLEANED THE POOL")
    a += 1
if pu[1] > plab:
    pu[1] = plab
    plab = 0.0
    print("PASSIVE p CLEANED THE POOL")
    a += 1

if a:
    assert False

# Calc costs of active uptake
Pargs = (amp[pls], plab - pu[1], sop, op, croot)
Nargs = (amp[pls], nmin - pu[0], on, croot)
cp = 0.0
pstrat = 0
cn = 0.0
nstrat = 0
nout = np.zeros(2,)
pout = np.zeros(3,)

# calculate the amount of nutrient actively  extracted from the specific pool if necessary
if to_pay[0] > 0.0:
    ccn = cc.active_costn(*Nargs)
    cn, nstrat = cc.select_active_strategy(ccn)
    nout = cc.prep_out_n(nstrat, nupt, to_pay[0])
else:
    nout[0] = nupt

# calculate the amount of nutrient actively  extracted from the specific pool if necessary
if to_pay[1] > 0.0:
    ccp = cc.active_costp(*Pargs)
    cp, pstrat = cc.select_active_strategy(ccp)
    pout = cc.prep_out_p(pstrat, pupt, to_pay[1])
else:
    # everthing is passive
    pout[0] = pupt
# Retranslocation costs
cc_r_n = cc.retran_nutri_cost(littern, rsn, 1)
cc_r_p = cc.retran_nutri_cost(litterp, rsp, 2)


# prints
print("\nAMP: ", amp[pls])
print("Fixed N: ", fixed_n, ' npp%: ', pdia[pls])
print("PASSIVE N uptk: ", pu[0])
print("PASSIVE P uptk: ", pu[1])
print("REALIZED ACTIVE N uptk: ", nout)
print("REALIZED ACTIVE P uptk: ", pout)
print("ALLOC N uptake: ", nupt)
print("ALLOC P uptake: ", pupt)
print('CC: ', cn + cp + cc_r_p + cc_r_n)
print('N / P strat: ', nstrat, pstrat)

# ccost, strategy = cc.select_active_strategy(*args)

# total_c_cost = to_pay * ccost

# ezc_ap = 0
# enzyme_conc = 0
# nut_out = 0
# n_invest_p = 0

# if to_pay[1] > 0.0:
#     # Calculate the N costs of P uptake
#     # subroutine ap_actvity1(c_xm, nut, strat, cc_array, ezc_ap)
#     # C that is used in enzymes
#     ezc_ap = cc.ap_actvity1(total_c_cost[1], strategy[1], cc_active)

#     # Calculate the enzyme concentration
#     # ezc_prod(c_ezc, nut, strat, cc_array, enzyme_conc)
#     enzyme_conc = cc.ezc_prod(ezc_ap, strategy[1], cc_active)

#     # Nutrient gains with enzymes
#     nut_out = cc.active_nutri_gain(enzyme_conc)

#     # N investido em P
#     n_invest_p = cc.n_invest_p(ezc_ap)

# Estimate resorption costs
# cc_r_n = cc.retran_nutri_cost(littern, rsn, 1)
# cc_r_p = cc.retran_nutri_cost(litterp, rsp, 2)

# print("\n\nFixed N: ", fixed_n, ' npp%: ', pdia[pls])
# print("To Pay N | P: ", to_pay)
# print("To sto N | P: ", to_sto)

# print("\nActive COSTS: ", cc_active)

# print("ccost N | P: ", ccost, pls, " Strategy: ", strategy)
# print("Total costs N & P: ", total_c_cost.sum() + cc_r_n + cc_r_p)
# print("AMP: ", amp[pls])
# print("C2E", ezc_ap)
# print("Ec", enzyme_conc)
# print("Pout", nut_out)
# print("Nout", n_invest_p)


def print1(amp):
    out = np.zeros(shape=(2, 4, 1000))
    phosp = np.linspace(0.005, 120, 1000)
    for i, p in enumerate(phosp):
        # print(i, p)
        out[:, :, i] = cc.active_cost(amp, p, 3.4, 950.0)

    legend = ['CaquN NAM', 'CaquN NEM', 'CaquN_AM', 'CaquN_EM']
    legend += ['CaquP NAM', 'CaquP NEM', 'CaquP_AM', 'CaquP_EM']
    colors = ['r', 'm', 'b', 'g', 'r', 'c', 'y', 'orange']

    count = 0
    for mode in range(4):
        for nut in range(1):
            # print(out[nut, mode, :])
            plt.plot(phosp, out[nut, mode, :], colors[count])
            count += 1
    plt.xlabel('Nutrient gm⁻²')
    plt.ylabel('g(C) g(Nutrient)⁻¹')
    plt.legend(legend[:4])
    plt.show()


def print2(amp):
    out = np.zeros(shape=(2, 4, 1000))
    phosp = np.linspace(0.0003, 3.0, 1000)
    for i, p in enumerate(phosp):
        #print(i, p)
        out[:, :, i] = cc.active_cost(amp, 15, p, 950.0)

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
    plt.legend(legend[4:])
    plt.show()

# count = 0
# for mode in range(4):
#     for nut in range(1, 2):
#         plt.plot(phosp, out[nut, mode, :], colors[count])
#         count += 1

# plt.legend(legend[4:])
# plt.show()
