from caete_import import *
from math import exp

f1 = model.photo.vcmax_a
fsla = model.photo.spec_leaf_area
n = 1000
table = table_gen(n)

leaf_n = table[9,:] * 1e3  # Convert to mg/g
leaf_p = table[12,:] * 1e3 # Convert to mg/g
tau_leaf = table[2,:] # tau_leaf in years

"""
vcmaxd = vcmax_a(npa,ppa,sla)

Wrapper for ``vcmax_a``.

Parameters
----------
npa : input float - mg / g
ppa : input float - mg / g
sla : input float - m2 / g

Returns
-------
vcmaxd : float
"""

def vm_temp(temp, vm):
    return  (vm*2.0e0**(0.1e0*(temp-25.0e0)))/(1.0e0+exp(0.3e0*(temp-36.0e0)))


def test_vcmax(tau_leaf, i):
    """
    Test the vcmax_a function from the model.photo module.
    """
    sla = fsla(tau_leaf)
    npa = leaf_n[i]
    ppa = leaf_p[i]
    vcmax = f1(npa, ppa, sla)
    print(f"tau_leaf: {tau_leaf}, npa: {npa}, ppa: {ppa}, vcmax: {vcmax * 1e6}, vm: {(vm_temp(25, vcmax) * 1e6):.15f} mol m-2 s-1")

    return f"{tau_leaf},{npa},{ppa},{vcmax * 1e6},{(vm_temp(25, vcmax) * 1e6):.15f}"


header = "tau_leaf,npa,ppa,vcmax,vm"

with open("vcmax_test.csv", "w") as f:
    f.write(header + "\n")


with open("vcmax_test.csv", "a") as f:
    for i in range(n):
        line = test_vcmax(tau_leaf[i], i)
        f.write(line + "\n")

