"""
Copyright 2017-2018 LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

# This module contains the procedures to create the set of PLant life strategies for CAETÊ runs

import os
import sys
from random import shuffle, sample
from math import ceil
import csv
from pathlib import Path

import numpy as np
from caete_module import photo as model
from caete_module import global_par as gp

__author__ = 'JP Darela'

# Cache data of allocation coefficients
woody_allocations_file = "wallo.npy"
grassy_allocations_file = "gallo.npy"


def vec_ranging(values, new_min, new_max):
    """ Ranges the vector (1D) values(np.array) to min max
        - Normalize values - preserves the distance
    """

    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max -
                                       old_min) * (v - old_min) + new_min
        output.append(new_v)

    return np.array(output, dtype=np.float32)


def check_viability(trait_values, wood):
    """ Check the viability of allocation(a) & residence time(ŧ) combinations.
        Some PLS combinations of allocation coefficients and residence times
        are not 'biomass acumulators' at low npp (< 0.01 kg m⁻² year⁻¹)
        do not have enough mass of carbon (< 0.01 kg m⁻²) in all CVEG compartments

        trait_values: np.array(shape=(6,), dtype=f64) allocation and residence time combination (possible PLS)
        wood: bool  Is this a woody PLS?
    """

    assert wood is not None
    rtur = np.array(model.spinup3(gp.cmin, trait_values))
    if wood:
        if rtur[0] <= gp.cmin or rtur[1] <= gp.cmin or rtur[2] <= gp.cmin:
            return False
        return True
    else:
        if rtur[0] <= gp.cmin or rtur[1] <= gp.cmin:
            return False
        return True


def assertion_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = 0.07
    diffg = ceil(dsize * g2w_ratio)
    diffw = int(dsize - diffg)
    assert diffg + diffw == dsize
    return diffg, diffw


def turnover_combinations(verbose=False):
    """CREATE the residence time and allocation combinations"""

    # constrained distributions (must sum up to 1.)
    file1 = False
    file2 = False

    if os.path.exists(grassy_allocations_file):
        plsa_grass = np.load(grassy_allocations_file)
        file1 = True

    else:
        print("Building grassy allocation combinations: \n")
        aleafg = np.arange(20., 80.1, 0.0125e1, dtype=np.float64)
        arootg = np.arange(20., 80.1, 0.0125e1, dtype=np.float64)

        plsa_grass = [[a, 0.0, c]
                      for a in aleafg for c in arootg if (a + c) == 100]
        np.save(grassy_allocations_file, np.array(plsa_grass))

    if os.path.exists(woody_allocations_file):
        plsa_wood = np.load(woody_allocations_file)
        file2 = True

    else:
        print("Building woody allocation combinations: \n")
        aleafw = np.arange(10., 80.1, 0.0125e1, dtype=np.float64)
        arootw = np.arange(10., 80.1, 0.0125e1, dtype=np.float64)
        awood = np.arange(10., 80.1, 0.0125e1, dtype=np.float64)

        plsa_wood = [[a, b, c] for a in aleafw for b in awood
                     for c in arootw if (a + b + c) == 100]
        np.save(woody_allocations_file, np.array(plsa_wood))

    if verbose:
        print('Number of ALLOCATION combinations (grass + wood) - aleaf/awood/aroot = %d' %
              (len(plsa_grass) + len(plsa_wood)))

    if file1:
        a1 = plsa_wood / 100.0
    else:
        a1 = np.array(plsa_wood) / 100.0
    if file2:
        a2 = plsa_grass / 100.0
    else:
        a2 = np.array(plsa_grass) / 100.0

    return a1, a2


def table_gen(NPLS, fpath=None):
    """AKA main - generate a trait table for CAETÊ - save it to a .csv"""

    def calc_ratios(pool):

        # TODO increase ranges to left
        pool_n2c = np.linspace(0.0005, 0.06, 1000)
        pool_p2c = np.linspace(0.2e-5, 0.008, 1000)  # TODO

        if pool == 'leaf' or pool == 'root':
            pass
        else:
            pool_n2c /= 11.37
            pool_p2c /= 11.37

        x = [[a, b] for a in pool_n2c for b in pool_p2c if (
            (a / b) >= 2.0) and ((a / b) <= 60.0)]
        assert len(x) > 0, "zero len"
        shuffle(x)
        return x

    diffg, diffw = assertion_data_size(NPLS)
    plsa_wood, plsa_grass = turnover_combinations(True)

    alloc_w = []
    alloc_g = []
    r_ceil = 30000

# REVER O TEMPO DE RESIDÊNCIA DAS RAÌZES FINAS - VARIAR ENTRE 1 mes e 2 anos
    index0 = 0
    # rtime = vec_ranging(np.random.beta(2, 4, r_ceil),
    #                     0.083333, 2)
    rtime = np.random.uniform(0.08333333333333333, 4, r_ceil)
    print("CREATE GRASSy STRATEGIES - Checking potential npp/alocation")
    while index0 < diffg:
        restime = np.zeros(shape=(3,), dtype=np.float64)

        allocatio = plsa_grass[np.random.randint(0, plsa_grass.shape[0])]
        restime[0] = rtime[np.random.randint(0, r_ceil)]
        restime[1] = 0.0
        restime[2] = rtime[np.random.randint(0, r_ceil)]

        data_to_test0 = np.concatenate((restime, allocatio), axis=0,)
        if check_viability(data_to_test0, False):
            alloc_g.append(data_to_test0)
            index0 += 1
        sys.stdout.write('\r%s' % (str(index0)))
    sys.stdout.flush()
    print("\n")
    print("CREATE WOODY STRATEGIES - Checking potential npp/alocation")
    # Creating woody plants (maybe herbaceous)
    index1 = 0
    # rtime_wood = vec_ranging(np.random.beta(
    # 2, 4, r_ceil), 1.0, 150)
    rtime_wood = np.random.uniform(0.08333333333333333, 80.0, r_ceil)
    while index1 < diffw:
        restime = np.zeros(shape=(3,), dtype=np.float64)
        allocatio = plsa_wood[np.random.randint(0, plsa_wood.shape[0])]
        restime[0] = rtime[np.random.randint(0, r_ceil)]
        restime[1] = rtime_wood[np.random.randint(0, r_ceil)]
        restime[2] = rtime[np.random.randint(0, r_ceil)]
        data_to_test1 = np.concatenate((restime, allocatio), axis=0,)
        if check_viability(data_to_test1, True):
            alloc_w.append(data_to_test1)
            index1 += 1
        sys.stdout.write('\r%s' % (str(index1)))
    sys.stdout.flush()
    print("\n")

    alloc_g = np.array(alloc_g)
    alloc_w = np.array(alloc_w)

    alloc = np.concatenate((alloc_g, alloc_w), axis=0,)

    # # # COMBINATIONS
    # # # Random samples from  distributions (g1, tleaf ...)
    # # # Random variables
    g1 = np.random.uniform(1.0, 15.0, NPLS)
    # g1 = vec_ranging(np.random.beta(1.2, 2, NPLS), 1.0, 15.0) # dimensionles
    # # vcmax = np.random.uniform(3e-5, 100e-5,N) # molCO2 m-2 s-1
    resorption = np.random.uniform(0.3, 0.6, NPLS)

    # # C4 STYLE
    c4 = np.zeros((NPLS,), dtype=np.float64)
    n123 = ceil(alloc_g.shape[0] * 0.60)
    c4[0: n123 - 1] = 1.0

    # # Nitrogen and Phosphorus content in carbon pools
    # # C : N : P

    leaf = np.array(sample(calc_ratios('leaf'), int(NPLS)))
    leaf_n2c = leaf[:, 0]
    leaf_p2c = leaf[:, 1]

    wood = np.array(sample(calc_ratios('wood'), int(NPLS)))
    awood_n2c = wood[:, 0]
    awood_p2c = wood[:, 1]
    test = alloc[:, 4] == 0.0

    np.place(awood_n2c, test, 0.0)
    np.place(awood_p2c, test, 0.0)

    root = np.array(sample(calc_ratios('root'), int(NPLS)))
    froot_n2c = root[:, 0]
    froot_p2c = root[:, 1]

    # new traits
    pdia = np.random.uniform(0.01, 0.05, NPLS)
    np.place(pdia, test, 0.0)
    woods = np.where(alloc[:, 4] > 0.0)[0]
    # return woods

    for i in woods:
        if np.random.normal() > 0:
            pdia[i] = 0.0

    amp = np.random.uniform(0.001, 0.999, NPLS)

    pls_id = np.arange(NPLS)

    stack = (pls_id, g1, resorption, alloc[:, 0], alloc[:, 1], alloc[:, 2],
             alloc[:, 3], alloc[:, 4], alloc[:, 5], c4, leaf_n2c,
             awood_n2c, froot_n2c, leaf_p2c, awood_p2c, froot_p2c,
             amp, pdia)

    head = ['PLS_id', 'g1', 'resopfrac', 'tleaf', 'twood', 'troot', 'aleaf', 'awood', 'aroot', 'c4',
            'leaf_n2c', 'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c', 'froot_p2c',
            'amp', 'pdia']

    if fpath is not None:

        pls_table = np.vstack(stack)

        # # ___side_effects
        if not fpath.exists():
            os.system(f" mkdir -p {fpath.resolve()}")
        fnp = Path(os.path.join(fpath, 'pls_attrs.csv')).resolve()
        with open(fnp, mode='w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(head)
            for x in range(pls_table.shape[1]):
                writer.writerow(list(pls_table[:, x]))
            # writer.writerows(pls_table)

    pls_table = np.vstack(stack[1:])
    return np.asfortranarray(pls_table, dtype=np.float64)
