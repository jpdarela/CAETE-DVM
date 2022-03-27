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
from math import ceil
import csv
from pathlib import Path
from joblib import Parallel, delayed
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
    lim = 0.01
    rtur = np.array(model.spinup3(lim, trait_values))
    if wood:
        if rtur[0] <= lim and rtur[1] <= lim and rtur[2] <= lim:
            return False
        return True
    else:
        if rtur[0] <= lim and rtur[1] <= lim:
            return False
        return True


def assertion_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = 0.2
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
        aleafg = np.arange(3.5, 96.5, 0.0125e1, dtype=np.float64)
        arootg = np.arange(3.5, 96.5, 0.0125e1, dtype=np.float64)

        plsa_grass = [[a, 0.0, c]
                      for a in aleafg for c in arootg if (a + c) == 100]
        np.save(grassy_allocations_file, np.array(plsa_grass))

    if os.path.exists(woody_allocations_file):
        plsa_wood = np.load(woody_allocations_file)
        file2 = True

    else:
        print("Building woody allocation combinations: \n")
        aleafw = np.arange(5., 95.1, 0.0125e1, dtype=np.float64)
        arootw = np.arange(5., 95.1, 0.0125e1, dtype=np.float64)
        awood = np.arange(5., 95.1, 0.0125e1, dtype=np.float64)

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


def calc_ratios1(NPLS):
    # LEAF POOL
    # Reich, P. B., & Oleksyn, J. (2004). 
    # Global patterns of plant leaf N and P in relation to temperature and latitude. 
    # Proceedings of the National Academy of Sciences, 101(30), 11001–11006. 
    # https://doi.org/10.1073/pnas.0403588101
    N0 = 0.001
    NM = 0.05
    P0 = 0.0002
    PM = 0.0095

    if os.path.exists(Path("./NP1.npy")):
        x1 = np.load("./NP1.npy")
    else:
        print('NP1...')
        pool_n2c = np.linspace(N0, NM, 5000)
        pool_p2c = np.linspace(P0, PM, 5000)

        x = [[a, b] for a in pool_n2c for b in pool_p2c if (
            (a / b) >= 1.5) and ((a / b) <= 70.0)]
        assert len(x) > 0, "zero len"
        x1 = np.array(x)
        np.save("./NP1.npy", x1)
    idx = np.random.randint(0, x1.shape[0], size=NPLS)
    sampleNP = x1[idx, :]
    return sampleNP


def calc_ratios2(NPLS):
    # WOOD POOL
    # Heineman, K. D., Turner, B. L., & Dalling, J. W. (2016). 
    # Variation in wood nutrients along a tropical soil fertility gradient. 
    # New Phytologist, 211(2), 440?454. https://doi.org/10.1111/nph.13904
    N0 = 0.001
    NM = 0.01
    P0 = 3.12e-5
    PM = 0.0035

    if os.path.exists(Path("./NP2.npy")):
        x1 = np.load("./NP2.npy")
    else:
        print('NP2...')
        pool_n2c = np.linspace(N0, NM, 5000)
        pool_p2c = np.linspace(P0, PM, 5000)

        x = [[a, b] for a in pool_n2c for b in pool_p2c if (
            (a / b) >= 4) and ((a / b) <= 180.0)]
        assert len(x) > 0, "zero len"
        x1 = np.array(x)
        np.save("./NP2.npy", x1)
    idx = np.random.randint(0, x1.shape[0], size=NPLS)
    sampleNP = x1[idx, :]
    return sampleNP

def calc_ratios3(NPLS):
    # FINE ROOT POOL
    # Iversen, C., McCormack, M., Baer, J., Powell, A., Chen, W., Collins, C.,
    # Fan, Y., Fanin, N., Freschet, G., Guo, D., Hogan JA, Kou, L., Laughlin, D.,
    # Lavely, E., Liese, R., Lin, D., Meier, I., Montagnoli, A., 
    # Roumet, C., … Zadworny, M. (2021). Fine-Root Ecology Database (FRED): A Global 
    # Collection of Root Trait Data with Coincident Site, Vegetation, Edaphic, and
    # Climate Data, Version 3. Oak Ridge National Laboratory, TES SFA,
    # U.S. Department of Energy, Oak Ridge, Tennessee, U.S.A. 
    # https://doi.org/https://doi.org/10.25581/ornlsfa.014/1459186
    # AND some references therein
    N0 = 0.001
    NM = 0.06
    P0 = 0.0003
    PM = 0.005

    if os.path.exists(Path("./NP3.npy")):
        x1 = np.load("./NP3.npy")
    else:
        print('NP3...')
        pool_n2c = np.linspace(N0, NM, 5000)
        pool_p2c = np.linspace(P0, PM, 5000)

        x = [[a, b] for a in pool_n2c for b in pool_p2c if (
            (a / b) >= 2) and ((a / b) <= 80)]
        assert len(x) > 0, "zero len"
        x1 = np.array(x)
        np.save("./NP3.npy", x1)
    idx = np.random.randint(0, x1.shape[0], size=NPLS)
    sampleNP = x1[idx, :]
    return sampleNP


def table_gen(NPLS, fpath=None):
    """AKA main - generate a trait table for CAETÊ - save it to a .csv"""

    diffg, diffw = assertion_data_size(NPLS)
    plsa_wood, plsa_grass = turnover_combinations(True)

    alloc_w = []
    alloc_g = []
    r_ceil = 10000

# REVER O TEMPO DE RESIDÊNCIA DAS RAÌZES FINAS - VARIAR ENTRE 1 mes e 2 anos
    index0 = 0
    # rtime = vec_ranging(np.random.beta(2, 4, r_ceil),
    #                     0.083333, 2)
    rtime_leaf = np.random.uniform(0.166, 8.3333, r_ceil)
    rtime_froot = np.random.uniform(0.08333, 8.3333, r_ceil)
    print("CREATE GRASSy STRATEGIES - Checking potential npp/alocation")
    while index0 < diffg:
        restime = np.zeros(shape=(3,), dtype=np.float64)

        allocatio = plsa_grass[np.random.randint(0, plsa_grass.shape[0])]
        restime[0] = rtime_leaf[np.random.randint(0, r_ceil)]
        restime[1] = 0.0
        restime[2] = rtime_froot[np.random.randint(0, r_ceil)]

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
    rtime_wood = np.random.uniform(0.20, 100.0, r_ceil)
    while index1 < diffw:
        restime = np.zeros(shape=(3,), dtype=np.float64)
        allocatio = plsa_wood[np.random.randint(0, plsa_wood.shape[0])]
        restime[0] = rtime_leaf[np.random.randint(0, r_ceil)]
        restime[1] = rtime_wood[np.random.randint(0, r_ceil)]
        restime[2] = rtime_froot[np.random.randint(0, r_ceil)]
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
    g1 = np.random.uniform(0.1, 19.0, NPLS)
    # g1 = vec_ranging(np.random.beta(1.2, 2, NPLS), 1.0, 15.0) # dimensionles
    # # vcmax = np.random.uniform(3e-5, 100e-5,N) # molCO2 m-2 s-1
    resorption = np.random.uniform(0.2, 0.7, NPLS)

    # # C4 STYLE
    c4 = np.zeros((NPLS,), dtype=np.float64)
    n123 = ceil(alloc_g.shape[0] * 0.50)
    c4[0: n123 - 1] = 1.0

    # # Nitrogen and Phosphorus content in carbon pools
    # # C : N : P

    leaf = calc_ratios1(NPLS)
    leaf_n2c = leaf[:, 0]
    leaf_p2c = leaf[:, 1]

    wood = calc_ratios2(NPLS)
    awood_n2c = wood[:, 0]
    awood_p2c = wood[:, 1]

    test = alloc[:, 4] == 0.0

    np.place(awood_n2c, test, 0.0)
    np.place(awood_p2c, test, 0.0)

    root = calc_ratios3(NPLS)
    froot_n2c = root[:, 0]
    froot_p2c = root[:, 1]

    # new traits
    pdia = np.random.uniform(0.01, 0.10, NPLS)
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
        fnp = Path(os.path.join(fpath, f'pls_attrs-{NPLS}.csv')).resolve()
        with open(fnp, mode='w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(head)
            for x in range(pls_table.shape[1]):
                writer.writerow(list(pls_table[:, x]))
            # writer.writerows(pls_table)

    pls_table = np.vstack(stack[1:])
    return np.asfortranarray(pls_table, dtype=np.float64)
