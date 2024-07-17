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

# Procedures to create the set of PLant life strategies for CAETÊ runs
import os
import sys
from config import fortran_compiler_dlls
if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_compiler_dlls)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")


from math import ceil
import csv
from pathlib import Path
import numpy as np
from caete_module import photo as model

__author__ = 'JP Darela'


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

def check_viability(trait_values, awood=False):
    """ Check the viability of allocation(a) & residence time(ŧ) combinations.
        Some PLS combinations of allocation coefficients and residence times
        are not 'biomass acumulators' at low npp (< 0.01 kg m⁻² year⁻¹)
        do not have enough mass of carbon (< 0.01 kg m⁻²) in all CVEG compartments

        trait_values: np.array(shape=(6,), dtype=f64) allocation and residence time combination (possible PLS)
        wood: bool  Is this a woody PLS?
    """

    lim = 0.001
    npp = 0.001


    if awood:
        rtur = np.array(model.spinup3(npp, trait_values))
        if rtur[0] < lim or rtur[1] < lim or rtur[2] < lim:
            return False
        return True

    # Grasses
    lim = 0.001
    npp = 0.002
    rtur = np.array(model.spinup3(npp, trait_values))
    if rtur[0] < lim or rtur[1] < lim:
        return False
    return True

def assertion_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = 0.07
    diffg = ceil(dsize * g2w_ratio)
    diffw = int(dsize - diffg)
    assert diffg + diffw == dsize
    return diffg, diffw

def allocation_combinations():
    num_samples = 10000
    woody_comb = np.random.dirichlet(np.ones(3), num_samples)
    grassy_tmp = np.random.dirichlet(np.ones(2), num_samples)

    grassy_comb = np.zeros((num_samples, 3))
    grassy_comb[:, 0] = grassy_tmp[:, 0]
    grassy_comb[:, 1] = 0.0
    grassy_comb[:, 2] = grassy_tmp[:, 1]


    return woody_comb, grassy_comb

def nutrient_ratios(n, N_min, N_max, P_min, P_max):
    sample_NP = np.zeros((n, 2))

    N_C = np.random.uniform(N_min, N_max, n)
    P_C = np.random.uniform(P_min, P_max, n)

    sample_NP[:, 0] = N_C
    sample_NP[:, 1] = P_C

    return sample_NP

def table_gen(NPLS, fpath=None):
    """AKA main - generate a trait table for CAETÊ - save it to a .csv"""

    diffg, diffw = assertion_data_size(NPLS)
    plsa_wood, plsa_grass = allocation_combinations()

    alloc_w = []
    alloc_g = []
    r_ceil = 10000

    index0 = 0
    rtime_leaf =  np.random.uniform(0.08333, 2.0, r_ceil)
    rtime_froot = np.random.uniform(0.08333, 2.0, r_ceil)

    print("CREATE GRASSy STRATEGIES - Checking potential npp/alocation")
    while index0 < diffg:
        restime = np.zeros(shape=(3,), dtype=np.float64)

        allocatio = plsa_grass[np.random.randint(0, plsa_grass.shape[0])]
        restime[0] = rtime_leaf[np.random.randint(0, r_ceil)]
        restime[1] = 0.0
        restime[2] = rtime_froot[np.random.randint(0, r_ceil)]

        data_to_test0 = np.concatenate((restime, allocatio), axis=0,)
        if check_viability(data_to_test0):
            alloc_g.append(data_to_test0)
            index0 += 1
        sys.stdout.write('\r%s' % (str(index0)))
    sys.stdout.flush()

    print("\n")
    rtime_leaf =  np.random.uniform(0.08333, 10.0, r_ceil)
    rtime_froot = np.random.uniform(0.08333, 4.0, r_ceil)
    print("CREATE WOODY STRATEGIES - Checking potential npp/alocation")
    # Creating woody plants (maybe herbaceous)
    index1 = 0
    # rtime_wood = vec_ranging(np.random.beta(
    # 2, 4, r_ceil), 1.0, 150)
    rtime_wood = np.random.uniform(20, 600.0, r_ceil)
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
    g1 = np.random.uniform(2, 15.0, NPLS)
    # g1 = vec_ranging(np.random.beta(1.2, 2, NPLS), 1.0, 15.0) # dimensionles
    # # vcmax = np.random.uniform(3e-5, 100e-5,N) # molCO2 m-2 s-1
    resorption = np.random.uniform(0.2, 0.7, NPLS)

    # # C4 STYLE
    c4 = np.zeros((NPLS,), dtype=np.float64)
    n123 = ceil(alloc_g.shape[0] * 0.50)
    c4[0: n123 - 1] = 1.0

    # # Nitrogen and Phosphorus content in carbon pools
    # # C : N : P

    # Values for C == 1.0


    # LEAF POOL
    # Reich, P. B., & Oleksyn, J. (2004).
    # Global patterns of plant leaf N and P in relation to temperature and latitude.
    # Proceedings of the National Academy of Sciences, 101(30), 11001–11006.
    # https://doi.org/10.1073/pnas.0403588101
    # leaf = calc_ratios1(NPLS)
    N0 = 0.005
    NM = 0.05
    P0 = 0.0005
    PM = 0.005
    leaf = nutrient_ratios(NPLS, N0, NM, P0, PM)
    leaf_n2c = leaf[:, 0]
    leaf_p2c = leaf[:, 1]


    # WOOD POOL
    # Heineman, K. D., Turner, B. L., & Dalling, J. W. (2016).
    # Variation in wood nutrients along a tropical soil fertility gradient.
    # New Phytologist, 211(2), 440?454. https://doi.org/10.1111/nph.13904
    # wood = calc_ratios2(NPLS)
    N0 = 0.0005
    NM = 0.003
    P0 = 5e-5
    PM = 0.0005
    wood = nutrient_ratios(NPLS, N0, NM, P0, PM)
    awood_n2c = wood[:, 0]
    awood_p2c = wood[:, 1]

    test = alloc[:, 4] == 0.0

    np.place(awood_n2c, test, 0.0)
    np.place(awood_p2c, test, 0.0)


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
    N0 = 0.001 # g(Nutrient)/g(Carbon)
    NM = 0.06
    P0 = 0.0003
    PM = 0.005
    # root = calc_ratios3(NPLS)
    root = nutrient_ratios(NPLS, N0, NM, P0, PM)
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
        with open(fnp, mode='w', newline="\n") as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(head)
            for x in range(pls_table.shape[1]):
                writer.writerow(list(pls_table[:, x]))
            # writer.writerows(pls_table)

    pls_table = np.vstack(stack[1:])
    return np.asfortranarray(pls_table, dtype=np.float32)


if  __name__ == "__main__":
    table_gen(20000)