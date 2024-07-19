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
# Residence times and nutrient ratios are randomly generated. Parameters are descibed in the pls_gen.toml file
import os
import sys
from config import fortran_compiler_dlls
if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_compiler_dlls)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")


from math import ceil
from typing import Tuple
import csv
import tomllib as tl
from pathlib import Path
import numpy as np
from caete_module import photo as model

__author__ = 'JP Darela'

# Read the pls_gen.toml file to get the parameters
with open('pls_gen.toml', 'rb') as f:
    data = tl.load(f)

GRASS_FRAC = data["parameters"]["grass_pls_fraction"]

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

def get_parameters():
    """ Get parameters from the pls_gen.toml file """

    with open('pls_gen.toml', 'rb') as f:
        data = tl.load(f)

    return data

def check_viability(trait_values, awood=False):

    """ Check the viability of allocation & residence time combinations.
        Some PLS combinations of allocation coefficients and residence times
        are not 'biomass acumulators' at low npp (< 0.01 kg m⁻² year⁻¹)
        do not have enough mass of carbon (< 0.001 kg m⁻²) in all CVEG compartments
        input:
        trait_values: np.array(shape=(6,), dtype=f64) allocation and residence time combination (possible PLS)
        wood: bool  Is this a woody PLS?
        output:bool True if the PLS is viable, False otherwise
    """
    data = get_parameters()
    lim = data["parameters"]["wood_cmin"]
    npp = data["parameters"]["wood_low_npp"]

    if awood:
        rtur = np.array(model.spinup3(npp, trait_values))
        if rtur[0] < lim or rtur[1] < lim or rtur[2] < lim:
            return False
        return True

    # Grasses
    lim = data["parameters"]["grass_cmin"]
    npp = data["parameters"]["grass_low_npp"]
    rtur = np.array(model.spinup3(npp, trait_values))
    if rtur[0] < lim or rtur[1] < lim:
        return False
    return True

def assert_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = GRASS_FRAC
    diffg = ceil(dsize * g2w_ratio)
    diffw = int(dsize - diffg)
    assert diffg + diffw == dsize
    return diffg, diffw

def allocation_combinations() -> Tuple[np.ndarray[float,], np.ndarray[float,]]:
    """ Generate allocation combinations for woody and grass plants based on Dirichlet distribution

    Returns:
        _type_: Tuple[np.ndarray, np.ndarray]
    """
    data = get_parameters()
    ma = data["parameters"]["minimum_allocation"]
    num_samples = 1_000_000
    alpha_wood = np.array([1.0, 1.0, 1.0])
    alpha_grass = np.array([1.0, 1.0])
    woody_comb = np.random.dirichlet(alpha_wood, num_samples)
    grass_tmp = np.random.dirichlet(alpha_grass, num_samples)

    grass_comb = np.zeros((num_samples, 3))
    grass_comb[:, 0] = grass_tmp[:, 0]
    grass_comb[:, 1] = 0.0
    grass_comb[:, 2] = grass_tmp[:, 1]

    grass_final = np.array([combo for combo in grass_comb if (combo[0] > ma or combo[2] > ma)])
    woody_final = np.array([combo for combo in woody_comb if (combo[0] > ma or combo[1] > ma or combo[2] > ma)])

    return woody_final, grass_final

def nutrient_ratios(n, N_min, N_max, P_min, P_max):
    sample_NP = np.zeros((n, 2))

    N_C = np.random.uniform(N_min, N_max, n)
    P_C = np.random.uniform(P_min, P_max, n)

    sample_NP[:, 0] = N_C
    sample_NP[:, 1] = P_C

    return sample_NP

def table_gen(NPLS, fpath=None):
    """AKA main - generate a trait table for CAETÊ - save it to a .csv"""

    # Read the pls_gen.toml file to get the parameters
    data = get_parameters()

    rwoody = data["residence_time"]["woody"]
    rgrass = data["residence_time"]["grass"]

    diffg, diffw = assert_data_size(NPLS)
    alloc_wood, alloc_grass = allocation_combinations()

    alloc_w = []
    alloc_g = []
    r_ceil = 10000

    if GRASS_FRAC == 0.0:
        pass
    else:
        index0 = 0
        rtime_leaf =  np.random.uniform(rgrass["leaf_min"], rgrass["leaf_max"], r_ceil)
        rtime_froot = np.random.uniform(rgrass["root_min"], rgrass["root_max"], r_ceil)
        # print("CREATE GRASSy STRATEGIES - Checking potential npp/alocation")
        while index0 < diffg:
            restime = np.zeros(shape=(3,), dtype=np.float64)

            allocatio = alloc_grass[np.random.randint(0, alloc_grass.shape[0])]
            restime[0] = rtime_leaf[np.random.randint(0, r_ceil)]
            restime[1] = 0.0
            restime[2] = rtime_froot[np.random.randint(0, r_ceil)]

            data_to_test0 = np.concatenate((restime, allocatio), axis=0,)
            if check_viability(data_to_test0):
                alloc_g.append(data_to_test0)
                index0 += 1
        #     sys.stdout.write('\r%s' % (str(index0)))
        # sys.stdout.flush()

    # print("\n")
    if GRASS_FRAC == 1.0:
        pass
    else:
        index1 = 0
        rtime_leaf =  np.random.uniform(rwoody["leaf_min"], rwoody["leaf_max"], r_ceil)
        rtime_froot = np.random.uniform(rwoody["root_min"], rwoody["root_max"], r_ceil)
        rtime_wood = np.random.uniform(rwoody["wood_min"], rwoody["wood_max"], r_ceil)
        # print("CREATE WOODY STRATEGIES - Checking potential npp/alocation")
        while index1 < diffw:
            restime = np.zeros(shape=(3,), dtype=np.float64)
            allocatio = alloc_wood[np.random.randint(0, alloc_wood.shape[0])]
            restime[0] = rtime_leaf[np.random.randint(0, r_ceil)]
            restime[1] = rtime_wood[np.random.randint(0, r_ceil)]
            restime[2] = rtime_froot[np.random.randint(0, r_ceil)]
            data_to_test1 = np.concatenate((restime, allocatio), axis=0,)
            if check_viability(data_to_test1, True):
                alloc_w.append(data_to_test1)
                index1 += 1
        #     sys.stdout.write('\r%s' % (str(index1)))
        # sys.stdout.flush()
        # print("\n")

    alloc_g = np.array(alloc_g)
    alloc_w = np.array(alloc_w)

    if GRASS_FRAC == 0.0:
        alloc = alloc_w
    elif GRASS_FRAC == 1.0:
        alloc = alloc_g
    else:
        alloc = np.concatenate((alloc_g, alloc_w), axis=0,)

    # # # COMBINATIONS
    g1 = np.random.uniform(2, 20.0, NPLS)
    resorption = np.random.uniform(0.1, 0.8, NPLS)

    # # C4 type
    c4 = np.zeros((NPLS,), dtype=np.float64)
    if GRASS_FRAC > 0.0 and index0 > 1:
        n123 = ceil(alloc_g.shape[0] * 0.50)
        c4[0: n123 - 1] = 1.0

    # # Nitrogen and Phosphorus content in carbon pools
    # # C : N : P

    # Value for C == 1.0
    nr = data["nutrient_carbon_ratios"]

    # LEAF POOL
    # Reich, P. B., & Oleksyn, J. (2004).
    # Global patterns of plant leaf N and P in relation to temperature and latitude.
    # Proceedings of the National Academy of Sciences, 101(30), 11001–11006.
    # https://doi.org/10.1073/pnas.0403588101
    # leaf = calc_ratios1(NPLS)
    N0 = nr["leaf_n2c"]["min"] #0.005
    NM = nr["leaf_n2c"]["max"] #0.05
    P0 = nr["leaf_p2c"]["min"] #0.0005
    PM = nr["leaf_p2c"]["max"] #0.005
    leaf = nutrient_ratios(NPLS, N0, NM, P0, PM)
    leaf_n2c = leaf[:, 0]
    leaf_p2c = leaf[:, 1]


    # WOOD POOL
    # Heineman, K. D., Turner, B. L., & Dalling, J. W. (2016).
    # Variation in wood nutrients along a tropical soil fertility gradient.
    # New Phytologist, 211(2), 440?454. https://doi.org/10.1111/nph.13904
    # wood = calc_ratios2(NPLS)
    N0 = nr["wood_n2c"]["min"]# 0.001
    NM = nr["wood_n2c"]["max"]# 0.005
    P0 = nr["wood_p2c"]["min"]# 7.5e-6
    PM = nr["wood_p2c"]["max"]# 0.00025
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
    N0 = nr["root_n2c"]["min"] #0.001 # g(Nutrient)/g(Carbon)
    NM = nr["root_n2c"]["max"] #0.06
    P0 = nr["root_p2c"]["min"] #0.0003
    PM = nr["root_p2c"]["max"] #0.005
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
            os.makedirs(fpath.resolve(), exist_ok=True,)
            # os.system(f" mkdir -p {fpath.resolve()}")
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
    table = table_gen(2000)