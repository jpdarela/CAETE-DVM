import os
import sys
from config import fortran_compiler_dlls

if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_compiler_dlls)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Union, Dict, Tuple

import copy
import numpy as np
from pandas import read_csv
from caete_module import global_par as gp
from caete_module import photo as m

def read_pls_table(pls_file):
    """Read the standard attributes table saved in csv format.
       Return numpy array (shape=(ntraits, npls), F_CONTIGUOUS)"""
    return np.asfortranarray(read_csv(pls_file).__array__()[:,1:].T).astype(np.float32)


# class pls:
#     """Plant Life Strategies (PLS) class. Instances of this class are used to define a
#         PLS as a structured array.
#     """

#     trait_names = ('g1', 'resopfrac', 'tleaf', 'twood', 'troot',
#                    'aleaf', 'awood', 'aroot', 'c4', 'leaf_n2c',
#                    'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c',
#                    'froot_p2c', 'amp', 'pdia')

#     dtypes = np.dtype([(name, _dtype) for name, _dtype in zip(trait_names, [np.float32] * len(trait_names))])

#     def __init__(self, id:int, pls_data:np.ndarray) -> None:
#         self.id = id
#         self.functional_identity = np.core.records.fromarrays(pls_data, dtype=pls.dtypes)


class pls_table:
    """ Interface for the main table of plant life strategies (Plant prototypes).
        Random subsamples without replacement are taken from this table to
        create communities. The main table should've a large number
        of PLSs (npls ~ 20000, for example).
    """
    def __init__(self, array:np.ndarray) -> None:
        """Provides an interface for the main table of PLSs.

        Args:
            array (np.ndarray(ntraits,npls)): PLS table.
        """
        self.table = array.astype(np.float32).copy(order='F')
        self.npls = self.table.shape[1]
        self.ntraits = self.table.shape[0]
        self.shape = self.table.shape
        self.id = np.arange(self.npls)

        assert self.npls > gp.npls, "The number of PLSs in the main table should be greater than the number of PLSs in a community."
        assert self.ntraits == gp.ntraits, "The number of traits in the main table should be equal to the number of traits in a community."


    def __len__(self):
        return self.npls


    def __getitem__(self, index:int):
        return self.table[:,index]


    def get_random_pls(self):
        pls_ids = np.random.randint(0, self.npls-1)
        return pls_ids, self.table[:,id]


    def create_npls_table(self, comm_npls) -> Tuple[np.ndarray[int], np.ndarray[float]]:
        """_summary_

        Args:
            comm_npls (_type_): _description_

        Returns:
            _type_: _description_
        """
        idx = np.random.randint(0, self.npls-1, comm_npls)
        table = self.table[:,idx]
        output = namedtuple("pls_data", ["id", "pls_table"])
        return output(idx, table)


class community:
    """Represents a community of plants. Instances of this class are used to
       create metacommunities."""


    def __init__(self, pls_data:Tuple[np.ndarray[int], np.ndarray[float]]) -> None:
        """An assembly of plants.
        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): Two arrays, the first stores the
                ids of the PLSs in the main table, the second stores the functional identity of PLSs.
        """
        self.id = pls_data.id
        self.pls_array = pls_data.pls_table
        self.npls = self.pls_array.shape[1]
        self.shape = self.pls_array.shape
        self.alive = np.ones(gp.npls, dtype=bool)

        # BIOMASS_STATE - we add some biomass to the plants
        # in the community to start the simulation
        self.vp_cleaf = np.random.uniform(0.3,0.4,self.npls)
        self.vp_croot = np.random.uniform(0.3,0.4,self.npls)
        self.vp_cwood = np.random.uniform(5.0,6.0,self.npls)
        self.vp_sto = np.zeros(shape=(3, self.npls), order='F')
        self.vp_sto[0,:] = np.random.uniform(0.0, 0.1, self.npls)
        self.vp_sto[1,:] = np.random.uniform(0.0, 0.01, self.npls)
        self.vp_sto[2,:] = np.random.uniform(0.0, 0.001, self.npls)

        # we set the wood biomass of the plants that are not woody to zero
        self.vp_cwood[self.pls_array[6,:] == 0.0] = 0.0

        self.vp_ocp, _, _, _ = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                               self.vp_cwood, self.pls_array[6, :])

        self.vp_lsid = np.where(self.vp_ocp > 0.0)[0]
        self.ls = self.vp_lsid.size
        self.uptk_costs = np.zeros(self.npls, order='F')


    def update_lsid(self, occupation:np.ndarray):
        """_summary_

        Args:
            occupation (np.ndarray): _description_
        """
        self.vp_lsid = np.where(occupation > 0.0)[0]


    def restore_from_main_table(self, pls_data:Tuple[np.ndarray[int], np.ndarray[float]]) -> None:
        """A call to the __init__ method.
        This is used to reset the community to a initial state
        with a new reandom sample of PLSs from the main table.

        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): a named tuple
        """
        self.__init__(pls_data)


    # @lru_cache(maxsize=None)
    def __getitem__(self, index:int):
        """Gets a PLS (1D array) for given index.

        Args:
            index (int): _description_

        Returns:
            1D np.ndarray: PLS of the plant at the given index.
        """
        return self.pls_array[:,index]


    def __setitem__(self, index:int, value:np.ndarray):
        """_summary_

        Args:
            index (int): _description_
            value (np.ndarray): _description_
        """
        self.pls_array[:,index] = value


    def __len__(self):
        return self.shape[1]


class metacommunity:
    """Represents a collwction of plant communities.
    """

    def __init__(self, n:int, main_table:pls_table) -> None:
        """A collection of plant communities.

        Args:
            n (int): number of communities that will be created in the metacommunity.
            main_table (pls_table): a pls_table interface object. The main table of plant life strategies.
        """

        self.communities:dict = {}
        self.pls_table = main_table
        self.comm_npls = copy.deepcopy(gp.npls)


        for i in range(n):
            self.communities[i] = community(self.pls_table.create_npls_table(gp.npls))

    # @lru_cache(maxsize=None)
    def __getitem__(self, index:Union[int, str]):
        """get a community by index. The index can be an integer or a string.
        The communuties are stored in a dictionary with int keys.

        Args:
            index (Union[int, str]): _description_

        Raises:
            KeyError: Raised if the key is not in the dictionary.

        Returns:
            community: A community object from the metacommunity.
        """
        __val__ = self.communities.get(index)
        if __val__ is None:
            raise KeyError(f"Key {index} is not a community in this metacommunity.")
        return __val__


    def __setitem__(self, index:int, value:community):
        """_summary_

        Args:
            index (int): _description_
            value (community): _description_
        """
        self.communities[index] = value


    def __len__(self):
        return len(self.communities)


    def __iter__(self):
        return iter(self.communities.values())


def main():
    main_table = read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))
    table_obj = pls_table(main_table)

    # comm = community(table_obj.create_npls_table(10))

    return metacommunity(99, table_obj)
    # # print(mt[0].pls_table.table)
    # # print(mt[0].pls_table.id)
    # # print(mt[0].pls_table.get_random_pls())
    # # print(mt[0].pls_table.create_npls_table(10))
    # # print(len(mt))


if __name__ == "__main__":
    mt = main()
