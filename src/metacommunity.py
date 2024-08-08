import copy
import csv
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Any
from numpy.typing import NDArray

import numpy as np

from config import fortran_compiler_dlls

# Add the fortran compiler DLLs to the PATH
# This is necessary to load the fortran compiled DLLs in windows systems
if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_compiler_dlls)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

from caete_module import global_par as gp
from caete_module import photo as m


class pls_table:
    """ Interface for the main table of plant life strategies (Plant prototypes).
        Random subsamples without replacement are taken from this table to
        create communities. The main table should've a large number
        of PLSs (npls ~ 20000, for example).
    """
    def __init__(self, array:NDArray[np.float32]) -> None:
        """Initializes the PLS table.
        Args:
            array (np.ndarray(ntraits,npls)): PLS table.
        """
        self.table = array.astype(np.float32, copy=False, order='F')
        self.npls = self.table.shape[1]
        self.ntraits = self.table.shape[0]
        self.shape = self.table.shape
        self.id = np.arange(self.npls, dtype=np.int32)

        assert self.npls > gp.npls, "The number of PLSs in the main table should be greater than the number of PLSs in a community."
        assert self.ntraits == gp.ntraits, "The number of traits in the main table should be equal to the number of traits in a community."


    def __len__(self) -> int:
        return self.npls


    def __getitem__(self, index:int) -> NDArray[np.float32]:
        return self.table[:,index]


    def get_subsample(self, size: int) -> NDArray[np.float32]:
        """Get a random subsample of PLSs without replacement.

        Args:
            size (int): The number of PLSs to sample.

        Returns:
            np.ndarray: A subsample of the PLS table.
        """
        assert size <= self.npls, "The size of the subsample should be less than or equal to the number of PLSs in the main table."
        indices = np.random.choice(self.npls, size=size, replace=False)
        return self.table[:, indices]


    @staticmethod
    def read_pls_table(pls_file: Union[str, Path]) -> NDArray[np.float32]:
        """
        Read the standard attributes table saved in csv format.
        Return numpy array (shape=(ntraits, npls), F_CONTIGUOUS, dtype=np.float32).

        Args:
            pls_file (Union[str, Path]): Path to the csv file.

        Returns:
            np.ndarray: PLS table. Shape=(ntraits, npls), F_CONTIGUOUS, dtype=np.float32
        """

        with open(pls_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        # Convert the list to a NumPy array, skipping the first column(id) and row (header)
        array_data = np.array(data[1:])[:, 1:].astype(np.float32)

        # Transpose the array and convert to Fortran-contiguous order
        return np.asfortranarray(array_data.T)


class community:
    """Represents a community of plants. Instances of this class are used to
       create metacommunities."""


    def __init__(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """An assembly of plants.
        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): Two arrays, the first stores the
                ids of the PLSs in the main table, the second stores the functional identity of PLSs.
        """
        self.vp_ocp: NDArray[np.float32] | Any
        self.id: NDArray[np.int32] = pls_data[0]
        self.pls_array: NDArray[np.float32] = pls_data[1]
        self.npls: int = self.pls_array.shape[1]
        self.shape: Tuple[int, int] = self.pls_array.shape

        # BIOMASS_STATE - we add some biomass to the PLSs in the community to start the simulation
        self.vp_cleaf: NDArray[np.float32] = np.random.uniform(0.3,0.4,self.npls)
        self.vp_croot: NDArray[np.float32] = np.random.uniform(0.3,0.4,self.npls)
        self.vp_cwood: NDArray[np.float32] = np.random.uniform(5.0,6.0,self.npls)
        self.vp_sto: NDArray[np.float32] = np.zeros(shape=(3, self.npls), order='F')
        self.vp_sto[0,:] = np.random.uniform(0.0, 0.1, self.npls)
        self.vp_sto[1,:] = np.random.uniform(0.0, 0.01, self.npls)
        self.vp_sto[2,:] = np.random.uniform(0.0, 0.001, self.npls)

        # we set the wood biomass of the plants that are not woody to zero
        self.vp_cwood[self.pls_array[6,:] == 0.0] = 0.0

        # PFT_AREA_FRAC based on the mass-ratio hypothesis (Grime 1998)
        self.vp_ocp, _, _, _ = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                               self.vp_cwood, self.pls_array[6, :])

        # we get the indices of the plants that are present in the community
        self.vp_lsid: NDArray[np.intp] = np.where(self.vp_ocp > 0.0)[0]
        self.ls = self.vp_lsid.size

        # These needs to be passed today from the previous timestep
        # nutrient Uptake costs that is subtracted from NPP
        self.sp_uptk_costs = np.zeros(self.npls, order='F')
        # Real NPP. I.e. NPP after nutrient uptake costs and allocation (incl. limitation)
        self.construction_npp = np.zeros(self.npls, order='F')

        return None


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


    def __contains__(self, pls_id:int):
        """_summary_

        Args:
            pls_id (int): ID in the main table of the PLS.

        Returns:
            bool: true if the PLS ID is in the community, false otherwise.
        """
        return pls_id in self.id


    def update_lsid(self, occupation:np.ndarray):
        """_summary_

        Args:
            occupation (np.ndarray): _description_
        """
        self.vp_lsid = np.where(occupation > 0.0)[0]


    def get_free_lsid(self):
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return np.where(self.vp_ocp == 0.0)[0]


    def restore_from_main_table(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """A call to the __init__ method.
        This is used to reset the community to a initial state with a new random sample of PLSs from the main table.

        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): a tuple of arrays with the ids and the PLSs.

        Returns:
            None: The community is reset to the initial state with a new sample of PLSs from the main table.
        """
        self.__init__(pls_data)


    def seed_pls(self, pls_id: int, pls: np.ndarray) -> None:
        """Seeds a PLS in a free position. Please, ensure no duplicate PLS IDs.

        Args:
            pls_id (int): ID of the PLS.
            pls (np.ndarray): Array representing the PLS.
        """
        # Assert that the PLS ID is not in the community
        free_slots = self.get_free_lsid()
        if free_slots.size == 0:

            return None
        elif free_slots.size == 1:
            pos = free_slots[0]
        else:
            pos = free_slots[np.random.randint(0, free_slots.size)]
        self.id[pos] = pls_id
        self.pls_array[:, pos] = pls
        self.vp_cleaf[pos] = np.random.uniform(0.3, 0.4)
        self.vp_croot[pos] = np.random.uniform(0.3, 0.4)
        self.vp_cwood[pos] = np.random.uniform(5.0, 6.0)
        self.vp_sto[0, pos] = np.random.uniform(0.0, 0.1)
        self.vp_sto[1, pos] = np.random.uniform(0.0, 0.01)
        self.vp_sto[2, pos] = np.random.uniform(0.0, 0.001)
        self.vp_ocp, _, _, _ = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                                self.vp_cwood, self.pls_array[6, :])
        self.update_lsid(self.vp_ocp)


    def kill_pls(self, pos: int) -> None:
        """Kills a PLS in the community. This should not be used in the code. It is only for testing purposes.

        Args:
            pls_id (int): ID of the PLS to kill
        """
        self.vp_cleaf[pos] = 0.0
        self.vp_croot[pos] = 0.0
        self.vp_cwood[pos] = 0.0
        self.vp_sto[0, pos] = 0.0
        self.vp_sto[1, pos] = 0.0
        self.vp_sto[2, pos] = 0.0
        self.vp_ocp, _, _, _ = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                                self.vp_cwood, self.pls_array[6, :])
        self.update_lsid(self.vp_ocp)


    def get_unique_pls(self, pls_selector: Callable) -> Tuple[int, np.ndarray]:
        """Gets a unique PLS from the main table using the provided callable.

        Args:
            pls_selector (callable): A function that returns a tuple (pls_id, pls).

        Returns:
            Tuple[int, np.ndarray]: A unique PLS ID and the PLS array.
        """
        while True:
            pls_id, pls = pls_selector()
            if pls_id not in self.id:
                return pls_id, pls




class metacommunity:
    """Represents a collection of plant communities.
    """

    def __init__(self, n:int, get_from_main_table:Callable) -> None:
        """A collection of plant communities.

        Args:
            n (int): number of communities that will be created in the metacommunity.
            get_from_main_table (Callable): a function that returns a random sample of PLSs from the main table.
            The get_from_main_table function is passed from the region class. It is used to create new communities.
        """

        self.cwm = np.zeros(gp.ntraits, order='F')
        self.cwv = np.zeros(gp.ntraits, order='F')
        # Communities
        self.communities:Dict[int, community] = {}
        self.get_table = get_from_main_table # Function defined in the region class [caete.py L ~1400]
        self.comm_npls = copy.deepcopy(gp.npls)


        for i in range(n):
            self.communities[i] = community(self.get_table(self.comm_npls))

        return None


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
        __val__ = self.communities.get(index) #type: ignore
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


if __name__ == "__main__":

    # Example of how to use the metacommunity class

    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-25000.csv"))

    # these functions mimics the behavior of the get_from_main_table function in the region class
    # Only for testing purposes
    def get_from_main_table(comm_npls, table = main_table):
        """Returns a number of IDs (in the main table) and the respective
        functional identities (PLS table) to set or reset a community

        Args:
        comm_npls: (int) Number of PLS in the output table (must match npls_max (see caete.toml))"""
        idx = np.random.randint(0, comm_npls, comm_npls)
        return idx, table[:, idx]

    def get_one_from_main_table(table = main_table):
        """Returns a single ID (in the main table) and the respective
        functional identity (PLS table) to set a free position in a community"""
        idx = np.random.randint(0, table.shape[1] - 1)
        return idx, table[:, idx]

    # Create a metacommunity with 99 communities. the number of PLSs in each community is set in the caete.toml file
    mt = metacommunity(99, get_from_main_table)
    print(f"Number of communities: {len(mt)}")
    print(f"Number of PLSs in each community: {mt.comm_npls}")

    # Kill the PLS in the posiion 233 in the first community
    mt[0].kill_pls(233)

    # Get a unique PLS from the main table (i.e. a PLS that is not in the community)
    ident, func_id = mt[0].get_unique_pls(get_one_from_main_table)

    # Seed the unique PLS in the free slot
    mt[0].seed_pls(ident, func_id)



