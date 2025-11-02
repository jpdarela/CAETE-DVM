# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """


import copy
import csv

from pathlib import Path
from typing import Callable, Dict, Union, Any, Optional
from numpy.typing import NDArray

from joblib import dump
import numpy as np

from community import community
from config import fetch_config

from caete_jit import process_tuples

config_data = fetch_config()

ntraits =  config_data.metacomm.ntraits
npls =  config_data.metacomm.npls_max


class pls_table:

    """ Interface for the main table of plant life strategies (Plant prototypes).
        Random subsamples without replacement are taken from this table to create communities. Each community as a set of unique PLS.
        The main table should have a large number of PLSs (npls ~ 20000, for example)
        representing a global set of virtual plant prototypes.

        Use the plsgen.py script to generate a PLS table).
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

        assert self.npls >= npls, "The number of PLSs in the main table should be greater than the number of PLSs in a community."
        assert self.ntraits == ntraits, "The number of traits in the main table should be equal to the number of traits in a community."


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


class metacommunity:
    """Represents a collection of plant communities.
    """

    def __init__(self, community_count:int, get_from_main_table:Optional[Callable]) -> None:
        """A collection of plant communities.

        Args:
            n (int): number of communities that will be created in the metacommunity.
            get_from_main_table (Callable): a function that returns a random sample of PLSs from the main table.
            The get_from_main_table function is passed from the region class. It is used to manage PLSs in communities.
        """
        # Community-level variables
        # Functional Identity
        # TODO: IMPLEMENT entropy and diversity
        if get_from_main_table is None:
            self.dummy = True
            return None
        self.dummy = False
        self.cwm = np.zeros(ntraits, order='F')
        self.cwv = np.zeros(ntraits, order='F')
        # self.entropy = np.zeros(ntraits, order='F') Shannon entropy
        # self.diversity = np.zeros(ntraits, order='F') Simpson diversity

        # Communities
        self.communities:Dict[int, community] = {}
        self.get_table = get_from_main_table # Function defined in the region class [region.py]
        self.comm_npls = copy.deepcopy(npls)
        self.mask: NDArray[np.int8] = np.zeros(community_count, dtype=np.int8)

        for i in range(community_count):
            # Create the communities
            self.communities[i] = community(self.get_table(self.comm_npls))
            # Set active at start
            self.communities[i].masked = np.int8(0)
            # print(f"Community {i} created with {self.communities[i].npls} PLSs.")
        # Update the metacommunity mask
        self.update_mask()
        return None


    def update_mask(self)-> None:
        """
        Updates the metacommunity mask based on its communities states.
        """
        for k, community in self.communities.items():
            i = int(k)
            self.mask[i] = community.masked


    def wrapp_state(self, year:int, pl:bool) -> Dict[str, Any]:
        """Returns a dictionary with the state of the metacommunity."""
        state = {}
        state['communities'] = {}
        counter = 1
        cveg = 0.0
        cleaf = 0.0
        croot = 0.0
        cwood = 0.0
        anpp = 0.0
        uptake_costs = 0.0
        for k, community in self.communities.items():
            if community.masked:
                continue
            state['communities'][k] = {}
            state['communities'][k]['vp_cleaf'] = community.vp_cleaf
            state['communities'][k]['vp_croot'] = community.vp_croot
            state['communities'][k]['vp_cwood'] = community.vp_cwood
            state['communities'][k]['vp_ocp'] = community.vp_ocp
            state['communities'][k]['id'] = community.id[community.vp_lsid]
            # state['communities'][k]['vp_lsid'] = community.vp_lsid
            if pl:
                state['communities'][k]['limitation_status_leaf'] = process_tuples(community.limitation_status_leaf) #type: ignore
                state['communities'][k]['limitation_status_root'] = process_tuples(community.limitation_status_root) #type: ignore
                state['communities'][k]['limitation_status_wood'] = process_tuples(community.limitation_status_wood) #type: ignore
                state['communities'][k]['uptk_strat_n'] = process_tuples(community.uptake_strategy_n) #type: ignore
                state['communities'][k]['uptk_strat_p'] = process_tuples(community.uptake_strategy_p) #type: ignore
            state['communities'][k]['cleaf'] = community.cleaf
            state['communities'][k]['croot'] = community.croot
            state['communities'][k]['cwood'] = community.cwood
            state['communities'][k]['masked'] = community.masked
            state['communities'][k]['uptake_costs'] = community.uptake_costs
            state['communities'][k]['anpp'] = community.anpp
            state['communities'][k]['shannon_entropy'] = community.shannon_entropy
            state['communities'][k]['shannon_diversity'] = community.shannon_diversity
            state['communities'][k]['shannon_evenness'] = community.shannon_evenness
            anpp += community.anpp
            uptake_costs += community.uptake_costs
            cleaf += community.cleaf
            croot += community.croot
            cwood += community.cwood
            cveg += community.cleaf + community.croot + community.cwood
            counter += 1
        state['cveg'] = cveg / counter
        state['cleaf'] = cleaf / counter
        state['croot'] = croot / counter
        state['cwood'] = cwood / counter
        state['anpp'] = anpp / counter
        state['uptake_costs'] = uptake_costs / counter
        state['mask'] = self.mask
        state['year'] = year
        return state


    def save_state(self, file_path: Union[str, Path], year:int, pl:bool) -> None:
        """Save the state of the metacommunity to a binary file using joblib.

        Args:
            file_path (Union[str, Path]): The path to the file where the state will be saved.
        """
        state = self.wrapp_state(year, pl)
        with open(file_path, 'wb') as f:
            dump(value=state, filename=f, compress=('lzma', 6), protocol=5) # type: ignore


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


def main():
    # Toy example to test the classes functionality
    main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-5000.csv"))

    # this function mimics the behavior of the get_from_main_table function in the region class
    # Only for testing purposes
    def __get_from_main_table(comm_npls, table = main_table):
        """Returns a number of IDs (in the main table) and the respective
        functional identities (PLS table) to set or reset a community

        Args:
        comm_npls: (int) Number of PLS in the output table (must match npls_max (see caete.toml))"""
        assert comm_npls > 0, "Number of PLS must be greater than 0"

        if comm_npls == 1:
            idx = np.random.choice(table.shape[1], 1, replace=False)[0]

            return idx, table[:, idx]

        idx = np.random.choice(table.shape[1], comm_npls, replace=False)
        return idx, table[:, idx]

    # Create a metacommunity with 99 communities. the number of PLSs in each community is set in the caete.toml file
    mt = metacommunity(1, __get_from_main_table)
    print(f"Number of communities: {len(mt)}")
    print(f"Number of PLSs in each community: {mt.comm_npls}")

    mt2 = metacommunity(1, None)
    return mt

if __name__ == "__main__":
    mt = main()


