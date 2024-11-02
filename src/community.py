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


import os
import sys
from typing import Any, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from config import fortran_runtime

if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_runtime)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

from caete_module import photo as m


class community:
    """Represents a community of plants. Instances of this class are used to
       create metacommunities."""


    def __init__(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """An assembly of plants.
        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): Two arrays, the first stores the
                ids of the PLSs in the main table, the second stores the functional traits of PLSs.
        """
        self.vp_ocp: NDArray[np.float64] | Any
        self.id: NDArray[np.int32] = pls_data[0]
        self.pls_array: NDArray[np.float32] = pls_data[1]
        self.npls: int = self.pls_array.shape[1]
        self.shape: Tuple[int, ...] = self.pls_array.shape

        # BIOMASS_STATE - we add some biomass to the PLSs in the community to start the simulation
        self.vp_cleaf: NDArray[np.float32] = np.random.uniform(0.3,0.4,self.npls).astype(np.float32)
        self.vp_croot: NDArray[np.float32] = np.random.uniform(0.3,0.4,self.npls).astype(np.float32)
        self.vp_cwood: NDArray[np.float32] = np.random.uniform(5.0,6.0,self.npls).astype(np.float32)
        self.vp_sto: NDArray[np.float32] = np.zeros(shape=(3, self.npls), order='F', dtype=np.float32)
        self.vp_sto[0,:] = np.random.uniform(0.0, 0.1, self.npls)
        self.vp_sto[1,:] = np.random.uniform(0.0, 0.01, self.npls)
        self.vp_sto[2,:] = np.random.uniform(0.0, 0.001, self.npls)

        # Set the wood biomass of the plants that are not woody to zero
        self.vp_cwood[self.pls_array[6,:] == 0.0] = 0.0

        # PFT_AREA_FRAC based on the mass-ratio hypothesis (Grime 1998)
        self.vp_ocp, _, _, _ = m.pft_area_frac(self.vp_cleaf, self.vp_croot,
                                               self.vp_cwood, self.pls_array[6, :])

        # Get the indices of the plants that are present in the community
        self.vp_lsid: NDArray[np.intp] = np.where(self.vp_ocp > 0.0)[0]
        self.ls: int = self.vp_lsid.size
        self.masked: np.int8 = np.int8(0)
        # These needs to be passed today from the previous timestep
        # nutrient Uptake costs that is subtracted from NPP
        self.sp_uptk_costs: NDArray[np.float32] = np.zeros(self.npls, order='F', dtype=np.float32)
        # Real NPP. I.e. NPP after nutrient uptake costs and allocation (incl. limitation)
        # NOTE: This is not implemented
        self.construction_npp: NDArray[np.float32] = np.zeros(self.npls, order='F', dtype=np.float32)

        # These variables are used to output data once a year
        self.cleaf: np.float32 = np.float32(0.0)
        self.croot: np.float32 = np.float32(0.0)
        self.cwood: np.float32 = np.float32(0.0)
        self.csto:  np.float32 = np.float32(0.0)
        self.limitation_status_leaf: List[NDArray[Any]]
        self.limitation_status_root: List[NDArray[Any]]
        self.limitation_status_wood: List[NDArray[Any]]

        self.uptake_strategy_n: List[NDArray[Any]]
        self.uptake_strategy_p: List[NDArray[Any]]

        # annual sums
        self.anpp: np.float32 = np.float32(0.0)
        self.uptake_costs: np.float32 = np.float32(0.0)

        self.shannon_entropy: float = 0.0
        self.shannon_diversity: float = 0.0
        self.shannon_evenness: float = 0.0

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


    def update_lsid(self, occupation: NDArray[np.float64]) -> None:
        """Updates the internal community ids of the living PLSs.

        Args:
            occupation (np.ndarray):
        """
        self.vp_lsid = np.where(occupation > 0.0)[0]
        if len(self.vp_lsid) == 0:
            self.masked = np.int8(1)


    def restore_from_main_table(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """A call to the __init__ method.
        This is used to reset the community to a initial state with a new random sample of PLSs from the main table.

        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): a tuple of arrays with the ids and the PLSs.

        Returns:
            None: The community is reset to the initial state with a new sample of PLSs from the main table.
        """
        self.__init__(pls_data)


    def get_free_lsid(self) -> NDArray[np.intp]:
        """Get the indices of the free slots in the community.

        Returns:
            np.ndarray: _description_
        """
        # Get the indices of the free slots in the community
        return np.where(self.vp_ocp == 0.0)[0]


    def seed_pls(self, pls_id: int, pls: np.ndarray) -> None:
        """
        Seeds a PLS in a free position. Uses the method get_free_lsid to find the free slots.


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
        self.vp_cleaf[pos] = np.random.uniform(0.3,0.4)
        self.vp_croot[pos] = np.random.uniform(0.3,0.4)
        self.vp_cwood[pos] = np.random.uniform(5.0,6.0)
        self.vp_cwood[self.pls_array[6,:] == 0.0] = 0.0
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


    def get_unique_pls(self, pls_selector: Callable[[int], Tuple[int, np.ndarray]]) -> Tuple[int, np.ndarray]:
        """Gets a unique PLS from the main table using the provided callable.

        Args:
            pls_selector (callable): A function that returns a tuple (pls_id, pls).

        Returns:
            Tuple[int, np.ndarray]: A unique PLS ID and the PLS array.
        """
        while True:
            pls_id, pls = pls_selector(1)
            if pls_id not in self.id:
                return pls_id, pls