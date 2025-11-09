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

from typing import Any, Callable, List, Tuple
import sys
import numpy as np
from numpy.typing import NDArray
from numba import njit

if sys.platform == "win32":
    from config import update_runtime_gcc_gfortran, update_runtime_oneapi
    update_runtime_gcc_gfortran()
    update_runtime_oneapi()

from caete_module import photo as m

UNIFORM_INIT_BIOMASS = False

@njit(cache=True)
def _update_living_status(occupation: NDArray[np.float64],
                         threshold: float = 0.0) -> Tuple[NDArray[np.intp], np.int8]:
    """Fast numba function to find living PLSs and determine if community is masked."""
    living_indices = np.flatnonzero(occupation > threshold)
    is_masked = np.int8(len(living_indices) == 0)
    return living_indices, is_masked


@njit(cache=True)
def carea_frac(cleaf1:NDArray[np.float64],
                  cfroot1:NDArray[np.float64],
                  cawood1:NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the area fraction of each PFT based on the leaf, root and wood biomass."""
    # Initialize variables
    npft = cleaf1.size
    ocp_coeffs = np.zeros(npft, dtype=np.float64)
    total_biomass_pft = np.zeros(npft,dtype=np.float64)
    # Compute total biomass for each PFT
    total_biomass_pft = cleaf1 + cfroot1 + cawood1
    # Compute total biomass for all PFTs
    total_biomass = np.sum(total_biomass_pft)
    # Calculate occupation coefficients
    if total_biomass > 0.0:
        ocp_coeffs = total_biomass_pft / total_biomass
        ocp_coeffs[ocp_coeffs < 0.0] = 0.0
    return ocp_coeffs

class community:
    """Represents a community of plants. Instances of this class are used to
       create metacommunities."""


    def _reset(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """Reset the community to an initial state with a new random sample of PLSs from the main table.

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
        # Add initial biomass values to the PLSs in the community using a fixed value plus some random noise
        if UNIFORM_INIT_BIOMASS:
            self.vp_cleaf: NDArray[np.float64] = np.random.uniform(self.bm_lr0, self.bm_lr0 + 0.1, self.npls).astype(np.float64)
            self.vp_croot: NDArray[np.float64] = np.random.uniform(self.bm_lr0, self.bm_lr0 + 0.2, self.npls).astype(np.float64)
            self.vp_cwood: NDArray[np.float64] = np.random.uniform(self.bm_w0,  self.bm_w0  + 0.1, self.npls).astype(np.float64)
        else:
            self.vp_cleaf: NDArray[np.float64] = np.zeros((self.npls,), dtype=np.float64)
            self.vp_croot: NDArray[np.float64] = np.zeros((self.npls,), dtype=np.float64)
            self.vp_cwood: NDArray[np.float64] = np.zeros((self.npls,), dtype=np.float64)
            for i in range(self.npls):
                cl, cr, cw = m.spinup3(self.npp_init, self.pls_array[2:8, i])
                # print(f"Initial biomass from spinup3 - cleaf: {cl}, croot: {cr}, cwood: {cw}")
                self.vp_cleaf[i] = cl
                self.vp_croot[i] = cr
                self.vp_cwood[i] = cw

        # Plant storage CNP initialized to small random values
        self.vp_sto: NDArray[np.float32] = np.zeros(shape=(3, self.npls), order='F', dtype=np.float32)
        self.vp_sto[0,:] = np.random.uniform(0.05, 0.1, self.npls)
        self.vp_sto[1,:] = np.random.uniform(0.005, 0.01, self.npls)
        self.vp_sto[2,:] = np.random.uniform(0.0005, 0.001, self.npls)

        # Set the wood biomass of the plants that are not woody to zero
        self.vp_cwood[self.pls_array[6,:] == 0.0] = 0.0

        # PFT_AREA_FRAC based on the mass-ratio hypothesis (Grime 1998)
        self.vp_ocp = carea_frac(self.vp_cleaf, self.vp_croot, self.vp_cwood)

        # Get the indices of the plants that are present in the community
        self.vp_lsid: NDArray[np.intp] = np.where(self.vp_ocp > 0.0)[0]
        self.ls: int = self.vp_lsid.size
        self.masked: np.int8 = np.int8(0)
        # These needs to be passed today from the previous timestep
        # nutrient Uptake costs that is subtracted from NPP
        self.sp_uptk_costs: NDArray[np.float32] = np.zeros(self.npls, order='F', dtype=np.float32)
        # Real NPP. I.e. NPP after nutrient uptake costs and allocation (incl. limitation)
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


    def __init__(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        #TODO: Move thhese initial biomass values to a configuration file (caete.toml) or parameterize a better way to set them.
        # These initial values set the initial conditions of the PLSs in the community.
        # They can have a significant impact on the model behavior during the first years of simulation.
        # They should be set based on empirical data or sensitivity analysis.

        # These values are currently hardcoded in that way to allow the model to run without crashing during the first years.
        # We set leaves and roots initial biomass to 0.125 kg m⁻² and wood to 0.01 kg m⁻².

        # This enable the PLSs to have enough biomass to survive the first years of simulation.
        # However, this is not a realistic representation of the initial biomass of PLSs in a community.
        #

        # One idea: use spinup3 and a NPP value based on remote sensing data (NTSG) to set these initial biomass values.
        # Call spinup3 with the allocation and residence time values of each PLS in the community to get a more realistic initial biomass.
        # Or Add cleaf cwood and croot initial biomass values based on the output of spinup3 in a lookup table during community initialization.
        # Use the table at runtime to set the initial biomass values of the PLSs in the community.
        self.bm_lr0 = 0.2 # Initial leaf and root biomass (kg m⁻²)
        self.bm_w0  = 0.2 # Initial wood biomass (kg m⁻²)
        self.npp_init = 0.1 # kg/m2/year - Initial NPP value to use in spinup3 to set initial biomass values.

        self._reset(pls_data)
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
        """Optimized version using numba and flatnonzero."""
        self.vp_lsid, self.masked = _update_living_status(occupation)

    # def update_lsid(self, occupation: NDArray[np.float64]) -> None:
    #     """Updates the internal community ids of the living PLSs.

    #     Args:
    #         occupation (np.ndarray):
    #     """
    #     self.vp_lsid = np.where(occupation > 0.0)[0]
    #     if len(self.vp_lsid) == 0:
    #         self.masked = np.int8(1)


    def restore_from_main_table(self, pls_data:Tuple[NDArray[np.int32], NDArray[np.float32]]) -> None:
        """Reset the community to a initial state with a new random sample of PLSs from the main table.

        Args:
            pls_data (Tuple[np.ndarray[int], np.ndarray[float]]): a tuple of arrays with the ids and the PLSs.

        Returns:
            None: The community is reset to the initial state with a new sample of PLSs from the main table.
        """
        self._reset(pls_data)


    def get_free_lsid(self) -> NDArray[np.intp]:
        """Get the indices of the free slots in the community.

        Returns:
            np.ndarray: _description_
        """
        # Get the indices of the free slots in the community
        ids = set(range(self.npls))
        free_slots = ids - set(self.vp_lsid)
        living = np.array(list(free_slots), dtype=np.intp)

        return living


    def seed_pls(self,
                 pls_id: int,
                 pls: np.ndarray,
                 cleaf: NDArray[np.float64],
                 croot: NDArray[np.float64],
                 cwood: NDArray[np.float64],
                 ) -> None:
        """
        Seeds a PLS in a free position. Uses the method get_free_lsid to find the free slots.

        Args:
            pls_id (int): ID of the PLS.
            pls (np.ndarray): Array representing the PLS.
            cleaf (NDArray[np.float64]): Leaf carbon allocation # Passed by reference. This will be updated.
            croot (NDArray[np.float64]): Root carbon allocation # Passed by reference. This will be updated.
            cwood (NDArray[np.float64]): Wood carbon allocation # Passed by reference. This will be updated.
        """
        # Assert that the PLS ID is not in the community
        free_slots = self.get_free_lsid()
        if free_slots.size == 0:
            return None
        elif free_slots.size == 1:
            pos = free_slots[0]
        else:
            pos = free_slots[np.random.randint(0, free_slots.size)] # Randomly select a free slot

        self.id[pos] = pls_id
        self.pls_array[:, pos] = pls
        cl, cr, cw = m.spinup3(self.npp_init, pls[2:8])
        cleaf[pos] = cl  
        croot[pos] = cr 
        cwood[pos] = cw 
        if pls[3] == 0:
            cwood[pos] = 0.0


    def get_unique_pls(self, pls_selector: Callable[[int], Tuple[int, np.ndarray]]) -> Tuple[int, np.ndarray]:
        """Gets a PLS that is not present in the community from the main table using the provided callable.

        Args:
            pls_selector (callable): A function that returns a tuple (pls_id, pls).

        Returns:
            Tuple[int, np.ndarray]: A unique PLS ID and the PLS array.
        """
        while True:
            pls_id, pls = pls_selector(1)
            if pls_id not in self.id:
                return pls_id, pls
