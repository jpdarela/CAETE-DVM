# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False


# Tentative Cython code for community.py

import numpy as np
cimport numpy as cnp
from typing import Any, Callable, List, Tuple
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log
import cython

# Define numpy types for Cython
ctypedef cnp.float64_t DTYPE_f64_t
ctypedef cnp.float32_t DTYPE_f32_t
ctypedef cnp.int32_t DTYPE_i32_t
ctypedef cnp.intp_t DTYPE_intp_t

# Initialize numpy
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_f64_t, ndim=1] carea_frac(
    cnp.ndarray[DTYPE_f64_t, ndim=1] cleaf1,
    cnp.ndarray[DTYPE_f64_t, ndim=1] cfroot1,
    cnp.ndarray[DTYPE_f64_t, ndim=1] cawood1
):
    """Calculate the area fraction of each PFT based on the leaf, root and wood biomass."""
    cdef int npft = cleaf1.shape[0]
    cdef cnp.ndarray[DTYPE_f64_t, ndim=1] ocp_coeffs = np.zeros(npft, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_f64_t, ndim=1] total_biomass_pft = np.zeros(npft, dtype=np.float64)
    cdef double total_biomass = 0.0
    cdef int i
    
    # Compute total biomass for each PFT
    for i in range(npft):
        total_biomass_pft[i] = cleaf1[i] + cfroot1[i] + cawood1[i]
        total_biomass += total_biomass_pft[i]
    
    # Calculate occupation coefficients
    if total_biomass > 0.0:
        for i in range(npft):
            ocp_coeffs[i] = total_biomass_pft[i] / total_biomass
            if ocp_coeffs[i] < 0.0:
                ocp_coeffs[i] = 0.0
    
    return ocp_coeffs

cdef class community:
    """Represents a community of plants. Instances of this class are used to
       create metacommunities."""
    
    # Declare all attributes as cdef for performance
    cdef public cnp.ndarray vp_ocp
    cdef public cnp.ndarray id
    cdef public cnp.ndarray pls_array
    cdef public int npls
    cdef public tuple shape
    cdef public cnp.ndarray vp_cleaf
    cdef public cnp.ndarray vp_croot
    cdef public cnp.ndarray vp_cwood
    cdef public cnp.ndarray vp_sto
    cdef public cnp.ndarray vp_lsid
    cdef public int ls
    cdef public cnp.int8_t masked
    cdef public cnp.ndarray sp_uptk_costs
    cdef public cnp.ndarray construction_npp
    
    # Annual variables
    cdef public cnp.float32_t cleaf
    cdef public cnp.float32_t croot
    cdef public cnp.float32_t cwood
    cdef public cnp.float32_t csto
    cdef public list limitation_status_leaf
    cdef public list limitation_status_root
    cdef public list limitation_status_wood
    cdef public list uptake_strategy_n
    cdef public list uptake_strategy_p
    cdef public cnp.float32_t anpp
    cdef public cnp.float32_t uptake_costs
    cdef public double shannon_entropy
    cdef public double shannon_diversity
    cdef public double shannon_evenness

    def __cinit__(self, pls_data):
        # Initialize basic attributes
        pass

    def __init__(self, pls_data: Tuple[cnp.ndarray, cnp.ndarray]) -> None:
        self._reset(pls_data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _reset(self, tuple pls_data):
        """Reset the community to an initial state with a new random sample of PLSs from the main table."""
        
        cdef cnp.ndarray[DTYPE_i32_t, ndim=1] ids = pls_data[0]
        cdef cnp.ndarray[DTYPE_f32_t, ndim=2] pls_arr = pls_data[1]
        
        self.id = ids.copy()
        self.pls_array = pls_arr.copy()
        self.npls = self.pls_array.shape[1]
        # Fix: Build shape tuple manually from individual dimensions
        self.shape = (self.pls_array.shape[0], self.pls_array.shape[1])
    
        # Initialize biomass arrays using C random generation for better performance
        cdef int i
        cdef double rand_val
        
        # Allocate arrays
        self.vp_cleaf = np.empty(self.npls, dtype=np.float64)
        self.vp_croot = np.empty(self.npls, dtype=np.float64)
        self.vp_cwood = np.empty(self.npls, dtype=np.float64)
        
        # Fill with random values using C loops for better performance
        for i in range(self.npls):
            self.vp_cleaf[i] = 0.3 + (0.4 - 0.3) * (<double>rand() / RAND_MAX)
            self.vp_croot[i] = 0.3 + (0.4 - 0.3) * (<double>rand() / RAND_MAX)
            self.vp_cwood[i] = 5.0 + (6.0 - 5.0) * (<double>rand() / RAND_MAX)
        
        # Initialize storage array
        self.vp_sto = np.zeros((3, self.npls), dtype=np.float32, order='F')
        
        # Fill storage arrays with random values
        for i in range(self.npls):
            self.vp_sto[0, i] = 0.0 + (0.1 - 0.0) * (<double>rand() / RAND_MAX)
            self.vp_sto[1, i] = 0.0 + (0.01 - 0.0) * (<double>rand() / RAND_MAX)
            self.vp_sto[2, i] = 0.0 + (0.001 - 0.0) * (<double>rand() / RAND_MAX)
    
        # Set wood biomass to zero for non-woody plants
        for i in range(self.npls):
            if self.pls_array[6, i] == 0.0:
                self.vp_cwood[i] = 0.0
    
        # Calculate area fractions
        self.vp_ocp = carea_frac(self.vp_cleaf, self.vp_croot, self.vp_cwood)
    
        # Get living species indices
        self.vp_lsid = np.where(self.vp_ocp > 0.0)[0]
        self.ls = self.vp_lsid.shape[0]
        self.masked = <cnp.int8_t>0
    
        # Initialize other arrays
        self.sp_uptk_costs = np.zeros(self.npls, dtype=np.float32, order='F')
        self.construction_npp = np.zeros(self.npls, dtype=np.float32, order='F')
    
        # Initialize annual variables
        self.cleaf = <cnp.float32_t>0.0
        self.croot = <cnp.float32_t>0.0
        self.cwood = <cnp.float32_t>0.0
        self.csto = <cnp.float32_t>0.0
        
        self.anpp = <cnp.float32_t>0.0
        self.uptake_costs = <cnp.float32_t>0.0
    
        self.shannon_entropy = 0.0
        self.shannon_diversity = 0.0
        self.shannon_evenness = 0.0

    def __getitem__(self, int index):
        """Gets a PLS (1D array) for given index."""
        return self.pls_array[:, index]

    def __setitem__(self, int index, cnp.ndarray value):
        """Set a PLS at given index."""
        self.pls_array[:, index] = value

    def __len__(self):
        return self.shape[1]

    def __contains__(self, int pls_id):
        """Check if PLS ID is in the community."""
        cdef int i
        for i in range(self.npls):
            if self.id[i] == pls_id:
                return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_lsid(self, cnp.ndarray[DTYPE_f64_t, ndim=1] occupation):
        """Updates the internal community ids of the living PLSs."""
        self.vp_lsid = np.where(occupation > 0.0)[0]
        if self.vp_lsid.shape[0] == 0:
            self.masked = <cnp.int8_t>1

    def restore_from_main_table(self, pls_data: Tuple[cnp.ndarray, cnp.ndarray]) -> None:
        """Reset the community to initial state with new PLSs."""
        self._reset(pls_data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[DTYPE_intp_t, ndim=1] get_free_lsid(self):
        """Get the indices of the free slots in the community."""
        cdef set ids = set(range(self.npls))
        cdef set living_set = set(self.vp_lsid.tolist())
        cdef set free_slots = ids - living_set
        return np.array(list(free_slots), dtype=np.intp)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def seed_pls(self,
                 int pls_id,
                 cnp.ndarray pls,
                 cnp.ndarray[DTYPE_f64_t, ndim=1] cleaf,
                 cnp.ndarray[DTYPE_f64_t, ndim=1] croot,
                 cnp.ndarray[DTYPE_f64_t, ndim=1] cwood) -> None:
        """Seeds a PLS in a free position."""
        
        cdef cnp.ndarray[DTYPE_intp_t, ndim=1] free_slots = self.get_free_lsid()
        if free_slots.shape[0] == 0:
            return None
        
        cdef int pos
        if free_slots.shape[0] == 1:
            pos = free_slots[0]
        else:
            pos = free_slots[rand() % free_slots.shape[0]]

        self.id[pos] = pls_id
        self.pls_array[:, pos] = pls
        cleaf[pos] = 0.3 + (0.4 - 0.3) * (<double>rand() / RAND_MAX)
        croot[pos] = 0.3 + (0.4 - 0.3) * (<double>rand() / RAND_MAX)
        cwood[pos] = 5.0 + (6.0 - 5.0) * (<double>rand() / RAND_MAX)
        
        if pls[3] == 0.0:
            cwood[pos] = 0.0

    def get_unique_pls(self, pls_selector: Callable) -> Tuple[int, cnp.ndarray]:
        """Gets a PLS that is not present in the community."""
        while True:
            pls_id, pls = pls_selector(1)
            if pls_id not in self:
                return pls_id, pls
    
        # Add these methods to your community cdef class
    # Remove the _rebuild_community function completely
    
    # In your community class, replace the __reduce__ method with:
    
    def __reduce_ex__(self, protocol):
        """Custom pickle support for multiprocessing."""
        # Store the initialization data
        pls_data = (self.id, self.pls_array)
        
        # Store all the current state
        state_dict = {
            'vp_cleaf': self.vp_cleaf,
            'vp_croot': self.vp_croot,
            'vp_cwood': self.vp_cwood,
            'vp_sto': self.vp_sto,
            'vp_ocp': self.vp_ocp,
            'vp_lsid': self.vp_lsid,
            'sp_uptk_costs': self.sp_uptk_costs,
            'construction_npp': self.construction_npp,
            'cleaf': self.cleaf,
            'croot': self.croot,
            'cwood': self.cwood,
            'csto': self.csto,
            'anpp': self.anpp,
            'uptake_costs': self.uptake_costs,
            'shannon_entropy': self.shannon_entropy,
            'shannon_diversity': self.shannon_diversity,
            'shannon_evenness': self.shannon_evenness,
            'npls': self.npls,
            'shape': self.shape,
            'ls': self.ls,
            'masked': self.masked,
            'limitation_status_leaf': getattr(self, 'limitation_status_leaf', []),
            'limitation_status_root': getattr(self, 'limitation_status_root', []),
            'limitation_status_wood': getattr(self, 'limitation_status_wood', []),
            'uptake_strategy_n': getattr(self, 'uptake_strategy_n', []),
            'uptake_strategy_p': getattr(self, 'uptake_strategy_p', []),
        }
        
        return (self.__class__, (pls_data,), state_dict)
    
    def __setstate__(self, state_dict):
        """Restore object state."""
        for attr, value in state_dict.items():
            setattr(self, attr, value)
