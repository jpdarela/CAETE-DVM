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

"""
GPU-accelerated functions using PyOpenCL
These functions provide GPU implementations of computational bottlenecks
identified in the CAETE model for significant performance improvements.


WARNING: THese functions are only faster when the data is large enough > 100k elements.
For small data sizes, the overhead of transferring data to the GPU can make them slower.

"""

import numpy as np
import pyopencl as cl
from numpy.typing import NDArray


class CAETEGPUAccelerator:
    """GPU accelerator class for CAETE model computations using PyOpenCL"""
    
    def __init__(self):
        """Initialize OpenCL context and compile kernels"""
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self._compile_kernels()
        
        # Pre-create kernels to avoid repeated retrieval warnings
        self.cw_mean_kernel = cl.Kernel(self.program, "cw_mean_kernel")
        self.masked_mean_kernel = cl.Kernel(self.program, "masked_mean_kernel") 
        self.masked_mean_2D_kernel = cl.Kernel(self.program, "masked_mean_2D_kernel")
        
    def _compile_kernels(self):
        """Compile OpenCL kernels for CAETE computations"""
        kernel_source = """
        __kernel void cw_mean_kernel(
            __global const double* ocp,
            __global const float* values,
            __global float* result,
            const int n_elements
        ) {
            int gid = get_global_id(0);
            if (gid == 0) {
                float sum = 0.0f;
                for (int i = 0; i < n_elements; i++) {
                    sum += (float)ocp[i] * values[i];
                }
                result[0] = sum;
            }
        }
        
        __kernel void masked_mean_kernel(
            __global const char* mask,
            __global const float* values,
            __global float* result,
            __global int* valid_count,
            const int n_elements
        ) {
            int gid = get_global_id(0);
            if (gid == 0) {
                float sum = 0.0f;
                int count = 0;
                
                for (int i = 0; i < n_elements; i++) {
                    if (mask[i] == 0) {
                        sum += values[i];
                        count++;
                    }
                }
                
                valid_count[0] = count;
                if (count > 0) {
                    result[0] = sum / (float)count;
                } else {
                    result[0] = NAN;
                }
            }
        }
        
        __kernel void masked_mean_2D_kernel(
            __global const char* mask,
            __global const float* values,
            __global float* result,
            __global int* valid_counts,
            const int n_elements,
            const int integrate_dim
        ) {
            int dim_id = get_global_id(0);
            
            if (dim_id < integrate_dim) {
                float sum = 0.0f;
                int count = 0;
                
                for (int i = 0; i < n_elements; i++) {
                    if (mask[i] == 0) {
                        sum += values[dim_id * n_elements + i];
                        count++;
                    }
                }
                
                valid_counts[dim_id] = count;
                if (count > 0) {
                    result[dim_id] = sum / (float)count;
                } else {
                    result[dim_id] = 0.0f;
                }
            }
        }
        """
        
        self.program = cl.Program(self.ctx, kernel_source).build()
        
    def cw_mean(self, ocp: NDArray[np.float64], values: NDArray[np.float32]) -> np.float32:
        """
        GPU implementation of community weighted mean calculation.
        
        Args:
            ocp: Array of area occupation coefficients (0 = empty, 1 = total dominance)
            values: Array of values to compute weighted mean for
            
        Returns:
            Community weighted mean as float32
        """
        n_elements = ocp.size
        
        # Create GPU buffers
        ocp_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ocp)
        values_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
        result_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=4)  # One float32
        
        # Execute kernel
        self.cw_mean_kernel(
            self.queue, (1,), None,
            ocp_buf, values_buf, result_buf, np.int32(n_elements)
        )
        
        # Read result
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, result_buf)
        
        return result[0]
    
    def masked_mean(self, mask: NDArray[np.int8], values: NDArray[np.float32]) -> float:
        """
        GPU implementation of masked mean calculation.
        
        Args:
            mask: Mask array (0 = valid, non-zero = masked)
            values: Array of values to compute mean for
            
        Returns:
            Mean of unmasked values, or NaN if no valid values
        """
        n_elements = mask.size
        
        # Convert mask to char for OpenCL compatibility
        mask_char = mask.astype(np.int8)
        
        # Create GPU buffers
        mask_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mask_char)
        values_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
        result_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=4)  # One float32
        count_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=4)   # One int32
        
        # Execute kernel
        self.masked_mean_kernel(
            self.queue, (1,), None,
            mask_buf, values_buf, result_buf, count_buf, np.int32(n_elements)
        )
        
        # Read result
        result = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, result_buf)
        
        return float(result[0])
    
    def masked_mean_2D(self, mask: NDArray[np.int8], values: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        GPU implementation of 2D masked mean calculation.
        
        Args:
            mask: Mask array (0 = valid, non-zero = masked)
            values: 2D array of values with shape (integrate_dim, n_elements)
            
        Returns:
            Array of means for each dimension, ignoring masked values
        """
        integrate_dim = values.shape[0]
        n_elements = values.shape[1]
        
        # Flatten values array for GPU processing (row-major order)
        values_flat = values.flatten()
        mask_char = mask.astype(np.int8)
        
        # Create GPU buffers
        mask_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mask_char)
        values_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values_flat)
        result_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=integrate_dim * 4)  # integrate_dim float32s
        count_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=integrate_dim * 4)   # integrate_dim int32s
        
        # Execute kernel with one work item per dimension
        self.masked_mean_2D_kernel(
            self.queue, (integrate_dim,), None,
            mask_buf, values_buf, result_buf, count_buf, 
            np.int32(n_elements), np.int32(integrate_dim)
        )
        
        # Read result
        result = np.empty(integrate_dim, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, result_buf)
        
        return result
        
    def __del__(self):
        """Cleanup OpenCL resources"""
        try:
            if hasattr(self, 'queue'):
                self.queue.finish()
        except:
            pass


# Global GPU accelerator instance (initialized on first use)
_gpu_accelerator = None

def get_gpu_accelerator():
    """Get or create the global GPU accelerator instance"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = CAETEGPUAccelerator()
    return _gpu_accelerator


# GPU-enabled wrapper functions that match the original API
def cw_mean_gpu(ocp: NDArray[np.float64], values: NDArray[np.float32]) -> np.float32:
    """
    GPU-accelerated community weighted mean calculation.
    
    Args:
        ocp: Array of area occupation coefficients (0 = empty, 1 = total dominance)
        values: Array of values to compute weighted mean for
        
    Returns:
        Community weighted mean as float32
    """
    gpu = get_gpu_accelerator()
    return gpu.cw_mean(ocp, values)


def masked_mean_gpu(mask: NDArray[np.int8], values: NDArray[np.float32]) -> float:
    """
    GPU-accelerated masked mean calculation.
    
    Args:
        mask: Mask array (0 = valid, non-zero = masked)
        values: Array of values to compute mean for
        
    Returns:
        Mean of unmasked values, or NaN if no valid values
    """
    gpu = get_gpu_accelerator()
    return gpu.masked_mean(mask, values)


def masked_mean_2D_gpu(mask: NDArray[np.int8], values: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    GPU-accelerated 2D masked mean calculation.
    
    Args:
        mask: Mask array (0 = valid, non-zero = masked)
        values: 2D array of values with shape (integrate_dim, n_elements)
        
    Returns:
        Array of means for each dimension, ignoring masked values
    """
    gpu = get_gpu_accelerator()
    return gpu.masked_mean_2D(mask, values)


# Wrapper functions with same names as JIT versions for easy replacement
def cw_mean(ocp: NDArray[np.float64], values: NDArray[np.float32]) -> np.float32:
    """GPU version of cw_mean - same signature as JIT version"""
    return cw_mean_gpu(ocp, values)

def masked_mean(mask: NDArray[np.int8], values: NDArray[np.float32]) -> float:
    """GPU version of masked_mean - same signature as JIT version"""
    return masked_mean_gpu(mask, values)

def masked_mean_2D(mask: NDArray[np.int8], values: NDArray[np.float32]) -> NDArray[np.float32]:
    """GPU version of masked_mean_2D - same signature as JIT version"""
    return masked_mean_2D_gpu(mask, values)
