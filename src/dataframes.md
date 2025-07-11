# Performance Optimization Suggestions for CAETE-DVM

This document contains performance optimization suggestions for the CAETE-DVM model implementation, particularly focusing on the `dataframes.py` module.

## 1. Replace Pandas with Polars (TEXT OUTPUTS)

Polars is a high-performance DataFrame library that significantly outperforms pandas for many operations, especially with larger datasets.

```python
# Change this:
import pandas as pd
# to this:
import polars as pl
```

Example transformation for the `write_metacomm_output()` function:

```python
# Using Polars instead of Pandas
data = pl.DataFrame(data_dict)
count_series = data.group_by("pls_id").count()
df = (data.group_by("pls_id").mean()
      .with_columns([
          pl.col("pls_id").map_dict(dict(zip(count_series["pls_id"], count_series["count"]))).alias("count"),
          (pl.col("vp_cleaf") + pl.col("vp_croot") + pl.col("vp_cwood")).alias("cveg")
      ]))
```

Benefits:
- Much faster performance, particularly for larger datasets
- Lower memory usage
- Better support for parallel execution
- More efficient I/O operations

IMPLEMENTED: This change has been implemented in the dataframes.py module. 60% of reduction in execution time has been observed in tests.

## 2. Use Memory Mapping for Large Arrays (GRIDED OUTPUTS)

For operations involving very large arrays, consider using NumPy's memory mapping functionality, which allows you to work with large arrays without loading the entire array into memory.

```python
import numpy as np

# Replace large array operations with memory-mapped arrays
array = np.memmap('temp_file.dat', dtype='float64', mode='w+', shape=(region_height, region_width))
```

This can significantly reduce memory consumption when working with large datasets.

## 3. Optimize the `create_masked_arrays()` Method(GRIDED OUTPUTS)

The `create_masked_arrays()` method contains nested loops that can be optimized with vectorized operations:

```python
def create_masked_arrays(data: dict):
    # Pre-calculate coordinate transformations once
    y_rel = data["coord"][:, 0] - ymin
    x_rel = data["coord"][:, 1] - xmin

    # Filter valid points
    valid_mask = (0 <= y_rel) & (y_rel < region_height) & (0 <= x_rel) & (x_rel < region_width)
    valid_coords = np.column_stack((y_rel[valid_mask], x_rel[valid_mask]))
    valid_indices = np.where(valid_mask)[0]

    # Then process arrays with vectorized operations using the valid indices
    # ...
```

This avoids repeated coordinate calculations and boundary checking.

## 4. Use Numba for Computation-Heavy Functions(GRIDED OUTPUTS)

For computationally intensive functions, consider using Numba to JIT-compile Python code to optimized machine code:

```python
import numba

@numba.njit(parallel=True)
def process_data(arrays, data, coords):
    # Your computation code
    return result
```

Particularly useful for the array processing in `create_masked_arrays()` and data transformations in `make_daily_dataframe()`.

## 5. Improve ThreadPoolExecutor Usage

Optimize the thread pool executor by explicitly setting the number of workers:

```python
max_workers = min(os.cpu_count(), config.multiprocessing.nprocs)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(worker, range(periods))
```

Consider adding a `chunksize` parameter to the `map` call for better work distribution when processing many small tasks.

## 6. Use Chunking for Large I/O Operations

When reading or writing large files, process them in chunks rather than loading everything at once:

```python
def make_daily_dataframe(r, variables, spin_slice=None):
    # Process in chunks rather than loading all at once
    chunk_size = 1000  # Adjust based on memory constraints
    for i in range(0, len(time), chunk_size):
        chunk_time = time[i:i+chunk_size]
        # Process chunk
```

This can significantly reduce memory usage when working with large time series.

## 7. Standardize on a Single Parallelization Approach

The code currently uses both `joblib.Parallel` and `concurrent.futures.ThreadPoolExecutor`. Standardize on one approach for better consistency and maintenance:

```python
# Replace Parallel with ThreadPoolExecutor in cities_output
with ThreadPoolExecutor(max_workers=min(len(results), os.cpu_count())) as executor:
    list(executor.map(output_manager.table_output_per_grd, results, [variables]*len(results)))
```

Alternatively, if sticking with `joblib.Parallel`, use it consistently throughout:

```python
Parallel(n_jobs=min(periods, os.cpu_count()))(
    delayed(worker)(i) for i in range(periods)
)
```

## 8. Use HDF5 Instead of CSV for Large Outputs

For large outputs, consider using HDF5 format instead of CSV, which is more efficient for large datasets:

```python
import h5py

# Instead of many CSV files
with h5py.File(f"output_data_{grd.y}_{grd.x}.h5", "w") as f:
    f.create_dataset("data", data=df_values)
    f.create_dataset("time", data=np.array(time, dtype="S10"))
```

Benefits include:
- Faster I/O operations
- Built-in compression
- Better handling of hierarchical data
- Support for partial reads

## 9. Cache Frequently Used Data

For repeated calculations, implement caching to avoid redundant computations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_coordinate_mapping(region_bounds):
    # Calculate coordinate mapping once
    return mapping
```

## 10. Optimize DataFrame Operations

Optimize pandas DataFrame operations by:

- Using `.loc` instead of chained indexing
- Pre-allocating DataFrames of the correct size when possible
- Using inplace operations when appropriate
- Avoiding unnecessary copies
- Using vectorized operations instead of loops

Example in `write_metacomm_output`:

```python
# Optimize DataFrame creation
df = pd.DataFrame(data_dict)
# Use vectorized operations
df["cveg"] = df["vp_cleaf"] + df["vp_croot"] + df["vp_cwood"]
```

## 11. Consider Using Dask for Large-Scale Parallelism

For processing very large datasets across many cores or machines, consider using Dask:

```python
import dask.dataframe as dd

# Convert pandas DataFrame to Dask DataFrame for parallel processing
dask_df = dd.from_pandas(df, npartitions=os.cpu_count())
result = dask_df.map_partitions(process_function).compute()
```

## 12. Profile-Guided Optimizations

Use the enhanced profiling capabilities to identify the most time-consuming parts of the `table_data` and `output_manager` classes. The new profiling system provides detailed insights into execution time, memory usage, and call patterns.

### 12.1 Using the New Profiling System

You can use the new profiling system in several ways:

#### Option 1: From the Command Line

Use the enhanced command line interface to profile specific components:

```bash
# Profile cities_output method
python dataframes.py --profile cities

# Profile a specific table_data method
python dataframes.py --profile table --method make_daily_df

# Profile both components with memory tracing
python dataframes.py --profile both --trace-memory
```

#### Option 2: Using the Dedicated Profiling Script

For more detailed analysis, use the dedicated profiling script:

```bash
# Profile the cities_output method with small dataset
python profile_dataframes.py --test cities_output --size small

# Profile all components with medium-sized dataset
python profile_dataframes.py --test all --size medium

# Run a specific test with visualization
python profile_dataframes.py --test write_daily_data --size small --visualize
```

#### Option 3: Using the Profiler Decorator

Apply the profiler decorator directly to functions you want to profile:

```python
from dataframes import profile_function

@profile_function("my_function_profile")
def my_function():
    # Code to profile
    pass
```

### 12.2 Profiling the `table_data` Class

For the `table_data` class, key areas to profile include:

- `make_daily_dataframe`: Analyze memory usage and I/O performance
- `write_metacomm_output`: Check for optimization opportunities in DataFrame operations
- `_process_year_data`: Look for potential parallelization opportunities

### 12.3 Profiling the `output_manager` Class

For the `output_manager` class, focus on:

- `table_output_per_grd`: Optimize the processing of grid data
- `cities_output`: Improve parallelism and resource allocation

### 12.4 Memory Profiling

The new profiling system also tracks memory usage, allowing you to identify memory leaks or inefficient memory usage patterns:

```python
profiler = ProfilerManager("memory_test")
profiler.start()
profiler.take_snapshot("before_processing")
# Run your code
profiler.take_snapshot("after_processing")
profiler.stop()
```

### 12.5 Analyzing Profile Results

When analyzing profile results, focus on:

1. **Hotspots**: Functions that consume the most time
2. **Frequency**: Functions called most often
3. **Memory**: Operations causing memory spikes
4. **I/O Operations**: File read/write operations that may be optimized

### 12.6 Implementing Optimizations

After identifying bottlenecks, implement targeted optimizations:

- **Vectorization**: Replace loops with NumPy vectorized operations
- **Parallelization**: Apply ThreadPoolExecutor with optimal worker counts
- **Memory Management**: Use generators or iterators for large data sets
- **I/O Optimization**: Use chunked reading/writing and consider alternate formats

Remember to measure performance before and after each optimization to ensure improvements are realized.

## 13. Scalability Concerns (TEXT OUTPUTS)

When scaling the `table_data` class to handle thousands of gridcells, several key concerns need to be addressed:

### 13.1 Memory Management

- **Problem**: In `make_daily_dataframe()`, each gridcell's data is fully loaded into memory before processing, which could lead to significant memory pressure with thousands of gridcells.
- **Solution**: Implement a streaming approach that processes gridcells in batches, with explicit memory cleanup between batches.

```python
def process_gridcells_in_batches(r, variables, batch_size=100):
    total = len(r)
    for i in range(0, total, batch_size):
        batch = r[i:min(i+batch_size, total)]
        process_batch(batch, variables)
        # Force garbage collection
        gc.collect()
```

### 13.2 I/O Bottlenecks

- **Problem**: Creating individual CSV files for each gridcell (potentially thousands of files) can overwhelm file systems and create I/O bottlenecks.
- **Solution**: Implement hierarchical storage or database solutions:

```python
# Option 1: Hierarchical directories
output_dir = Path(f"output/batch_{batch_id}")
output_dir.mkdir(exist_ok=True, parents=True)

# Option 2: Use HDF5 with groups
with h5py.File("gridcell_data.h5", "a") as f:
    grp = f.create_group(f"gridcell_{grd.y}_{grd.x}")
    grp.create_dataset("data", data=array_data, compression="gzip")
```

### 13.3 Workload Distribution

- **Problem**: The current parallelization doesn't account for variable workloads between gridcells, leading to inefficient resource utilization.
- **Solution**: Implement dynamic load balancing:

```python
def process_with_load_balancing(gridcells):
    # Sort gridcells by estimated workload (e.g., data size)
    sorted_cells = sorted(gridcells, key=estimate_workload, reverse=True)

    # Use ProcessPoolExecutor with task queue for dynamic distribution
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_gridcell, cell): cell for cell in sorted_cells}
        for future in as_completed(futures):
            # Process results as they complete
            cell = futures[future]
            try:
                result = future.result()
                # Handle result
            except Exception as e:
                # Handle exception
```

### 13.4 Data Aggregation

- **Problem**: Analyzing data across thousands of gridcells becomes challenging when stored in separate files.
- **Solution**: Implement efficient aggregation strategies:

```python
def aggregate_gridcell_data(output_path, variables):
    # Use dask for out-of-core data processing
    import dask.dataframe as dd

    # Create pattern for globbing files
    pattern = str(output_path / "grd_*_*.csv")

    # Read all matching files into a dask dataframe
    ddf = dd.read_csv(pattern)

    # Perform aggregation operations
    result = ddf.groupby(['day']).mean().compute()

    # Save aggregated results
    result.to_csv(output_path / "aggregated_results.csv")
```

### 13.5 Progress Monitoring and Fault Tolerance

- **Problem**: With thousands of gridcells, processing can take considerable time with no visibility into progress.
- **Solution**: Implement robust progress tracking and fault tolerance:

```python
def process_with_checkpointing(gridcells, checkpoint_interval=50):
    # Track progress
    progress = tqdm(total=len(gridcells))
    results = []
    failures = []

    for i, cell in enumerate(gridcells):
        try:
            result = process_gridcell(cell)
            results.append(result)
        except Exception as e:
            failures.append((cell, str(e)))

        # Update progress
        progress.update(1)

        # Save checkpoint periodically
        if i % checkpoint_interval == 0:
            save_checkpoint(results, failures, i)

    return results, failures
```

By addressing these concerns, the `table_data` class can be optimized to scale effectively to thousands of gridcells, maintaining performance while managing resources efficiently.
