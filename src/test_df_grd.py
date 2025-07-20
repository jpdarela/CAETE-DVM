from pathlib import Path
from worker import worker
from dataframes import gridded_data
import time

region = worker.load_state_zstd("./pan_amazon_hist_result.psz")
vr = ("npp", "gpp")

print("Aggregating region data...")
s1 = time.perf_counter()
a = gridded_data.aggregate_region_data(region, vr, (1,2))
s2 = time.perf_counter()
print(f"Aggregated data in {s2 - s1:.2f} seconds")
print()
print("Creating masked arrays...")
s3 = time.perf_counter()
b = gridded_data.create_masked_arrays(a)
s4 = time.perf_counter()
print(f"Created masked arrays in {s4 - s3:.2f} seconds")
print(f"Total time: {s4 - s1:.2f} seconds")