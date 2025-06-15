import time
from dataframes import output_manager


start = time.perf_counter()
output_manager.cities_output()
end = time.perf_counter()
print(f"Execution time: {end - start:.2f} seconds")