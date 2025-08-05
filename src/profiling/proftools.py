
import cProfile
import pstats
import time
import tracemalloc
from pstats import SortKey
from functools import wraps


# Profiler decorator to use on specific methods
def profile_function(output_file=None):
    """Decorator for profiling individual functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate filename based on function if not provided
            fname = output_file or f"{func.__name__}_profile"
            print(f"\nProfiling function: {func.__name__}")

            # Start memory tracking
            tracemalloc.start()

            # Start timing
            start_time = time.time()

            # Run the function with profiling
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Get memory info
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Save and print results
            print(f"\nResults for {func.__name__}:")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            print(f"Memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")

            # Save profile data
            stats_file = f"{fname}.prof"
            profiler.dump_stats(stats_file)

            # Print profile summary
            stats = pstats.Stats(stats_file)
            stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)

            # Save stats for visualization
            stats.dump_stats(f"{fname}.pstats")

            # Try to generate visualization
            try:
                import gprof2dot
                import os
                print(f"Generating visualization in {fname}.png")
                os.system(f"gprof2dot -f pstats {fname}.pstats | dot -Tpng -o {fname}.png")
            except ImportError:
                print("For visualization, install: pip install gprof2dot")

            return result
        return wrapper
    return decorator

# Class profiler to use on entire class or module
class ProfilerManager:
    def __init__(self, name="caete_profiler"):
        self.name = name
        self.profiler = None
        self.memory_tracking = False
        self.start_time = None
        self.snapshots = []

    def start(self):
        """Start profiling session"""
        if self.profiler is None:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            print(f"Started profiling session: {self.name}")

            # Start memory tracking
            tracemalloc.start()
            self.memory_tracking = True

            # Start timer
            self.start_time = time.time()

    def take_snapshot(self, label="checkpoint"):
        """Take a memory and time snapshot during profiling"""
        if self.memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            elapsed = time.time() - self.start_time
            snapshot = {
                'label': label,
                'time': elapsed,
                'current_memory': current,
                'peak_memory': peak
            }
            self.snapshots.append(snapshot)
            print(f"Snapshot '{label}': {elapsed:.2f}s, {current/1024/1024:.1f}MB current, {peak/1024/1024:.1f}MB peak")

    def stop(self):
        """Stop profiling and show results"""
        if self.profiler is not None:
            self.profiler.disable()

            # Record final time
            total_time = time.time() - self.start_time

            # Get final memory stats
            if self.memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.memory_tracking = False
                print(f"Final memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")

            # Print time
            print(f"Total execution time: {total_time:.2f} seconds")

            # Save stats to file
            stats_file = f"{self.name}.prof"
            self.profiler.dump_stats(stats_file)

            # Print summary
            print(f"\nProfile results for {self.name}:")
            stats = pstats.Stats(stats_file)
            stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)

            # Generate detailed reports sorted by different metrics
            print("\nTop 10 functions by cumulative time:")
            stats.sort_stats(SortKey.CUMULATIVE).print_stats(10)

            print("\nTop 10 functions by total time:")
            stats.sort_stats(SortKey.TIME).print_stats(10)

            print("\nTop 10 function calls by frequency:")
            stats.sort_stats(SortKey.CALLS).print_stats(10)

            # Save for visualization
            stats.dump_stats(f"{self.name}.pstats")

            # Try to generate visualization
            try:
                import gprof2dot
                import os
                print(f"\nGenerating visualization in {self.name}.png")
                os.system(f"gprof2dot -f pstats {self.name}.pstats | dot -Tpng -o {self.name}.png")
            except ImportError:
                print("\nFor visualization, install gprof2dot and graphviz:")
                print("pip install gprof2dot")

            # Print memory snapshots if collected
            if self.snapshots:
                print("\nMemory and time snapshots:")
                for snap in self.snapshots:
                    print(f"  {snap['label']}: {snap['time']:.2f}s, " +
                            f"{snap['current_memory']/1024/1024:.1f}MB current, " +
                            f"{snap['peak_memory']/1024/1024:.1f}MB peak")

            self.profiler = None
            print("\nProfiling completed")
            return stats
        return None