#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiling script for CAETE-DVM dataframes.py module

This script provides detailed profiling for the table_data and output_manager classes.
It helps identify bottlenecks and potential optimization targets in the data processing
pipeline.

Usage:
    python profile_dataframes.py --profile cities
    python profile_dataframes.py --profile table --method make_daily_df
    python profile_dataframes.py --profile both --trace-memory
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import the module to profile
from dataframes import table_data, output_manager, worker

def visualize_profile_results(profile_file, title="Function Call Distribution"):
    """Generate a visualization of profiling results"""
    try:
        import pstats
        from pstats import SortKey

        # Load stats
        p = pstats.Stats(profile_file)
        stats = p.get_stats_profile().func_profiles

        # Extract data for top functions
        func_names = []
        times = []
        calls = []

        # Get top 10 functions by cumulative time
        sorted_funcs = sorted(stats.items(), key=lambda x: x[1].cumulative, reverse=True)
        for func, profile in sorted_funcs[:10]:
            if hasattr(func, 'co_name'):
                name = f"{func.co_name} ({func.co_filename.split('/')[-1]}:{func.co_firstlineno})"
            else:
                name = str(func)

            # Truncate long names
            if len(name) > 40:
                name = name[:37] + "..."

            func_names.append(name)
            times.append(profile.cumulative)
            calls.append(profile.ncalls)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot cumulative time
        y_pos = np.arange(len(func_names))
        ax1.barh(y_pos, times, align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(func_names)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel('Cumulative Time (seconds)')
        ax1.set_title('Top Functions by Cumulative Time')
        ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add time values as text
        for i, v in enumerate(times):
            ax1.text(v + 0.01, i, f"{v:.3f}s", va='center')

        # Plot call counts
        ax2.barh(y_pos, calls, align='center', color='green')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(func_names)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel('Number of Calls')
        ax2.set_title('Call Counts for Top Functions')
        ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add call count values as text
        for i, v in enumerate(calls):
            ax2.text(v + 1, i, f"{v}", va='center')

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save the figure
        output_path = profile_file.replace('.prof', '.png')
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()

    except ImportError:
        print("Matplotlib is required for visualization. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating visualization: {e}")

def run_profiling_test(test_name, test_size="small"):
    """Run a specific profiling test with given parameters"""

    # Define test parameters based on size
    if test_size == "small":
        # Small test with limited data
        variables = ("cue", "wue", "csoil", "hresp")
        results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    elif test_size == "medium":
        # Medium-sized test with more variables
        variables = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp", "photo", "npp", "evapm")
        results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    else:  # large
        # Large test with many variables
        variables = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp", "photo", "npp",
                   "evapm", "lai", "f5", "wsoil", "pupt", "nupt", "ls", "c_cost",
                   "rcm", "storage_pool", "inorg_n", "inorg_p", "snc", "vcmax")
        results = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")

    # Make sure the result file exists
    if not results.exists():
        # Fall back to a different file that might exist
        print(f"Warning: {results} not found. Looking for alternatives...")
        alt_files = ["./cities_MPI-ESM1-2-HR-ssp370_output.psz",
                    "./cities_MPI-ESM1-2-HR-ssp585_output.psz",
                    "./pan_amazon_hist_result.psz"]

        for alt in alt_files:
            if Path(alt).exists():
                results = Path(alt)
                print(f"Using alternative file: {alt}")
                break
        else:
            print("No valid result files found. Aborting test.")
            return

    # Load region data
    reg = worker.load_state_zstd(results)

    # Run the requested test
    if test_name == "write_daily_data":
        # Profile the write_daily_data method
        from dataframes import profile_function

        @profile_function(f"write_daily_data_{test_size}")
        def run_test():
            table_data.write_daily_data(reg, variables)

        run_test()

    elif test_name == "write_metacomm_output":
        # Profile write_metacomm_output
        from dataframes import profile_function

        @profile_function(f"write_metacomm_output_{test_size}")
        def run_test():
            for grd in reg[:5]:  # Limit to first 5 grids for testing
                table_data.write_metacomm_output(grd)

        run_test()

    elif test_name == "make_daily_dataframe":
        # Profile make_daily_dataframe
        from dataframes import profile_function

        @profile_function(f"make_daily_dataframe_{test_size}")
        def run_test():
            table_data.make_daily_dataframe(reg, variables, 1)

        run_test()

    elif test_name == "cities_output":
        # Profile the full cities_output method
        from dataframes import ProfilerManager

        profiler = ProfilerManager(f"cities_output_{test_size}")
        profiler.start()

        # Take memory snapshot before processing
        profiler.take_snapshot("before_processing")

        # For testing, we can limit which outputs are processed
        # to avoid long run times
        if test_size == "small":
            # Only process one file for small test
            results = [Path("./cities_MPI-ESM1-2-HR_hist_output.psz")]
            output_manager.table_output_per_grd(results[0], variables)
        else:
            # Process normally for medium/large tests
            output_manager.cities_output()

        # Take memory snapshot after processing
        profiler.take_snapshot("after_processing")

        # Stop profiling
        stats = profiler.stop()

    # Visualize the results if matplotlib is available
    try:
        profile_file = f"{test_name}_{test_size}.prof"
        if Path(profile_file).exists():
            visualize_profile_results(profile_file, f"{test_name} ({test_size} dataset)")
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CAETE-DVM dataframes.py module")
    parser.add_argument("--test", choices=["write_daily_data", "write_metacomm_output",
                                         "make_daily_dataframe", "cities_output", "all"],
                      default="cities_output", help="Test to run")
    parser.add_argument("--size", choices=["small", "medium", "large"],
                      default="small", help="Test size/complexity")
    parser.add_argument("--visualize", action="store_true",
                      help="Create visualization of results")

    args = parser.parse_args()

    if args.test == "all":
        print("Running all profiling tests...")
        tests = ["write_daily_data", "write_metacomm_output",
                "make_daily_dataframe", "cities_output"]
        for test in tests:
            print(f"\n--- Running test: {test} ({args.size}) ---")
            run_profiling_test(test, args.size)
    else:
        print(f"Running profiling test: {args.test} ({args.size})")
        run_profiling_test(args.test, args.size)

    print("\nProfiling complete.")
