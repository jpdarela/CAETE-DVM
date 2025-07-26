# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
MPI-enabled CAETÊ driver script

Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Usage:
    Run with MPI: mpirun -n <num_processes> python caete_driver_mpi.py
    
    Example:
    mpirun -n 4 python caete_driver_mpi.py
    mpirun -n 8 --bind-to core python caete_driver_mpi.py
    
    For SLURM clusters:
    srun -n 16 python caete_driver_mpi.py
"""

import sys
import time
from pathlib import Path

# Import MPI at the top level
try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py not found. Please install with: pip install mpi4py")
    sys.exit(1)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def mpi_print(message: str, rank_filter: int = 0):
    """Print message only from specified rank (default: rank 0)"""
    if rank == rank_filter:
        print(f"[Rank {rank}] {message}")

def mpi_print_all(message: str):
    """Print message from all ranks with rank prefix"""
    print(f"[Rank {rank}/{size}] {message}")

def main():
    """Main MPI driver function"""
    
    # Initialize timing (all ranks)
    time_start = time.time()
    
    # Check MPI environment
    if rank == 0:
        print(f"Starting CAETÊ MPI driver with {size} processes")
        print(f"MPI Version: {MPI.Get_version()}")
    
    try:
        # Import all necessary modules
        from metacommunity import pls_table
        from parameters import tsoil, ssoil, hsoil
        from region_mpi import region_mpi, check_mpi_environment
        from worker import worker
        
        # Validate MPI environment
        mpi_success, _, _ = check_mpi_environment()
        if not mpi_success:
            mpi_print("MPI environment validation failed!")
            comm.Abort(1)
            return
        
        # Create worker instance (all ranks need this)
        fn: worker = worker()
        
        # Configuration parameters (same for all ranks)
        region_name = "pan_amazon_hist_mpi"  # Name with MPI suffix
        
        # Input file paths
        obsclim_files = "../input/20CRv3-ERA5/obsclim_test/"
        spinclim_files = "../input/20CRv3-ERA5/spinclim_test/"
        transclim_files = "../input/20CRv3-ERA5/transclim_test/"
        counterclim_files = "../input/20CRv3-ERA5/counterclim_test/"
        
        # Soil hydraulic parameters
        soil_tuple = tsoil, ssoil, hsoil
        
        # CO2 data path
        co2_path = Path("../input/co2/historical_CO2_annual_1765-2024.csv")
        
        # Read PLS table (each rank reads independently to avoid communication overhead)
        mpi_print("Loading PLS table...")
        main_table = pls_table.read_pls_table(Path("./PLS_MAIN/pls_attrs-9999.csv"))
        mpi_print(f"PLS table loaded: {main_table.shape if hasattr(main_table, 'shape') else 'unknown shape'}")
        
        # Create the region using MPI (all ranks participate)
        mpi_print("Creating MPI region with spinup climate files...")
        r = region_mpi(region_name,
                      spinclim_files,
                      soil_tuple,
                      co2_path,
                      main_table)
        
        # Synchronize all processes before starting
        comm.Barrier()
        mpi_print("Region created successfully")
        
        # ===== SPINUP PHASE =====
        mpi_print("Starting soil pools spinup phase...")
        spinup_start = time.time()
        
        r.run_region_map(fn.spinup)
        
        spinup_time = time.time() - spinup_start
        mpi_print(f"Spinup completed in {spinup_time/60:.2f} minutes")
        
        # ===== TRANSCLIM PHASE =====
        mpi_print("Starting transclim run (1851-1900)...")
        transclim_start = time.time()
        
        r.update_input(transclim_files)
        r.run_region_map(fn.transclim_run)
        
        transclim_time = time.time() - transclim_start
        mpi_print(f"Transclim run completed in {transclim_time/60:.2f} minutes")
        
        # ===== SAVE INTERMEDIATE STATE =====
        if rank == 0:
            state_file = Path(f"./{region_name}_after_spinup_state_file.psz")
            print(f"Saving intermediate state file as {state_file}")
        
        # Only rank 0 handles state saving to avoid conflicts
        r.save_state(Path(f"./{region_name}_after_spinup_state_file.psz"))
        r.set_new_state()
        
        mpi_print("Intermediate state saved successfully")
        
        # ===== TRANSIENT RUN PHASE =====
        mpi_print("Starting transient run (1901-1905)...")  # Very short test period
        transient_start = time.time()
        
        r.update_input(obsclim_files)
        
        # Create run breaks for the transient period
        # Use very short periods for test data to avoid date range issues
        run_breaks = fn.create_run_breaks(1901, 1905, 2)  # Very short test period
        mpi_print(f"Created {len(run_breaks)} run periods")
        
        # Process each period with error handling
        successful_periods = 0
        for i, period in enumerate(run_breaks):
            period_start = time.time()
            mpi_print(f"Running period {i+1}/{len(run_breaks)}: {period[0]} - {period[1]}")
            
            try:
                r.run_region_starmap(fn.transient_run_brk, period)
                successful_periods += 1
                
                period_time = time.time() - period_start
                mpi_print(f"Period {period[0]}-{period[1]} completed in {period_time/60:.2f} minutes")
                
            except Exception as e:
                mpi_print(f"Error in period {period[0]}-{period[1]}: {e}")
                if rank == 0:
                    print(f"Skipping failed period and continuing...")
                continue
            
            # Periodic progress report
            if rank == 0 and (i + 1) % 3 == 0:  # Report every 3 periods instead of 5
                elapsed = (time.time() - transient_start) / 60
                remaining_periods = len(run_breaks) - (i + 1)
                estimated_remaining = (elapsed / (i + 1)) * remaining_periods if i > 0 else 0
                print(f"Progress: {i+1}/{len(run_breaks)} periods completed. "
                      f"Elapsed: {elapsed:.1f}min, Est. remaining: {estimated_remaining:.1f}min")
        
        transient_time = time.time() - transient_start
        mpi_print(f"Transient run completed: {successful_periods}/{len(run_breaks)} periods successful in {transient_time/60:.2f} minutes")
        
        # ===== FINAL STATE PROCESSING =====
        mpi_print("Saving final states...")
        
        # Save final state with model data
        final_period = run_breaks[-1] if run_breaks else [2021, 2021]
        r.save_state(Path(f"./{region_name}_{final_period[1]}_final_state.psz"))
        r.set_new_state()
        
        # Clean model state for output processing
        mpi_print("Cleaning model state for output...")
        r.clean_model_state()
        
        # Save cleaned result state
        r.save_state(Path(f"./{region_name}_result.psz"))
        
        # ===== PERFORMANCE STATISTICS =====
        total_time = time.time() - time_start
        
        if rank == 0:
            # Collect and display performance statistics
            try:
                perf_stats = r.get_performance_stats()
                print("\n" + "="*60)
                print("CAETÊ MPI EXECUTION COMPLETED")
                print("="*60)
                print(f"Total execution time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
                print(f"Spinup time: {spinup_time/60:.2f} minutes")
                print(f"Transclim time: {transclim_time/60:.2f} minutes") 
                print(f"Transient time: {transient_time/60:.2f} minutes")
                print(f"MPI processes used: {size}")
                
                if isinstance(perf_stats, dict):
                    print(f"Region size: {perf_stats.get('region_size', 'unknown')}")
                    print(f"Load balance ratio: {perf_stats.get('load_balance', 'unknown'):.2f}")
                    if 'files_per_rank' in perf_stats:
                        files_per_rank = perf_stats['files_per_rank']
                        print(f"Files per rank: min={min(files_per_rank)}, max={max(files_per_rank)}, avg={sum(files_per_rank)/len(files_per_rank):.1f}")
                
                print(f"Average time per process: {total_time/(size*60):.2f} minutes")
                print(f"Parallel efficiency: {(total_time/size)/total_time*100:.1f}%")
                print("="*60)
                
                # Save execution log
                with open(f"{region_name}_execution_log.txt", "w") as log_file:
                    log_file.write(f"CAETÊ MPI Execution Log\n")
                    log_file.write(f"Region: {region_name}\n")
                    log_file.write(f"MPI processes: {size}\n")
                    log_file.write(f"Total time: {total_time/60:.2f} minutes\n")
                    log_file.write(f"Spinup: {spinup_time/60:.2f} min\n")
                    log_file.write(f"Transclim: {transclim_time/60:.2f} min\n")
                    log_file.write(f"Transient: {transient_time/60:.2f} min\n")
                    if isinstance(perf_stats, dict):
                        log_file.write(f"Performance stats: {perf_stats}\n")
                
            except Exception as e:
                print(f"Error collecting performance statistics: {e}")
                print(f"Total execution time: {total_time/60:.2f} minutes")
        
        # Final synchronization
        comm.Barrier()
        mpi_print("CAETÊ MPI execution completed successfully")
        
    except KeyboardInterrupt:
        mpi_print_all("Execution interrupted by user")
        comm.Abort(1)
        
    except Exception as e:
        mpi_print_all(f"Error during execution: {e}")
        import traceback
        if rank == 0:
            print("Full traceback:")
            traceback.print_exc()
        comm.Abort(1)

if __name__ == "__main__":
    # Ensure proper MPI execution context
    if rank == 0:
        print("CAETÊ MPI Driver Starting...")
        print("="*50)
        print("This script should be run with MPI:")
        print("  mpirun -n <processes> python caete_driver_mpi.py")
        print("  or")
        print("  srun -n <processes> python caete_driver_mpi.py")
        print("="*50)
    
    # Run main function
    main()
    
    # Ensure clean MPI finalization
    if rank == 0:
        print("MPI driver finished.")
