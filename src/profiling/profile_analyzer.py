# profile_analyzer.py
import pstats
import sys

def analyze_profile(profile_file):
    """Analyze cProfile output file"""
    
    print(f"\n{'='*60}")
    print(f"PROFILE ANALYSIS: {profile_file}")
    print(f"{'='*60}")
    
    stats = pstats.Stats(profile_file)
    
    # Top 20 functions by cumulative time
    print("\nTop 20 Functions by Cumulative Time:")
    print("-" * 50)
    stats.sort_stats('cumulative').print_stats(20)
    
    # Top 20 functions by total time (self time)
    print("\nTop 20 Functions by Total Time (Self):")
    print("-" * 50)
    stats.sort_stats('tottime').print_stats(20)
    
    # Functions that call specific modules (e.g., numpy, multiprocessing)
    print("\nNumPy/NumPy-related calls:")
    print("-" * 30)
    stats.print_stats('numpy')
    
    print("\nMultiprocessing-related calls:")
    print("-" * 30)
    stats.print_stats('multiprocessing')
    
    print("\nCAETE module calls:")
    print("-" * 30)
    stats.print_stats('caete')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_profile(sys.argv[1])
    else:
        # Analyze all profile files
        import glob
        profile_files = glob.glob("*.prof")
        for pf in profile_files:
            analyze_profile(pf)