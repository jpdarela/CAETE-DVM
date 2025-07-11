import os
import sys
import shutil
import glob
from pathlib import Path

"""Tool for managing runtime DLLs for Fortran extensions (Windows only).

This script provides functionality to:
1. Find and copy Intel Fortran runtime DLLs
2. Find and copy MSVC runtime DLLs that may be needed by Fortran extensions
3. Check if the caete_module can be imported
4. List all loaded DLLs in the current process
5. Analyze dependencies of the caete_module.pyd file

Run this script if you encounter errors like:

In [1]: import caete_module as caete
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
Cell In[1], line 1
----> 1 import caete_module as caete

ImportError: DLL load failed while importing caete_module: The specified module could not be found.

This typically indicates that runtime DLLs are not found in the expected locations.

Usage:
  python fortran_runtime_dlls.py                   # Bundle all runtime DLLs
  python fortran_runtime_dlls.py --no-msvc         # Bundle only Intel runtime DLLs
  python fortran_runtime_dlls.py --list-only       # List DLLs without copying them
  python fortran_runtime_dlls.py --check           # Check if caete_module can be imported
  python fortran_runtime_dlls.py --analyze         # Perform deep analysis of dependencies
  python fortran_runtime_dlls.py --loaded-dlls     # List all loaded DLLs in the process
"""



def find_intel_runtime():
    """Find Intel runtime DLLs"""
    default_paths = [r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest"] # Default Intel oneAPI compiler path

    default_env_vars = ['IFORT_COMPILER',
                        'IFORT_COMPILER24',
                        'IFORT_COMPILER25',
                        'FORTRAN_COMPILER',
                        'IFX_COMPILER',
                        'FORTRAN_RUNTIME',
                        'IFORT_COMPILER_DIR',
                        'IFORT_COMPILER_ROOT'] # Common environment variables for Intel compilers
    places = []
    for env_var in default_env_vars:
        for path in default_paths:
            if Path(path).exists():
                if envi := os.environ.get(env_var):
                    places.append((env_var, envi, path))
    # Check if any Intel runtime directories were found
    if not places:
        # If no environment variables found, check default installation path
        default_install_path = Path(r"C:\Program Files (x86)\Intel\oneAPI")
        if default_install_path.exists():
            # Search for the compiler directory
            compiler_dirs = list(default_install_path.glob('compiler/*'))
            if compiler_dirs:
                for compiler_dir in compiler_dirs:
                    places.append((compiler_dir.name, str(compiler_dir), str(compiler_dir)))

    assert places, "No Intel runtime directories found. Please check your installation."

    # Get the binary directory from the first found place
    oneapi_root = Path(places[0][1]) / 'bin'
    if not oneapi_root.exists():
        raise FileNotFoundError(f"Intel runtime directory not found: {oneapi_root}")
    print(f"Using Intel runtime from: {oneapi_root}")

    dll_patterns = [
        'libifcoremd.dll',
        'libmmd.dll',
        'svml_dispmd.dll',
        'libiomp5md.dll'  # OpenMP runtime
    ]

    search_paths = [
        f"{oneapi_root}",
    ]

    found_dlls = []
    for pattern in dll_patterns:
        for search_path in search_paths:
            matches = glob.glob(f"{search_path}/{pattern}")
            if matches:
                found_dlls.extend(matches)
                break

    return found_dlls

def find_msvc_runtime():
    """Find MSVC runtime DLLs needed by Fortran extensions"""
    try:
        # Try to use vs_finder if available
        from windows_utils import vs_finder
        msvc_paths = vs_finder.find_msvc_redist_paths()

        # Always add System32 as it typically has the redistributable DLLs
        if r"C:\Windows\System32" not in msvc_paths:
            msvc_paths.append(r"C:\Windows\System32")
    except ImportError:
        print("Warning: vs_finder module not found, falling back to basic search method")
        # Fallback to a minimal search if vs_finder is not available
        msvc_paths = [r"C:\Windows\System32"]

        # Add environment variable if exists
        if vcdir := os.environ.get('VCINSTALLDIR'):
            msvc_paths.append(vcdir)

    # Filter out None and non-existent paths
    msvc_paths = [path for path in msvc_paths if path and Path(path).exists()]

    if not msvc_paths:
        print("Warning: No MSVC runtime paths found.")
        return []

    # Print found paths for debugging
    print(f"Found {len(msvc_paths)} MSVC runtime paths:")
    for i, path in enumerate(msvc_paths):
        print(f"  {i+1}. {path}")

    dll_patterns = [
        'msvcp140.dll',       # C++ standard library
        'vcruntime140.dll',   # C runtime
        'vcruntime140_1.dll', # Additional C runtime (newer versions)
        'concrt140.dll',      # Concurrency runtime
        'msvcp140_1.dll',     # C++ standard library additional
        'msvcp140_2.dll',     # C++ standard library additional
        'msvcp140_atomic_wait.dll',  # Additional component for C++20 features
        'vcruntime140_thread.dll'    # Thread-related runtime
    ]

    found_dlls = []
    for pattern in dll_patterns:
        for search_path in msvc_paths:
            matches = glob.glob(f"{search_path}/{pattern}")
            if matches:
                found_dlls.extend(matches)
                break

    return found_dlls

def bundle_dlls(include_msvc=True):
    """Copy Intel and optionally MSVC runtime DLLs to current directory

    Args:
        include_msvc (bool): Whether to include MSVC runtime DLLs. Default: True
    """
    # Get Intel runtime DLLs
    intel_dlls = find_intel_runtime()
    if not intel_dlls:
        print("No Intel runtime DLLs found.")
        intel_dlls = []

    # Get MSVC runtime DLLs if requested
    msvc_dlls = []
    if include_msvc:
        msvc_dlls = find_msvc_runtime()
        if not msvc_dlls:
            print("No MSVC runtime DLLs found.")

    # Combine DLL lists
    all_dlls = intel_dlls + msvc_dlls
    if not all_dlls:
        print("No runtime DLLs found.")
        return

    current_dir = Path('.')
    copied_count = 0

    for dll in all_dlls:
        dest = current_dir / Path(dll).name
        # Don't copy if already exists
        if not dest.exists():
            shutil.copy2(dll, dest)
            print(f"Copied: {Path(dll).name}")
            copied_count += 1
        else:
            print(f"Already exists: {Path(dll).name}")

    print(f"Total: {copied_count} DLLs copied")

def check_loaded_dlls():
    """Check if specific DLLs are loaded in the current process

    This function is useful for debugging DLL loading issues.
    It's only available on Windows and requires the win32process module.
    """
    try:
        import win32process
        import win32api
    except ImportError:
        print("win32process module not available. Install pywin32 to use this function.")
        return {}

    # Get list of loaded modules in current process
    handle = win32api.GetCurrentProcess()
    modules = win32process.EnumProcessModules(handle)

    # Get the module filenames
    module_names = {}
    for module in modules:
        try:
            module_path = win32process.GetModuleFileNameEx(handle, module)
            module_name = Path(module_path).name.lower()
            module_names[module_name] = module_path
        except:
            pass

    # SWearch dll in the local directory
    local_dlls = glob.glob('./*.dll')
    for dll in local_dlls:
        dll_name = Path(dll).name.lower()
        # Check if the DLL is already in the loaded modules
        if dll_name not in module_names:
            module_names[dll_name] = dll

    return module_names

def check_caete_module_dlls():
    """Check if caete_module is loadable and which DLLs it requires

    This function attempts to import the caete_module and reports on its DLL dependencies
    """
    try:
        print("Attempting to import caete_module...")
        import caete_module as caete
        print("✓ caete_module imported successfully!")

        # Check loaded DLLs
        loaded_dlls = check_loaded_dlls()

        # Common DLLs we're interested in
        dll_patterns = [
            'libifcoremd.dll',
            'libmmd.dll',
            'svml_dispmd.dll',
            'libiomp5md.dll',
            'msvcp140.dll',
            'vcruntime140.dll',
            'vcruntime140_1.dll',
            'concrt140.dll'
        ]

        print("\nChecking for specific DLLs:")
        for pattern in dll_patterns:
            pattern = pattern.lower()
            if pattern in loaded_dlls:
                print(f"✓ {pattern} loaded from: {loaded_dlls[pattern]}")
            else:
                print(f"✗ {pattern} not loaded")

        return True

    except ImportError as e:
        print(f"✗ Failed to import caete_module: {e}")
        return False

def analyze_caete_module():
    """Analyze caete_module dependencies using dll_analyzer

    This function is only available on Windows.
    """
    if not sys.platform.startswith('win'):
        print("This function is only available on Windows.")
        return

    try:
        from windows_utils import dll_analyzer
    except ImportError:
        raise ImportError("dll_analyzer module not found. Please ensure it is installed or available in the path.")

    try:
        # Find the caete_module.pyd file
        import importlib.util

        # Try to find the caete_module
        try:
            import caete_module
            module_path = importlib.util.find_spec('caete_module').origin
        except ImportError:
            # If not found, try to find the .pyd file in current directory
            possible_paths = [
                './caete_module.cp315-win_amd64.pyd',
                './caete_module.cp314-win_amd64.pyd',
                './caete_module.cp313-win_amd64.pyd',
                './caete_module.cp312-win_amd64.pyd',
                './caete_module.cp311-win_amd64.pyd',
                './caete_module.cp310-win_amd64.pyd',
                './caete_module.cp39-win_amd64.pyd',
                './caete_module.cp38-win_amd64.pyd',
                './caete_module.pyd'
            ]

            module_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    module_path = os.path.abspath(path)
                    break

            if not module_path:
                print("Could not find caete_module.pyd. Make sure it's in the current directory.")
                return

        print(f"Found caete_module at: {module_path}")
        analysis = dll_analyzer.analyze_python_extension(module_path)
        dll_analyzer.print_dependency_analysis(analysis)

        # Return which DLLs are missing
        return analysis['missing_dependencies']

    except Exception as e:
        print(f"Error analyzing caete_module: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bundle runtime DLLs for Fortran extensions")
    parser.add_argument("--no-msvc", action="store_true", help="Don't include MSVC runtime DLLs")
    parser.add_argument("--list-only", action="store_true", help="Only list DLLs, don't copy them")
    parser.add_argument("--check", action="store_true", help="Check if caete_module can be imported and check DLL dependencies")
    parser.add_argument("--loaded-dlls", action="store_true", help="List all DLLs currently loaded in the process")
    parser.add_argument("--analyze", action="store_true", help="Perform deep analysis of caete_module dependencies")
    args = parser.parse_args()

    if args.analyze:
        # Deep analysis of caete_module dependencies
        missing_dlls = analyze_caete_module()

        if missing_dlls and len(missing_dlls) > 0:
            print("\nSome DLLs are missing. Try bundling them:")
            print("python fortran_runtime_dlls.py")
        else:
            print("\nAll dependencies appear to be satisfied!")

    elif args.check:
        # Check if caete_module can be imported
        check_caete_module_dlls()

    elif args.loaded_dlls:
        # List all loaded DLLs
        loaded = check_loaded_dlls()
        if loaded:
            print(f"\nLoaded DLLs ({len(loaded)}):")
            for name, path in sorted(loaded.items()):
                print(f"  {name}: {path}")
        else:
            print("Could not retrieve loaded DLLs")

    elif args.list_only:
        # Just list the DLLs that would be bundled
        intel_dlls = find_intel_runtime()
        print("\nIntel runtime DLLs found:")
        if intel_dlls:
            for dll in intel_dlls:
                print(f"  {Path(dll).name}")
        else:
            print("  None found")

        if not args.no_msvc:
            msvc_dlls = find_msvc_runtime()
            print("\nMSVC runtime DLLs found:")
            if msvc_dlls:
                for dll in msvc_dlls:
                    print(f"  {Path(dll).name}")
            else:
                print("  None found")
    else:
        # Bundle the DLLs
        bundle_dlls(include_msvc=not args.no_msvc)