"""
VS Installation Finder for Windows

This module provides functions to find Visual Studio installations and their components.
"""

import os
import glob
import subprocess
import json

def find_vs_installations():
    """
    Find all Visual Studio installations using vswhere if available

    Returns:
        list: List of paths to Visual Studio installations
    """
    vswhere_paths = [
            # Standard VS installer locations
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
            r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",

            # Chocolatey installation
            r"C:\ProgramData\chocolatey\bin\vswhere.exe",

            # Build Tools installations
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\vswhere.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\vswhere.exe",

            # Older Visual Studio versions
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vswhere.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\vswhere.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\Tools\vswhere.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\Common7\Tools\vswhere.exe",

            # Scoop installation paths
            r"C:\Users\{}\scoop\shims\vswhere.exe".format(os.environ.get('USERNAME', '')),
            r"C:\ProgramData\scoop\shims\vswhere.exe"
        ]

    vswhere_path = None
    for path in vswhere_paths:
        if os.path.exists(path):
            vswhere_path = path
            break

    installations = []

    if vswhere_path:
        try:
            # Run vswhere to find all VS installations
            cmd = [vswhere_path, "-all", "-products", "*", "-format", "json"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                try:
                    vs_instances = json.loads(result.stdout)
                    for instance in vs_instances:
                        if "installationPath" in instance:
                            installations.append(instance["installationPath"])
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error using vswhere: {e}")

    # Fallback to common installation paths if vswhere doesn't work
    if not installations:
        base_dirs = [
            r"C:\Program Files\Microsoft Visual Studio",
            r"C:\Program Files (x86)\Microsoft Visual Studio"
        ]

        editions = ["Community", "Professional", "Enterprise", "BuildTools"]
        years = ["2022", "2019", "2017", "2015"]

        for base in base_dirs:
            if os.path.exists(base):
                for year in years:
                    for edition in editions:
                        path = os.path.join(base, year, edition)
                        if os.path.exists(path):
                            installations.append(path)

    return installations

def find_msvc_redist_paths():
    """
    Find all MSVC redistributable paths

    Returns:
        list: List of paths to MSVC redistributable directories
    """
    redist_paths = []


    # Find redist paths in all VS installations
    vs_installations = find_vs_installations()
    for vs_path in vs_installations:
        # Check for Redist directory
        redist_base = os.path.join(vs_path, "VC", "Redist", "MSVC")
        if os.path.exists(redist_base):
            # Look for version-specific directories
            version_dirs = glob.glob(os.path.join(redist_base, "[0-9]*"))
            for version_dir in version_dirs:
                # Check for x64 CRT directories
                for crt_name in ["Microsoft.VC143.CRT", "Microsoft.VC142.CRT", "Microsoft.VC141.CRT"]:
                    crt_path = os.path.join(version_dir, "x64", crt_name)
                    if os.path.exists(crt_path):
                        redist_paths.append(crt_path)

            # Check for 'latest' symbolic link
            latest_path = os.path.join(redist_base, "latest")
            if os.path.exists(latest_path):
                for crt_name in ["Microsoft.VC143.CRT", "Microsoft.VC142.CRT", "Microsoft.VC141.CRT"]:
                    crt_path = os.path.join(latest_path, "x64", crt_name)
                    if os.path.exists(crt_path):
                        redist_paths.append(crt_path)

    # Add System32 which typically has redistributable DLLs
    system32_path = r"C:\Windows\System32"
    if os.path.exists(system32_path):
        redist_paths.append(system32_path)

    # Remove duplicates while preserving order
    unique_paths = []
    for path in redist_paths:
        if path not in unique_paths:
            unique_paths.append(path)

    return unique_paths

def find_dumpbin_paths():
    """
    Find all dumpbin.exe paths

    Returns:
        list: List of paths to dumpbin.exe
    """
    dumpbin_paths = []

    # Find all VS installations
    vs_installations = find_vs_installations()

    # Look for dumpbin.exe in all VS installations
    for vs_path in vs_installations:
        # Find all MSVC version directories
        msvc_dirs = glob.glob(os.path.join(vs_path, "VC", "Tools", "MSVC", "[0-9]*"))
        for msvc_dir in msvc_dirs:
            dumpbin_path = os.path.join(msvc_dir, "bin", "Hostx64", "x64", "dumpbin.exe")
            if os.path.exists(dumpbin_path):
                dumpbin_paths.append(dumpbin_path)

    # Look in PATH
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        potential_path = os.path.join(path_dir, "dumpbin.exe")
        if os.path.exists(potential_path):
            dumpbin_paths.append(potential_path)

    # Remove duplicates while preserving order
    unique_paths = []
    for path in dumpbin_paths:
        if path not in unique_paths:
            unique_paths.append(path)

    return unique_paths

def check_process_loaded_dlls(process_name=None):
    """
    Check DLLs loaded by a specific process or the current Python process

    Args:
        process_name (str, optional): Name of the process to check. If None, checks current Python process.

    Returns:
        dict: Dictionary mapping DLL names to their full paths
    """
    try:
        import win32process
        import win32api
        import psutil
    except ImportError:
        print("Required modules not available. Install pywin32 and psutil to use this function:")
        print("pip install pywin32 psutil")
        return {}

    if process_name is None:
        # Check current Python process
        pid = os.getpid()
        handle = win32api.GetCurrentProcess()
    else:
        # Find the process by name
        matching_processes = [p for p in psutil.process_iter(['pid', 'name'])
                             if p.info['name'].lower() == process_name.lower()]

        if not matching_processes:
            print(f"No processes found matching name: {process_name}")
            return {}

        # Use the first matching process
        pid = matching_processes[0].info['pid']
        try:
            handle = win32api.OpenProcess(
                win32process.PROCESS_QUERY_INFORMATION | win32process.PROCESS_VM_READ,
                False, pid
            )
        except Exception as e:
            print(f"Error opening process (pid={pid}): {e}")
            print("This might be due to insufficient permissions. Try running as administrator.")
            return {}

    # Get list of loaded modules
    try:
        modules = win32process.EnumProcessModules(handle)
    except Exception as e:
        print(f"Error enumerating process modules: {e}")
        return {}

    # Get the module filenames
    module_paths = {}
    for module in modules:
        try:
            module_path = win32process.GetModuleFileNameEx(handle, module)
            module_name = os.path.basename(module_path).lower()
            module_paths[module_name] = module_path
        except Exception:
            pass

    return module_paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find Visual Studio components")
    parser.add_argument("--vs", action="store_true", help="Find Visual Studio installations")
    parser.add_argument("--redist", action="store_true", help="Find MSVC redistributable paths")
    parser.add_argument("--dumpbin", action="store_true", help="Find dumpbin.exe paths")
    parser.add_argument("--dll-check", metavar="DLL", help="Check for a specific DLL in redist folders")
    parser.add_argument("--process", metavar="NAME", help="Check DLLs loaded by a specific process")
    parser.add_argument("--current", action="store_true", help="Check DLLs loaded by current Python process")
    parser.add_argument("--all", action="store_true", help="Show all information")
    args = parser.parse_args()

    # Default to --all if no specific options are provided
    if not (args.vs or args.redist or args.dumpbin or args.dll_check or args.process or args.current):
        args.all = True

    # Find Visual Studio installations
    if args.vs or args.all:
        vs_installations = find_vs_installations()
        print(f"\nFound {len(vs_installations)} Visual Studio installation(s):")
        for i, path in enumerate(vs_installations):
            print(f"  {i+1}. {path}")

    # Find MSVC redistributable paths
    if args.redist or args.all:
        redist_paths = find_msvc_redist_paths()
        print(f"\nFound {len(redist_paths)} MSVC redistributable path(s):")
        for i, path in enumerate(redist_paths):
            print(f"  {i+1}. {path}")

    # Find dumpbin.exe paths
    if args.dumpbin or args.all:
        dumpbin_paths = find_dumpbin_paths()
        print(f"\nFound {len(dumpbin_paths)} dumpbin.exe path(s):")
        for i, path in enumerate(dumpbin_paths):
            print(f"  {i+1}. {path}")

    # Check for specific DLL in redist folders
    if args.dll_check:
        redist_paths = find_msvc_redist_paths()
        dll_name = args.dll_check.lower()
        if not dll_name.endswith('.dll'):
            dll_name += '.dll'

        found_locations = []
        for redist_path in redist_paths:
            dll_path = os.path.join(redist_path, dll_name)
            if os.path.exists(dll_path):
                found_locations.append(dll_path)

        print(f"\nSearching for {dll_name}:")
        if found_locations:
            print(f"Found {dll_name} in {len(found_locations)} location(s):")
            for i, path in enumerate(found_locations):
                print(f"  {i+1}. {path}")
        else:
            print(f"  DLL {dll_name} not found in any redistributable folders")

    # Check DLLs loaded by a specific process
    if args.process:
        loaded_dlls = check_process_loaded_dlls(args.process)
        if loaded_dlls:
            print(f"\nDLLs loaded by process '{args.process}' ({len(loaded_dlls)} DLLs):")
            for name, path in sorted(loaded_dlls.items()):
                print(f"  {name}: {path}")
        else:
            print(f"\nCould not retrieve DLLs for process: {args.process}")

    # Check DLLs loaded by current Python process
    if args.current:
        loaded_dlls = check_process_loaded_dlls(None)
        if loaded_dlls:
            print(f"\nDLLs loaded by current Python process ({len(loaded_dlls)} DLLs):")
            for name, path in sorted(loaded_dlls.items()):
                print(f"  {name}: {path}")

            # Special check for MSVC and Intel runtime DLLs
            msvc_dlls = [name for name in loaded_dlls if name.startswith(('msvcp', 'vcruntime', 'concrt'))]
            intel_dlls = [name for name in loaded_dlls if name.startswith(('libifcoremd', 'libmmd', 'svml'))]

            print("\nMSVC Runtime DLLs loaded:")
            if msvc_dlls:
                for dll in msvc_dlls:
                    print(f"  ✓ {dll}: {loaded_dlls[dll]}")
            else:
                print("  None found")

            print("\nIntel Runtime DLLs loaded:")
            if intel_dlls:
                for dll in intel_dlls:
                    print(f"  ✓ {dll}: {loaded_dlls[dll]}")
            else:
                print("  None found")
        else:
            print("\nCould not retrieve DLLs for current Python process")
