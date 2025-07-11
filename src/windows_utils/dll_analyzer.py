"""
DLL Dependency Analyzer for Windows

This module provides functions to analyze DLL dependencies for executables and DLLs
on Windows systems. It can be used to identify missing dependencies for Python extensions.
"""

import os
import sys
import subprocess
import re
from collections import defaultdict

def find_dependencies(file_path):
    """
    Find direct DLL dependencies for a given DLL or executable

    Args:
        file_path (str): Path to the DLL or executable

    Returns:
        list: List of DLL names this file depends on
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get list of dumpbin paths
    try:
        # Try to use vs_finder if available (recommended approach)
        try:
           from .vs_finder import find_dumpbin_paths
        except:
            from vs_finder import find_dumpbin_paths
        dumpbin_paths = find_dumpbin_paths()
    except ImportError:
        # If vs_finder is not available, use a hardcoded path
        raise ImportError("vs_finder module not found. Please ensure it is installed or available in the path.")

    # Look for dumpbin in PATH as well
    for path_dir in os.environ.get('PATH', '').split(os.pathsep):
        dumpbin_path = os.path.join(path_dir, "dumpbin.exe")
        if os.path.exists(dumpbin_path):
            dumpbin_paths.append(dumpbin_path)

    # Remove duplicates while preserving order
    dumpbin_paths = list(dict.fromkeys(dumpbin_paths))

    dumpbin_path = None
    for path in dumpbin_paths:
        if os.path.exists(path):
            dumpbin_path = path
            print(f"Using dumpbin from: {dumpbin_path}")
            break

    if dumpbin_path:
        try:
            # Run dumpbin to get dependencies
            result = subprocess.run([dumpbin_path, "/DEPENDENTS", file_path],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                print(f"Error running dumpbin: {result.stderr}")
                return []

            # Debug: Print the entire output from dumpbin
            print("=== START DUMPBIN OUTPUT ===")
            print(result.stdout)
            print("=== END DUMPBIN OUTPUT ===")

            # Parse output to extract DLL names
            dependencies = []
            in_dependencies_section = False
            for line in result.stdout.splitlines():
                line = line.strip()

                if "Image has the following dependencies" in line:
                    in_dependencies_section = True
                    print(f"Found dependencies section: {line}")
                    continue

                if in_dependencies_section:
                    print(f"Analyzing line: '{line}'")
                    if line and ".dll" in line.lower():
                        # Remove leading spaces and any other characters
                        dll_name = line.strip()
                        dependencies.append(dll_name)
                        print(f"Added dependency: {dll_name}")
                    elif line.startswith("Summary") or line == "":
                        print(f"End of dependencies section: '{line}'")
                        in_dependencies_section = False

            # Check if we found dependencies in the expected format
            # If not, try an alternative parsing approach for dumpbin output
            if not dependencies:
                print("No dependencies found with standard parsing. Trying alternative parsing...")
                # Look for lines containing .dll after the dependencies header
                collect_dlls = False
                for line in result.stdout.splitlines():
                    line = line.strip()

                    if "Image has the following dependencies:" in line:
                        collect_dlls = True
                        continue

                    if collect_dlls:
                        if line.startswith("Summary"):
                            break
                        if line and ".dll" in line.lower():
                            # Found a DLL
                            dll_name = line.strip()
                            dependencies.append(dll_name)
                            print(f"Added dependency (alt method): {dll_name}")

            if not dependencies:
                print("Warning: Failed to parse dependencies from dumpbin output.")

            return dependencies
        except Exception as e:
            print(f"Error using dumpbin: {e}")

def find_dll_in_path(dll_name):
    """
    Find a DLL in system paths

    Args:
        dll_name (str): Name of the DLL to find

    Returns:
        str or None: Path where the DLL was found, or None if not found
    """
    # Check current directory first
    if os.path.exists(dll_name):
        return os.path.abspath(dll_name)

    # Check in the same directory as the Python executable
    python_dir = os.path.dirname(sys.executable)
    dll_path = os.path.join(python_dir, dll_name)
    if os.path.exists(dll_path):
        return dll_path

    # Check Windows system directories
    for path in [r"C:\Windows\System32", r"C:\Windows\SysWOW64", r"C:\Windows"]:
        dll_path = os.path.join(path, dll_name)
        if os.path.exists(dll_path):
            return dll_path

    # Check in PATH
    for path in os.environ.get('PATH', '').split(os.pathsep):
        dll_path = os.path.join(path, dll_name)
        if os.path.exists(dll_path):
            return dll_path

    return None

def analyze_dependency_tree(file_path, max_depth=3):
    """
    Analyze the full dependency tree for a DLL or executable

    Args:
        file_path (str): Path to the DLL or executable
        max_depth (int): Maximum depth to recurse in the dependency tree

    Returns:
        dict: Dictionary of dependencies by level
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Track visited DLLs to avoid cycles
    visited = set()

    # Track dependencies by level
    dependencies_by_level = defaultdict(list)

    def _analyze_recursive(path, current_depth=0):
        if current_depth > max_depth:
            return

        # Get basename for tracking visited DLLs
        dll_basename = os.path.basename(path).lower()

        # Skip if already visited
        if dll_basename in visited:
            return

        # Mark as visited
        visited.add(dll_basename)

        # Get direct dependencies
        direct_deps = find_dependencies(path)

        # Add to dependencies by level
        for dep in direct_deps:
            dep_path = find_dll_in_path(dep)
            if dep_path:
                dep_info = {
                    "name": dep,
                    "path": dep_path,
                    "exists": True
                }
            else:
                dep_info = {
                    "name": dep,
                    "path": None,
                    "exists": False
                }

            dependencies_by_level[current_depth].append(dep_info)

            # Recurse if dependency exists
            if dep_path and dep_info["exists"]:
                _analyze_recursive(dep_path, current_depth + 1)

    # Start recursion
    _analyze_recursive(file_path)

    return dict(dependencies_by_level)

def analyze_python_extension(module_path):
    """
    Analyze dependencies for a Python extension module

    Args:
        module_path (str): Path to the .pyd file

    Returns:
        dict: Dictionary with analysis results
    """
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    print(f"Analyzing Python extension: {module_path}")

    # Get direct dependencies
    direct_deps = find_dependencies(module_path)

    # Check which dependencies are missing
    missing_deps = []
    found_deps = []

    for dep in direct_deps:
        path = find_dll_in_path(dep)
        if path:
            found_deps.append((dep, path))
        else:
            missing_deps.append(dep)

    # Get full dependency tree
    dependency_tree = analyze_dependency_tree(module_path)

    return {
        "module": module_path,
        "direct_dependencies": direct_deps,
        "found_dependencies": found_deps,
        "missing_dependencies": missing_deps,
        "dependency_tree": dependency_tree
    }

def print_dependency_analysis(analysis):
    """
    Print the dependency analysis in a readable format

    Args:
        analysis (dict): Analysis result from analyze_python_extension
    """
    print(f"\nDependency Analysis for {analysis['module']}")
    print("=" * 80)

    print(f"\nDirect Dependencies ({len(analysis['direct_dependencies'])}):")
    for dep in analysis['direct_dependencies']:
        print(f"  {dep}")

    print(f"\nFound Dependencies ({len(analysis['found_dependencies'])}):")
    for name, path in analysis['found_dependencies']:
        print(f"  {name}: {path}")

    print(f"\nMissing Dependencies ({len(analysis['missing_dependencies'])}):")
    if analysis['missing_dependencies']:
        for dep in analysis['missing_dependencies']:
            print(f"  {dep}")
    else:
        print("  None - all dependencies found!")

    print("\nDependency Tree:")
    tree = analysis['dependency_tree']
    for level, deps in sorted(tree.items()):
        print(f"\nLevel {level}:")
        for dep in deps:
            status = "✓" if dep["exists"] else "✗"
            path = dep["path"] if dep["exists"] else "NOT FOUND"
            print(f"  {status} {dep['name']}: {path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze DLL dependencies for Windows")
    parser.add_argument("file_path", help="Path to the DLL or executable to analyze")
    parser.add_argument("--depth", type=int, default=2, help="Maximum depth for dependency tree analysis")

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    analysis = analyze_python_extension(args.file_path)
    print_dependency_analysis(analysis)
