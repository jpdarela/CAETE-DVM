#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Handler
-------------
Functionality to read input files for CAETE model.

"""

from abc import ABC, abstractmethod
from pathlib import Path
import concurrent.futures
import pickle
import sys

import numpy as np

# Suppress NumPy warnings that occurs during multiprocessing. 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed", category=RuntimeWarning)
    import netCDF4 as nc

import polars as pl

from caete import str_or_path, read_bz2_file
from config import fetch_config
from _geos import find_coordinates_xy, find_indices_xy 

cfg = fetch_config()


# Utility functions to create gridlists from input files

class base_handler(ABC):
    """
    An abstract base class for input handlers used by the CAETE model.
    
    This class provides a common interface for reading input data from different file formats
    (NetCDF, BZ2, etc.). It defines the essential methods that all input handlers must implement
    and provides common functionality for coordinate matching and gridlist validation.
    
    The class handles station-based data where each station has:
    - Time-varying variables: ['hurs', 'tas', 'ps', 'pr', 'rsds', 'sfcwind', 'vpd']
    - Station-specific variables: ['tn', 'tp', 'ap', 'ip', 'op']
    
    Attributes:
        fpath (Path): Path to the input file or data source folder
        isfile (bool): True if fpath points to a file, False if directory
        isdir (bool): True if fpath points to a directory, False if file
        time_vars (list): List of time-varying variable names expected in input data
        station_vars (list): List of station-specific variable names expected in input data
        gridlist (pl.DataFrame): Polars DataFrame containing station information
        coord_tolerance (float): Strict coordinate matching tolerance in degrees
        fallback_tolerance (float): Fallback tolerance for closest coordinate matching
        allow_closest (bool): Whether to allow closest coordinate fallback matching
    """
    def __init__(self, fpath, gridlist_df):
        """
        Initialize the base handler with a file/directory path and gridlist DataFrame.

        Args:
            fpath (str or Path): Path to the input file or data source folder
            gridlist_df (pl.DataFrame): Polars DataFrame containing station information
                                      Must include columns: 'station_name', 'lat', 'lon'
                                      
        Raises:
            ValueError: If fpath is empty or None
            TypeError: If gridlist_df is not a Polars DataFrame
        """
        if not fpath:
            raise ValueError("fpath is required")
        self.fpath = str_or_path(fpath)
        self.isfile = self.fpath.is_file()
        self.isdir = not self.isfile

        # Common attributes for all handlers.
        # All input files must store input data with these variables and names
        # The handlers must endure comatible variable units and names
        self.time_vars = ['hurs', 'tas', 'ps', 'pr', 'rsds', 'sfcwind', 'vpd']
        self.station_vars = ['tn', 'tp', 'ap', 'ip', 'op']


        # Store the gridlist DataFrame
        if not isinstance(gridlist_df, pl.DataFrame):
            raise TypeError("gridlist_df must be a Polars DataFrame")

        self.gridlist = gridlist_df
        self._validate_gridlist()

        # Coordinate matching settings
        self.coord_tolerance=1e-5
        self.fallback_tolerance=(cfg.crs.res / 2) - 0.001 # Subtract a small value to snap to grid
        self.allow_closest=True


    @abstractmethod
    def load_data(self):
        """
        Abstract method to read data from the input source.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def load_metadata(self):
        """
        Abstract method to read metadata from the input source.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def update_fpath(self, fpath):
        """
        Abstract method to update the file path.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def update_gridlist(self, gridlist_df):
        """
        Abstract method to update the gridlist DataFrame.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def close(self):
        """
        Abstract method to close any open resources.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def __enter__(self):
        """
        Abstract method for context manager entry.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Abstract method for context manager exit.
        Must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def __del__(self):
        """
        Default cleanup method that calls close().
        Subclasses can override if needed.
        """
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup

    
    # Common methods for all handlers
    def _find_closest_coordinate(self, target_lat, target_lon, nc_lats, nc_lons):
        """
        Find the index of the closest coordinate in NetCDF arrays using Euclidean distance.
        
        Args:
            target_lat (float): Target latitude coordinate to match
            target_lon (float): Target longitude coordinate to match
            nc_lats (np.ndarray): Array of NetCDF latitude values
            nc_lons (np.ndarray): Array of NetCDF longitude values
            
        Returns:
            int: Index of the closest coordinate in the NetCDF arrays
        """
        distances = np.sqrt((nc_lats - target_lat)**2 + (nc_lons - target_lon)**2)
        return np.argmin(distances)


    def _calculate_min_distance(self, target_lat, target_lon, nc_lats, nc_lons):
        """
        Calculate the minimum distance from target coordinates to any NetCDF coordinate.
        
        Args:
            target_lat (float): Target latitude coordinate
            target_lon (float): Target longitude coordinate
            nc_lats (np.ndarray): Array of NetCDF latitude values
            nc_lons (np.ndarray): Array of NetCDF longitude values
            
        Returns:
            float: Minimum Euclidean distance in degrees
        """
        distances = np.sqrt((nc_lats - target_lat)**2 + (nc_lons - target_lon)**2)
        return np.min(distances)


    def _validate_gridlist(self):
        """
        Validate gridlist DataFrame for duplicate station names and data integrity.
        
        Raises:
            ValueError: If duplicate station names are found in the gridlist
        """
        station_names = self.gridlist.get_column('station_name').to_list()
        if len(station_names) != len(set(station_names)):
            duplicates = [name for name in set(station_names)
                            if station_names.count(name) > 1]
            raise ValueError(f"Duplicate station names in gridlist: {duplicates}")

class bz2_handler(base_handler):
    """
    A handler for reading bz2 compressed input files for the CAETE model.
    
    This class is designed to read input data from bz2 compressed files containing
    pickled station data. It expects a directory structure where each station's
    data is stored in a separate .pbz2 file, with a naming convention based on
    global coordinates (input_data_{global_y}-{global_x}.pbz2).
    
    The handler supports multithreaded file reading for improved performance
    and includes metadata handling through a separate METADATA.pbz2 file.
    
    Attributes:
        b22_metadata_filename (str): Filename for the metadata file ("METADATA.pbz2")
        input_files (list): List of Path objects for input data files
        station_names (list): List of corresponding station names for input files
    """
    def __init__(self, fpath, gridlist_df):
        """
        Initialize the bz2 handler with a directory path and gridlist DataFrame.

        Args:
            fpath (str or Path): Path to the directory containing bz2 files
            gridlist_df (pl.DataFrame): Polars DataFrame containing station information
                                      Must include columns: 'global_y', 'global_x'
                                      
        Raises:
            ValueError: If fpath is not a valid directory
        """
        super().__init__(fpath, gridlist_df)

        # Metadata filename for pickled bz2 files.
        self.b22_metadata_filename = "METADATA.pbz2"

        # For the bz2 handler, we expect a valid directory path
        # Enforce that fpath is a directory
        if not self.isdir:
            raise ValueError(f"Provided path {self.fpath} is not a valid directory")
        
        self.original_gridlist = gridlist_df

        self.input_files = []
        self.station_names = []
        self._find_files(gridlist_df)


    def _find_files(self, gridlist_df):
        """
        Generate file paths based on gridlist coordinates and validate file existence.
        
        Creates file paths using the pattern: input_data_{global_y}-{global_x}.pbz2
        where global_y and global_x are extracted from the gridlist DataFrame.
        
        Args:
            gridlist_df (pl.DataFrame): DataFrame containing global coordinate columns
                                      'global_y' and 'global_x'
                                      
        Note:
            This method updates self.input_files with existing file paths and
            self.station_names with corresponding station identifiers.
        """
        # This method is a placeholder and should be implemented when bz2 handling is added
        base_name = "input_data"
        file_extension = ".pbz2"
        self.input_files = (
            gridlist_df
            .select([
                pl.format("{}_{}-{}{}",
                        pl.lit(base_name),
                        pl.col("global_y"),
                        pl.col("global_x"),
                        pl.lit(file_extension))
                .alias("filename")
            ])
            .get_column("filename")
            .to_list()
        )
        # Convert to Path objects
        self.input_files = [Path(self.fpath / f) for f in self.input_files]
        # Filter out non-existent files
        self.input_files = self._filter_non_existent_files()


    def _filter_non_existent_files(self):
        """
        Filter the input_files list to include only files that exist on the filesystem.
        
        Also generates corresponding station names based on the filtered file paths
        using the pattern "station_{coordinate_suffix}" where coordinate_suffix
        is extracted from the filename.
        
        Returns:
            list: List of Path objects for files that exist on the filesystem
            
        Note:
            Updates self.station_names with corresponding station identifiers
        """
        file_paths = [f for f in self.input_files if f.exists()]
        self.station_names = ["station_" + f.stem.split("_")[-1] for f in file_paths]
        return file_paths


    def _read_metadata_file(self):
        """
        Read and process metadata from the METADATA.pbz2 file.
        
        Extracts time, latitude, and longitude metadata from the compressed metadata file
        and filters it to match only the stations present in the current gridlist.
        Converts masked arrays to regular numpy arrays for consistency.
        
        Returns:
            tuple: A tuple containing three dictionaries:
                - filtered_time_meta (dict): Time dimension metadata
                - filtered_lat_meta (dict): Latitude metadata for gridlist stations
                - filtered_lon_meta (dict): Longitude metadata for gridlist stations
                
        Raises:
            FileNotFoundError: If the METADATA.pbz2 file does not exist
        """
        fpath = self.fpath / self.b22_metadata_filename
        if not fpath.exists():
            raise FileNotFoundError(f"Metadata file {fpath} does not exist.")

        # Read the full metadata
        full_metadata = read_bz2_file(self.fpath / self.b22_metadata_filename)

        # Extract time, lat, lon metadata
        time_meta, lat_meta, lon_meta = full_metadata

        # Get the coordinates for our specific stations from the gridlist
        gridlist_data = self.gridlist.select(['global_y', 'global_x', 'lat', 'lon']).to_pandas()

        # Filter latitude metadata to only include our stations
        filtered_lat_meta = lat_meta.copy()
        # Remove the full global array and replace with station-specific data
        if 'lat_index' in filtered_lat_meta:
            # Convert masked array to regular array if needed
            if hasattr(filtered_lat_meta['lat_index'], 'filled'):
                filtered_lat_meta['lat_index'] = gridlist_data['lat'].values.astype(np.float64)
            else:
                filtered_lat_meta['lat_index'] = gridlist_data['lat'].values.astype(np.float64)

        # Filter longitude metadata to only include our stations
        filtered_lon_meta = lon_meta.copy()
        # Remove the full global array and replace with station-specific data
        if 'lon_index' in filtered_lon_meta:
            # Convert masked array to regular array if needed
            if hasattr(filtered_lon_meta['lon_index'], 'filled'):
                filtered_lon_meta['lon_index'] = gridlist_data['lon'].values.astype(np.float64)
            else:
                filtered_lon_meta['lon_index'] = gridlist_data['lon'].values.astype(np.float64)

        # Convert time metadata to regular array if it's masked
        filtered_time_meta = time_meta.copy()
        if 'time_index' in filtered_time_meta:
            if hasattr(filtered_time_meta['time_index'], 'filled'):
                filtered_time_meta['time_index'] = filtered_time_meta['time_index'].filled()

        return filtered_time_meta, filtered_lat_meta, filtered_lon_meta

    # Required methods by base_handler
    def load_data(self):
        """
        Load data from multiple bz2 files using multithreaded file reading.
        
        Reads all bz2 files in parallel using ThreadPoolExecutor for improved performance.
        Each file is expected to contain pickled station data that can be unpickled
        using the read_bz2_file function.
        
        Returns:
            dict: Dictionary with station names as keys and unpickled data as values.
                 Returns empty dict if no input files are available.
                 
        Raises:
            IOError: If any file cannot be read or unpickled
        """
        if not self.input_files:
            return {}

        def read_file_with_name(file_station_pair):
            """Read a single file and return (station_name, data) tuple."""
            filepath, station_name = file_station_pair
            try:
                data = read_bz2_file(filepath)
                return station_name, data
            except Exception as e:
                raise IOError(f"Error reading {filepath}: {e}")

        # Prepare input data
        file_station_pairs = list(zip(self.input_files, self.station_names))
        max_workers = min(len(file_station_pairs), 8)

        # REad files
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(read_file_with_name, file_station_pairs))

        # Convert results to dictionary
        return dict(results)


    def load_metadata(self):
        """
        Load and return metadata from the METADATA.pbz2 file.
        
        This is a wrapper method that calls _read_metadata_file() to extract
        metadata information filtered for the current gridlist stations.
        
        Returns:
            tuple: A tuple containing three dictionaries:
                - time_meta (dict): Time dimension metadata
                - lat_meta (dict): Latitude metadata for gridlist stations  
                - lon_meta (dict): Longitude metadata for gridlist stations
        """
        return self._read_metadata_file()


    def update_fpath(self, fpath, gridlist_df=None):
        """
        Update the directory path for the bz2 handler and optionally update the gridlist.
        
        Changes the source directory for bz2 files and re-scans for available files
        based on the current or new gridlist. Validates that the new path is a directory
        and that matching files exist.

        Args:
            fpath (str or Path): New directory path containing bz2 files
            gridlist_df (pl.DataFrame, optional): New gridlist DataFrame to use.
                                                 If None, uses existing gridlist.
                                                 
        Raises:
            ValueError: If fpath is empty, not a directory, or no matching files found
        """
        if gridlist_df is not None:
            self.update_gridlist(gridlist_df)
        else:
            print("[WARN] No gridlist_df provided, using existing gridlist")
        if not fpath:
            raise ValueError("fpath is required")

        self.fpath = str_or_path(fpath)
        if not self.fpath.is_dir():
            raise ValueError("fpath must be a directory for bz2_handler")
        self.input_files = []
        self._find_files(self.gridlist)
        if len(self.input_files) == 0:
            raise ValueError("No input files found for the given gridlist DataFrame")


    def update_gridlist(self, gridlist_df):
        """
        Update the gridlist DataFrame and re-map input files accordingly.
        
        Replaces the current gridlist with a new one and rescans the directory
        for files matching the new stations. Validates the new gridlist and
        ensures that matching files exist.

        Args:
            gridlist_df (pl.DataFrame): New gridlist DataFrame containing station information.
                                       Must include 'global_y' and 'global_x' columns.
                                       
        Raises:
            TypeError: If gridlist_df is not a Polars DataFrame
            ValueError: If no input files found for the new gridlist
        """
        if not isinstance(gridlist_df, pl.DataFrame):
            raise TypeError("gridlist_df must be a Polars DataFrame")
        self.gridlist = gridlist_df
        self._validate_gridlist()
        self.input_files = []
        self._find_files(gridlist_df)
        if len(self.input_files) == 0:
            raise ValueError("No input files found for the given gridlist DataFrame")


    def close(self):
        """
        Close any open resources for the bz2_handler.
        
        This is a no-op method since bz2_handler doesn't keep files persistently open.
        Implemented to maintain consistency with the base_handler interface.
        """
        # bz2_handler doesn't keep files open, so nothing to close
        # But we implement this for consistency with the interface
        pass


    def __enter__(self): # type: ignore[override]
        """
        Enter the context manager for the bz2_handler.
        
        Returns:
            bz2_handler: Returns self to enable context manager usage
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager for the bz2_handler.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred  
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()


    def __del__(self):
        """
        Destructor that ensures proper cleanup when the object is garbage collected.
        """
        self.close()

class netcdf_handler(base_handler):
    """
    A class for handling NetCDF input files for the CAETE model.

    This class reads a NetCDF file and loads all the variables into a dictionary.
    It also provides methods to access metadata about the NetCDF file.

    The class is designed to handle gridlists that contain only a subset of stations
    from the netCDF file. It will efficiently map station names to their indices
    in the netCDF file and return only the data for the stations in the gridlist.

    All data is converted from masked arrays to regular numpy arrays for consistency
    and easier use in downstream processing.

    Attributes:
        nc_file (str): Path to the NetCDF file
        nc_data (netCDF4.Dataset): NetCDF dataset
        gridlist (polars.DataFrame): Gridlist data
    """

    def __init__(self, nc_file_path, gridlist_df):
        """
        Initialize the NetCDF handler with file path and gridlist DataFrame.

        Args:
            nc_file_path (str or Path): Path to the NetCDF file containing input data
            gridlist_df (pl.DataFrame): Pre-loaded gridlist as a Polars DataFrame
                                       Must include columns: 'station_name', 'lat', 'lon'
                                       
        Raises:
            ValueError: If nc_file_path is not a valid file
            IOError: If NetCDF file cannot be opened or handler initialization fails
        """
        super().__init__(nc_file_path, gridlist_df)

        # For the netCDF handler, we expect a valid file path
        if not self.isfile:
            raise ValueError(f"Provided path {self.fpath} is not a valid file file")
        self.nc_file = self.fpath
        self.nc_data = None

        try:
            self.nc_data = nc.Dataset(self.nc_file, 'r')
            # Pre-load station indices and prepare for data loading
            self._station_indices = self._map_station_indices()
            self.mpi = cfg.input_handler.mp #type: ignore[assignment]
        except Exception as e:
            # Ensure file is closed if initialization fails
            if self.nc_data is not None:
                try:
                    self.nc_data.close()
                except:
                    pass  # Ignore errors during cleanup
            raise IOError(f"Error during NetCDF handler initialization: {e}")

    def _map_station_indices(self):
        """
        Create a mapping between station names and their indices in the NetCDF file.
        
        This method performs coordinate matching between the gridlist and NetCDF file,
        supporting both exact matching and closest-coordinate fallback. It validates
        that each station in the gridlist has a corresponding location in the NetCDF file.
        
        The method supports multiple coordinate matching strategies:
        1. Exact coordinate matching within coord_tolerance
        2. Closest coordinate fallback (if allow_closest=True)
        3. Station name validation (if available in NetCDF)
        
        Returns:
            dict: Dictionary mapping station names to their NetCDF array indices
            
        Raises:
            ValueError: If coordinates don't match, required variables are missing,
                       or multiple stations map to the same NetCDF index
            KeyError: If required columns are missing from gridlist
        """
        # Validate required columns in gridlist
        required_cols = ['station_name', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in self.gridlist.columns]
        if missing_cols:
            raise KeyError(f"Required columns missing from gridlist: {missing_cols}")

        # Find latitude and longitude variables in NetCDF (support both naming conventions)
        lat_var_name = self._find_coordinate_variable('latitude')
        lon_var_name = self._find_coordinate_variable('longitude')

        # Get coordinates from gridlist
        gridlist_data = self.gridlist.select(['station_name', 'lat', 'lon']).to_pandas()
        station_names = gridlist_data['station_name'].tolist()
        gridlist_lats = gridlist_data['lat'].values
        gridlist_lons = gridlist_data['lon'].values

        # Load NetCDF coordinates using the detected variable names
        nc_lats = self.nc_data.variables[lat_var_name][:].astype(np.float64) # type: ignore[assignment]
        nc_lons = self.nc_data.variables[lon_var_name][:].astype(np.float64) # type: ignore[assignment]

        # Handle masked arrays
        if isinstance(nc_lats, np.ma.MaskedArray):
            if nc_lats.mask.any():
                raise ValueError("NetCDF latitude data contains masked/invalid values")
            nc_lats = nc_lats.filled()

        if isinstance(nc_lons, np.ma.MaskedArray):
            if nc_lons.mask.any():
                raise ValueError("NetCDF longitude data contains masked/invalid values")
            nc_lons = nc_lons.filled()

        # Validate that NetCDF has station dimension
        if len(nc_lats.shape) != 1 or len(nc_lons.shape) != 1:
            raise ValueError("NetCDF lat/lon variables must be 1-dimensional station arrays")

        if len(nc_lats) != len(nc_lons):
            raise ValueError("NetCDF lat and lon arrays have different lengths")

        # Define coordinate tolerance (adjust based on your data precision)
        coord_tolerance = self.coord_tolerance  # Use instance variable

        station_indices = {}
        unmatched_stations = []
        fallback_matches = []  # Track stations matched via closest distance

        # For each station in gridlist, find exact coordinate match in NetCDF
        for station_name, grid_lat, grid_lon in zip(station_names, gridlist_lats, gridlist_lons):
            # Validate gridlist coordinates
            if np.isnan(grid_lat) or np.isnan(grid_lon):
                raise ValueError(f"Station '{station_name}' has invalid coordinates: lat={grid_lat}, lon={grid_lon}")

            # Find matching coordinates in NetCDF
            lat_matches = np.abs(nc_lats - grid_lat) < coord_tolerance
            lon_matches = np.abs(nc_lons - grid_lon) < coord_tolerance
            coord_matches = lat_matches & lon_matches

            match_indices = np.where(coord_matches)[0]

            if len(match_indices) == 0:
                # No exact coordinate match found - try fallback if enabled
                if self.allow_closest:
                    closest_idx = self._find_closest_coordinate(grid_lat, grid_lon, nc_lats, nc_lons)
                    min_dist = self._calculate_min_distance(grid_lat, grid_lon, nc_lats, nc_lons)

                    if min_dist <= self.fallback_tolerance:
                        # Check if this NetCDF index is already used
                        if closest_idx in station_indices.values():
                            existing_station = [k for k, v in station_indices.items() if v == closest_idx][0]
                            raise ValueError(
                                f"Closest coordinate fallback failed: NetCDF index {closest_idx} already used by station '{existing_station}'. "
                                f"Station '{station_name}' would also map to the same index (distance: {min_dist:.6f}°)"
                            )

                        # Use closest match as fallback
                        station_indices[station_name] = closest_idx
                        fallback_matches.append({
                            'station': station_name,
                            'gridlist_lat': grid_lat,
                            'gridlist_lon': grid_lon,
                            'netcdf_lat': nc_lats[closest_idx],
                            'netcdf_lon': nc_lons[closest_idx],
                            'netcdf_idx': closest_idx,
                            'distance': min_dist
                        })
                        continue

                # No coordinate match found (either no fallback or fallback failed)
                unmatched_stations.append({
                    'station': station_name,
                    'gridlist_lat': grid_lat,
                    'gridlist_lon': grid_lon,
                    'netcdf_lat': nc_lats[closest_idx],
                    'netcdf_lon': nc_lons[closest_idx],
                    'closest_nc_idx': self._find_closest_coordinate(grid_lat, grid_lon, nc_lats, nc_lons),
                    'min_distance': self._calculate_min_distance(grid_lat, grid_lon, nc_lats, nc_lons)
                })
                continue

            elif len(match_indices) > 1:
                # Multiple matches - this shouldn't happen with good data
                raise ValueError(
                    f"Station '{station_name}' matches multiple NetCDF grid points at "
                    f"lat={grid_lat}, lon={grid_lon}. Indices: {match_indices.tolist()}"
                )

            # Exact match found
            nc_idx = match_indices[0]

            # Additional validation: check for duplicate NetCDF indices
            if nc_idx in station_indices.values():
                existing_station = [k for k, v in station_indices.items() if v == nc_idx][0]
                raise ValueError(
                    f"Multiple stations map to the same NetCDF index {nc_idx}: "
                    f"'{existing_station}' and '{station_name}'"
                )

            station_indices[station_name] = nc_idx

        # Report fallback matches if any were used
        if fallback_matches:
            import warnings
            warning_msg = f"\nUsed closest coordinate fallback for {len(fallback_matches)} stations:\n"
            for match_info in fallback_matches[:5]:  # Show first 5 for brevity
                warning_msg += (
                    f"  - '{match_info['station']}': "
                    f"gridlist=({match_info['gridlist_lat']:.6f}, {match_info['gridlist_lon']:.6f}), "
                    f"netcdf=({match_info['netcdf_lat']:.6f}, {match_info['netcdf_lon']:.6f}), "
                    f"distance={match_info['distance']:.6f}°, "
                    f" NetCDF Index {match_info['netcdf_idx']}\n"
                )
            if len(fallback_matches) > 5:
                warning_msg += f"  ... and {len(fallback_matches) - 5} more stations\n"

            warning_msg += f"Fallback tolerance: {self.fallback_tolerance}°"

            # Fallback matches are not errors, so use a warning
            warnings.warn(warning_msg, UserWarning)

        # Report unmatched stations
        if unmatched_stations:
            error_msg = f"Failed to match {len(unmatched_stations)} stations to NetCDF coordinates:\n"
            for station_info in unmatched_stations[:5]:  # Show first 5 for brevity
                error_msg += (
                    f"  - '{station_info['station']}': "
                    f"gridlist=({station_info['gridlist_lat']:.6f}, {station_info['gridlist_lon']:.6f}), "
                    f"closest_netcdf=({station_info['netcdf_lat']:.6f}, {station_info['netcdf_lon']:.6f}), "
                    f"min_distance={station_info['min_distance']:.6f}°\n"
                )
            if len(unmatched_stations) > 5:
                error_msg += f"  ... and {len(unmatched_stations) - 5} more stations\n"

            error_msg += f"\nCoordinate tolerance: {coord_tolerance}°"
            raise ValueError(error_msg)

        # Final validation: ensure we have the expected number of stations
        if len(station_indices) != len(station_names):
            raise ValueError(
                f"Mismatch in station count: gridlist has {len(station_names)} stations, "
                f"but only {len(station_indices)} were successfully mapped"
            )

        # Optional: Validate station names if available in NetCDF
        if 'station_name' in self.nc_data.variables: # type: ignore[union-attr]
            self._validate_station_names(station_indices)

        return station_indices

    def _validate_station_names(self, station_indices):
        """
        Perform additional validation using station names if available in the NetCDF file.
        
        Cross-references station names between the gridlist and NetCDF file to identify
        any mismatches that might indicate coordinate mapping errors.

        Args:
            station_indices (dict): Mapping of station names to NetCDF array indices
            
        Note:
            Emits warnings for station name mismatches but does not raise exceptions,
            allowing the process to continue with coordinate-based matching.
        """
        try:
            nc_station_names = self._get_netcdf_station_names()

            mismatched_names = []
            for station_name, nc_idx in station_indices.items():
                if nc_idx < len(nc_station_names):
                    nc_name = nc_station_names[nc_idx]
                    if nc_name != station_name:
                        mismatched_names.append({
                            'gridlist_name': station_name,
                            'netcdf_name': nc_name,
                            'index': nc_idx
                        })

            if mismatched_names:
                error_msg = "Station name mismatches detected:\n"
                for mismatch in mismatched_names[:5]:
                    error_msg += (
                        f"  Index {mismatch['index']}: "
                        f"gridlist='{mismatch['gridlist_name']}' vs "
                        f"netcdf='{mismatch['netcdf_name']}'\n"
                    )
                if len(mismatched_names) > 5:
                    error_msg += f"  ... and {len(mismatched_names) - 5} more mismatches"

                # This could be a warning instead of an error depending on your use case
                import warnings
                warnings.warn(error_msg, UserWarning)

        except Exception as e:
            # Don't fail the whole process if station name validation fails
            import warnings
            warnings.warn(f"Could not validate station names: {e}", UserWarning)

    def _get_netcdf_station_names(self):
        """
        Extract and convert station names from the NetCDF file to a list of strings.
        
        Reads the 'station_name' variable from the NetCDF file and converts each
        entry to a properly formatted string with whitespace stripped.

        Returns:
            list: List of station names as strings extracted from the NetCDF file
            
        Note:
            Assumes the NetCDF file has a 1D 'station_name' variable.
        """

        # Get the station_name variable
        station_name_var = self.nc_data.variables['station_name']

        # We assume 1D string arrays for the station names
        names = [str(name).strip() for name in station_name_var[:]]
        return names

    def _find_coordinate_variable(self, coord_type):
        """
        Find coordinate variable name in NetCDF file using multiple naming conventions.
        
        Searches for latitude or longitude variables using various naming patterns
        and CF conventions, including variable names, standard_name attributes,
        axis attributes, and long_name attributes.

        Args:
            coord_type (str): Either 'latitude' or 'longitude' to specify coordinate type
            
        Returns:
            str: The actual variable name found in the NetCDF file
            
        Raises:
            ValueError: If coord_type is invalid or no coordinate variable is found
        """
        if coord_type == 'latitude':
            # Check for latitude variable names in order of preference
            possible_names = ['lat', 'latitude', 'LAT', 'LATITUDE', 'Lat', 'Latitude']
            standard_name_attrs = ['latitude']
            axis_attrs = ['Y']
        elif coord_type == 'longitude':
            # Check for longitude variable names in order of preference
            possible_names = ['lon', 'longitude', 'LON', 'LONGITUDE', 'Lon', 'Longitude']
            standard_name_attrs = ['longitude']
            axis_attrs = ['X']
        else:
            raise ValueError(f"Invalid coordinate type: {coord_type}")

        # First, try to find by exact variable name match
        for name in possible_names:
            if name in self.nc_data.variables:
                return name

        # If not found by name, try to find by standard_name attribute
        for var_name, var in self.nc_data.variables.items():
            if hasattr(var, 'standard_name'):
                if var.standard_name.lower() in [attr.lower() for attr in standard_name_attrs]:
                    return var_name

        # If still not found, try to find by axis attribute
        for var_name, var in self.nc_data.variables.items():
            if hasattr(var, 'axis'):
                if var.axis.upper() in axis_attrs:
                    return var_name

        # If still not found, try to find by long_name attribute (common fallback)
        long_name_patterns = {
            'latitude': ['latitude', 'lat', 'y', 'north'],
            'longitude': ['longitude', 'lon', 'x', 'east', 'west']
        }

        for var_name, var in self.nc_data.variables.items():
            if hasattr(var, 'long_name'):
                long_name = var.long_name.lower()
                for pattern in long_name_patterns[coord_type]:
                    if pattern in long_name:
                        return var_name

        # If nothing found, raise an error with helpful information
        available_vars = list(self.nc_data.variables.keys())
        raise ValueError(
            f"No {coord_type} variable found in NetCDF file. "
            f"Searched for: {possible_names}. "
            f"Available variables: {available_vars}. "
            f"Consider checking variable names or CF convention compliance."
        )

    def load_data_parallel(self):
        """
        Load NetCDF data using MPI for parallel file reading of time-varying variables.
        
        Executes an external MPI script to read time-varying variables in parallel,
        which can significantly improve performance for large NetCDF files. Each
        time variable is read by a separate MPI process.
        
        Returns:
            dict: Dictionary containing time-varying variable data for all stations
            
        Raises:
            RuntimeError: If the MPI process fails or returns non-zero exit code
            
        Note:
            Requires 'netcdf_reader.py' script and MPI installation.
            Station variables are not read in parallel due to their small size.
        """
        import subprocess

        indices = np.array(list(self._station_indices.values()))
        indices_str = ','.join(map(str, indices))

        # Call MPI script for time variables
        nprocs = len(self.time_vars)
        cmd_time = [
            'mpiexec', '-n', f'{nprocs}',  # Use 4 processes
            sys.executable, 'netcdf_reader.py',
            str(self.nc_file),
            indices_str,
            '--var-type', 'time'
        ] + self.time_vars

        result_time = subprocess.run(cmd_time, capture_output=True)
        if result_time.returncode != 0:
            raise RuntimeError(f"MPI time variables process failed: {result_time.stderr.decode()}")
        time_vars_data = pickle.loads(result_time.stdout)

        # Causes overhead launching processes, so commented out for now
        # # Call MPI script for station variables
        # nprocs = len(self.station_vars)
        # cmd_station = [
        #     'mpiexec', '-n', f'{nprocs}',
        #     sys.executable, 'netcdf_reader.py',
        #     str(self.nc_file),
        #     indices_str,
        #     '--var-type', 'station'
        # ] + self.station_vars

        # result_station = subprocess.run(cmd_station, capture_output=True)
        # if result_station.returncode != 0:
        #     raise RuntimeError(f"MPI station variables process failed: {result_station.stderr.decode()}")
        # station_vars_data = pickle.loads(result_station.stdout)

        return time_vars_data #, station_vars_data

    def load_data(self):
        """
        Load data from the NetCDF file into a dictionary organized by station names.
        
        This method efficiently loads all required variables for the stations specified
        in the gridlist. It supports both parallel (MPI) and sequential reading modes
        depending on the configuration and NetCDF4 parallel support availability.
        
        The method performs bulk loading of data to minimize file I/O operations:
        1. Loads all time-varying variables at once for all stations
        2. Loads all station-specific variables at once for all stations  
        3. Distributes the data into per-station dictionaries
        
        Variables loaded:
        - Time-varying: ['hurs', 'tas', 'ps', 'pr', 'rsds', 'sfcwind', 'vpd']
        - Station-specific: ['tn', 'tp', 'ap', 'ip', 'op']

        Returns:
            dict: Dictionary with station names as keys and variable data as values.
                 Each station's data contains all available variables as numpy arrays.
                 Returns empty dict if no stations are mapped.
                 
        Raises:
            RuntimeError: If parallel NetCDF support is not available when MPI is enabled
            
        Note:
            Masked arrays are automatically converted to regular numpy arrays with
            mean-value filling for missing data.
        """
        data_dict = {}
        station_names = list(self._station_indices.keys())

        if not station_names:
            return data_dict

        # Get all the indices we need to extract from the netCDF file
        indices = np.array(list(self._station_indices.values()))

        # Pre-load all time-series variables into memory at once
        time_vars_data = {}
        station_vars_data = {}

        if self.mpi:
            if not nc.__has_parallel4_support__:
                raise RuntimeError("NetCDF4 parallel support is not available. "
                                   "Ensure you have compiled NetCDF with parallel I/O support."
                                   "Or set mp [in the input_handler session] to false in the caete.toml file.")
            # Use parallel loading with MPI processes to read time-varying variables
            time_vars_data = self.load_data_parallel()
        else:
            # Read sequentially
            for var_name in self.time_vars:
                if var_name in self.nc_data.variables:
                    # Read all station data at once using fancy indexing
                    # This is much faster than reading each station individually
                    var_data = self.nc_data.variables[var_name][:, indices].astype(np.float32)

                    # Convert masked arrays to regular arrays if needed
                    if isinstance(var_data, np.ma.MaskedArray):
                        fill_value = var_data.mean()
                        var_data = var_data.filled(fill_value=fill_value)

                    # Store for later distribution
                    time_vars_data[var_name] = var_data

        # Read non time-varying station-specific variables. Always serially read these (small in size)
        for var_name in self.station_vars:
            if var_name in self.nc_data.variables:
                # Read all station data at once
                var_data = self.nc_data.variables[var_name][indices].astype(np.float32)

                # Convert masked arrays to regular arrays if needed
                if isinstance(var_data, np.ma.MaskedArray):
                    fill_value = var_data.mean()
                    var_data = var_data.filled(fill_value=fill_value)

                # Store for later distribution
                station_vars_data[var_name] = var_data

        # Distribute data to station dictionaries
        for i, station_name in enumerate(station_names):
            station_data = {}

            # Extract time-varying variables for this station from pre-loaded data
            for var_name in self.time_vars:
                if var_name in time_vars_data:
                    # Use the pre-loaded data instead of accessing the file again
                    station_data[var_name] = time_vars_data[var_name][:, i]

            # Extract station-specific variables from pre-loaded data
            for var_name in self.station_vars:
                if var_name in station_vars_data:
                    # Use the pre-loaded data instead of accessing the file again
                    station_data[var_name] = float(station_vars_data[var_name][i])

            data_dict[station_name] = station_data

        return data_dict

    def load_metadata(self):
        """
        Load metadata from the NetCDF file for time, latitude, and longitude dimensions.
        
        Extracts comprehensive metadata including attributes and coordinate values
        for the time, latitude, and longitude dimensions. The metadata is filtered
        to include only the stations present in the current gridlist.
        
        Uses flexible coordinate variable detection to handle different NetCDF
        naming conventions (lat/latitude, lon/longitude, etc.).

        Returns:
            tuple: A tuple containing three dictionaries:
                - time_meta (dict): Time dimension metadata with attributes and values
                - lat_meta (dict): Latitude metadata with attributes and coordinate values
                - lon_meta (dict): Longitude metadata with attributes and coordinate values
                
        Note:
            - Masked arrays are converted to regular numpy arrays
            - Metadata includes both variable attributes and coordinate index arrays
            - Only stations matching the current gridlist are included in lat/lon metadata
        """
        # Extract time metadata
        time_meta = {}
        time_var = self.nc_data.variables['time']

        # Load time data all at once for better performance
        time_data = time_var[:].astype(float)

        for attr in time_var.ncattrs():
            if attr != '_FillValue':
                time_meta[attr] = time_var.getncattr(attr)        # Add time index - convert to regular numpy array if it's masked
        if isinstance(time_data, np.ma.MaskedArray):
            time_meta['time_index'] = time_data.filled()
        else:
            time_meta['time_index'] = time_data

        # Extract latitude metadata using flexible variable name detection
        lat_var_name = self._find_coordinate_variable('latitude')
        indices = np.array(list(self._station_indices.values())) # Get indices for all stations in the gridlist
        lat_meta = {}
        lat_var = self.nc_data.variables[lat_var_name]

        # Load latitude data all at once for better performance
        lat_data = lat_var[indices]

        for attr in lat_var.ncattrs():
            if attr != '_FillValue':
                lat_meta[attr] = lat_var.getncattr(attr)

        # Adjust axis attribute if needed
        if 'axis' in lat_meta and lat_meta['axis'] == 'X':
            lat_meta['axis'] = 'Y'

        # Add lat index as regular numpy array
        if isinstance(lat_data, np.ma.MaskedArray):
            lat_meta['lat_index'] = lat_data.filled()
        else:
            lat_meta['lat_index'] = lat_data

        # Extract longitude metadata using flexible variable name detection
        lon_var_name = self._find_coordinate_variable('longitude')
        lon_meta = {}
        lon_var = self.nc_data.variables[lon_var_name]

        # Load longitude data all at once for better performance
        lon_data = lon_var[indices]

        for attr in lon_var.ncattrs():
            if attr != '_FillValue':
                lon_meta[attr] = lon_var.getncattr(attr)

        # Set standard name if missing
        if 'standard_name' not in lon_meta:
            lon_meta['standard_name'] = 'longitude'

        # Add lon index as regular numpy array
        if isinstance(lon_data, np.ma.MaskedArray):
            lon_meta['lon_index'] = lon_data.filled()
        else:
            lon_meta['lon_index'] = lon_data

        return time_meta, lat_meta, lon_meta

    def update_fpath(self, fpath, gridlist_df=None):
        """
        Update the NetCDF file path and optionally the gridlist, reopening the file.
        
        Closes the current NetCDF file, opens the new file, and remaps station indices
        based on the current or new gridlist. This allows switching between different
        NetCDF files while maintaining the same handler instance.

        Args:
            fpath (str or Path): Path to the new NetCDF file
            gridlist_df (pl.DataFrame, optional): New gridlist DataFrame to use.
                                                 If None, uses existing gridlist.
                                                 
        Raises:
            ValueError: If fpath is empty or not a valid NetCDF file
            IOError: If the new NetCDF file cannot be opened
        """

        if gridlist_df is not None:
            self.update_gridlist(gridlist_df)
        else:
            print("[WARN] No gridlist_df provided, using existing gridlist")
        if not fpath:
            raise ValueError("fpath is required")

        self.fpath = str_or_path(fpath)
        if not self.isfile:
            raise ValueError("fpath must be a valid NetCDF file for netcdf_handler")
        self.nc_file = self.fpath

        # Close the existing NetCDF dataset if open
        if hasattr(self, 'nc_data'):
            self.nc_data.close()

        # Reopen the NetCDF file
        try:
            self.nc_data = nc.Dataset(self.nc_file, 'r')
        except Exception as e:
            raise IOError(f"Error opening NetCDF file: {e}")

        # Remap station indices
        self._station_indices = self._map_station_indices()

    def update_gridlist(self, gridlist_df):
        """
        Update the gridlist DataFrame and remap station indices for the NetCDF file.
        
        Replaces the current gridlist with a new one and recalculates the mapping
        between station names and NetCDF array indices. This allows processing
        different subsets of stations from the same NetCDF file.

        Args:
            gridlist_df (pl.DataFrame): New gridlist DataFrame containing station information.
                                       Must include 'station_name', 'lat', 'lon' columns.
                                       
        Raises:
            TypeError: If gridlist_df is not a Polars DataFrame
            ValueError: If station coordinate matching fails with the NetCDF file
        """
        if not isinstance(gridlist_df, pl.DataFrame):
            raise TypeError("gridlist_df must be a Polars DataFrame")
        self.gridlist = gridlist_df

        # Remap station indices based on the new gridlist
        self._station_indices = self._map_station_indices()


    def close(self):
        """
        Safely close the NetCDF file and clean up resources.
        
        Ensures the NetCDF dataset is properly closed and marks it as None
        to prevent further access. Handles cases where the file might
        already be closed or invalid.
        """
        if hasattr(self, 'nc_data') and self.nc_data is not None:
            try:
                if self.nc_data.isopen():
                    self.nc_data.close()
            except Exception:
                # Ignore errors during close (file might already be closed)
                pass
            finally:
                self.nc_data = None  # Mark as closed


    def __enter__(self):
        """
        Enter the context manager for the NetCDF handler.
        
        Returns:
            netcdf_handler: Returns self to enable context manager usage
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and ensure proper cleanup.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()

    def __len__(self):
        """
        Return the number of stations in the current gridlist.
        
        Returns:
            int: Number of stations in the gridlist DataFrame
        """
        return len(self.gridlist)

    def __del__(self):
        """
        Destructor that ensures the NetCDF file is closed when the handler is deleted.
        """
        self.close()

class input_handler:
    
    """
    A simplified input handler for CAETÊ input data
    """
    
    @staticmethod
    def create_gridlist_from_files(finput, lat_var='lat', lon_var='lon', output_path=None):
        """
        Create a gridlist DataFrame from a list of bz2 input files or a NetCDF file.

        Args:
            input_files (list): List of input file paths
            lat_var (str, optional), Default: 'lat': Name of the latitude variable in the NetCDF file
            lon_var (str, optional), Default: 'lon': Name of the longitude variable in the NetCDF file
            output_path (str or Path, optional): Path to save the gridlist CSV file

        Returns:
            pl.DataFrame: Polars DataFrame containing station information
        """
        inp = str_or_path(finput)
        
        if inp.is_file():
            # Manege the expected netCDF file structure
            with nc.Dataset(inp, 'r') as ds:
                # Extract latitude and longitude variables
                lats = ds.variables[lat_var][:].data
                lons = ds.variables[lon_var][:].data
                station_names = ds.variables['station_name'][:]
                station_ids = ds.variables['station'][:].data
                global_y, global_x = np.array([find_indices_xy(ny, nx) for ny, nx in zip(lats, lons)]).T
            
            # Create a Polars DataFrame
            gridlist = pl.DataFrame({
                "station_id": station_ids,
                "lat": lats,
                "lon": lons,
                "station_name": station_names,
                "global_y": global_y,
                "global_x": global_x
            })

        else:
            # Handle directory input
            files = list(inp.glob("input_data*"))
            station_id = list(range(len(files)))
            station_name = ["station_" + f.stem.split("_")[-1] for f in files]
            y, x = np.array(list(map(lambda x: np.array(x.split("-")), [f.stem.split("_")[-1] for f in files])), dtype=int).T
            lat, lon = np.array([find_coordinates_xy(ny, nx) for ny, nx in zip(y,x)]).T

            gridlist = pl.DataFrame({
                "station_id": station_id,
                "lat": lat,
                "lon": lon,
                "station_name": station_name,
                "global_y": y,
                "global_x": x
            })
        
        if output_path is not None:
            gridlist.write_csv(output_path)

        return gridlist


    def __init__(self, fpath, gridlist_df, batch_size=1):
            """
            Initialize the simplified input handler.
            
            Args:
                fpath (str or Path): Path to the input file (NetCDF) or directory (bz2)
                gridlist_df (pl.DataFrame): Polars DataFrame containing station information
                batch_size (int, optional): Number of stations to process per batch. Defaults to 1.
            """
            self.fpath = str_or_path(fpath)
            self.gridlist = gridlist_df
            self.batch_size = batch_size
            self.input_type = cfg.input_handler.input_type.lower()
    
            # Initialize the handler based on input type
            self._handler = None
            self._init_handler()
            self._metadata = self._handler.load_metadata()
    
            # Simple batch tracking
            self._current_batch = 0
            self._total_batches = (len(self.gridlist) + self.batch_size - 1) // self.batch_size


    def _init_handler(self):
        """Initialize the appropriate backend handler."""
        if self.input_type == 'netcdf':
            self._handler = netcdf_handler(self.fpath, self.gridlist)
        elif self.input_type == 'bz2':
            self._handler = bz2_handler(self.fpath, self.gridlist)
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")


    def get_batch_data(self, batch_index):
        """
        Synchronously load data for a specific batch.
        
        Args:
            batch_index (int): Zero-based index of the batch to load
            
        Returns:
            dict: Dictionary containing batch data
        """
        if batch_index < 0 or batch_index >= self._total_batches:
            return None
        start_index = batch_index * self.batch_size
        chunk_size = min(self.batch_size, len(self.gridlist) - start_index)
        batch_gridlist = self.gridlist.slice(start_index, chunk_size)

        # Update handler with batch gridlist
        self._handler.update_gridlist(batch_gridlist)

        # Load data synchronously
        data = self._handler.load_data()

        return {
            'batch_index': batch_index,
            'data': data,
            'gridlist': batch_gridlist,
            'batch_size': len(batch_gridlist)
        }
    

    def update_fpath(self, fpath, gridlist_df=None):
        """
        Update the file path and optionally the gridlist for the input handler.
        
        Args:
            fpath (str or Path): New path to the input file (NetCDF) or directory (bz2)
            gridlist_df (pl.DataFrame, optional): New gridlist DataFrame to use.
                                                 If None, uses existing gridlist.
                                                 
        Raises:
            ValueError: If fpath is empty or handler initialization fails
        """
        if gridlist_df is not None:
            self.update_gridlist(gridlist_df)
        else:
            print("[WARN] No gridlist_df provided, using existing gridlist")
            
        if not fpath:
            raise ValueError("fpath is required")

        # Close current handler
        self.close()
        
        # Update path and reinitialize
        self.fpath = str_or_path(fpath)
        self._init_handler()
        self._metadata = self._handler.load_metadata()
        
        # Recalculate total batches based on current gridlist
        self._current_batch = 0
        self._total_batches = (len(self.gridlist) + self.batch_size - 1) // self.batch_size


    def update_gridlist(self, gridlist_df):
        """
        Update the gridlist DataFrame and remap data accordingly.
        
        Args:
            gridlist_df (pl.DataFrame): New gridlist DataFrame containing station information.
                                       Must include required columns for the handler type.
                                       
        Raises:
            TypeError: If gridlist_df is not a Polars DataFrame
            ValueError: If gridlist validation fails
        """
        if not isinstance(gridlist_df, pl.DataFrame):
            raise TypeError("gridlist_df must be a Polars DataFrame")
            
        self.gridlist = gridlist_df
        
        # Update the underlying handler with the new gridlist
        if self._handler is not None:
            self._handler.update_gridlist(gridlist_df)
            
        # Recalculate total batches
        self._current_batch = 0
        self._total_batches = (len(self.gridlist) + self.batch_size - 1) // self.batch_size


    def update(self, fpath, gridlist_df):
        """
        Convenience method to update both file path and gridlist at once.
        
        Args:
            fpath (str or Path): New path to the input file (NetCDF) or directory (bz2)
            gridlist_df (pl.DataFrame): New gridlist DataFrame to use
        """
        self.update_fpath(fpath, gridlist_df)


    @property
    def get_data(self):
        """
        Load all data at once (no batching).
        
        Returns:
            dict: Dictionary with station names as keys and data as values
        """
        return self._handler.load_data()


    @property
    def get_metadata(self):
        """Get metadata from the handler."""
        return self._handler.load_metadata()


    @property
    def total_batches(self):
        """Get total number of batches."""
        return self._total_batches


    def close(self):
        """Close the handler."""
        if self._handler is not None:
            if hasattr(self._handler, 'close'):
                self._handler.close()
            self._handler = None


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def __del__(self):
        try:
            self.close()
        except:
            pass


if __name__ == "__main__":
    import time


    # df = gridlist_df.slice(0, 128)  # Use a slice for testing
    cfg.input_handler.input_type = 'bz2' 
    input_dir = '../input/20CRv3-ERA5/obsclim_test'
    gridlist_path = '../grd/gridlist_test.csv'
    gridlist_df = pl.read_csv(gridlist_path)
    ih = input_handler(input_dir, gridlist_df, batch_size=cfg.multiprocessing.max_processes)
    
    cfg.input_handler.input_type = 'netcdf'  # Set input type to netcdf for testing
    nc_file_path = '../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc'
    gridlist_path = '../grd/gridlist_test.csv'
    gridlist_df = pl.read_csv(gridlist_path)
    ih2 = input_handler(nc_file_path, gridlist_df, batch_size=cfg.multiprocessing.max_processes) 

    # # Example usage of netcdf_handler
    # nc_file_path = '../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc'
    # s1 = time.perf_counter()
    # handler = netcdf_handler(nc_file_path, df)
    #     # Get data for all stations
    # data_nc = handler.load_data()
    # metadata_nc = handler.load_metadata()
    # s2 = time.perf_counter() - s1
    # print(f"netcdf_handler read data in {s2:.2f} seconds with {len(data_nc)} stations")

    # # Example usage of bz2_handler
    # input_dir = '../input/20CRv3-ERA5/obsclim'
    # s1 = time.perf_counter()
    # bz = bz2_handler(input_dir, df)
    # data_bz = bz.load_data()
    # metadata_bz = bz.load_metadata()
    # s2 = time.perf_counter() - s1
    # print(f"bz2_handler read data in {s2:.2f} seconds with {len(data_bz)} stations")

    # # Example usage of netcdf_handler
    # nc_file_path = '../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc'
    # s1 = time.perf_counter()
    # with netcdf_handler(nc_file_path, df) as handler:
    #     # Get data for all stations
    #     data_nc = handler.load_data()
    #     metadata_nc = handler.load_metadata()
    # s2 = time.perf_counter() - s1
    # print(f"netcdf_handler read data in {s2:.2f} seconds with {len(data_nc)} stations")

    # Example usage of netcdf_handler with chunking
    # inc = 128
    # total_rows = len(gridlist_df)

    # # Calculate how many chunks we need
    # num_chunks = (total_rows + inc - 1) // inc  # This is equivalent to math.ceil(total_rows / inc)

    # for x in range(1): # Change range(1) to range(num_chunks) to process all chunks
    #     start_index = x * inc
    #     # For the last chunk, make sure we don't go beyond the total rows
    #     chunk_size = min(inc, total_rows - start_index)

    #     df = gridlist_df.slice(start_index, chunk_size)

    #     print(f"Processing chunk {x}: rows {start_index} to {start_index + chunk_size - 1}")
    #     print(f"Chunk size: {len(df)}")

    #     with netcdf_handler(nc_file_path, df) as handler:
    #         # Get data for all stations
    #         data = handler.load_data()
    #         metadata = handler.load_metadata()

    #         print("Data loaded successfully.")
    #         print(f"Loaded {len(data)} stations.")
    #         print("Time Metadata:", metadata)



# # ## TEST input_handler class
#     if cfg.input_handler.input_type == 'netcdf':
#         start = time.perf_counter()

#         with input_handler(nc_file_path, gridlist_df, batch_size=128) as handler:
#             # handler = input_handler(nc_file_path, gridlist_df, batch_size=128)

#             mtdt_nc = handler.get_metadata()
#             # Process all batches
#             for batch in handler.batch_generator():
#                 print(f"Processing batch {batch['batch_index'] + 1}/{handler.total_batches}")
#                 print(f"Batch size: {batch['batch_size']}")
#                 # Process batch['data'] and batch['metadata']
#                 # print(batch['data'])  # Example processing step
#                 # print(batch['metadata'])  # Example processing step

#             print("All batches processed.")
#             print(f"Total batches: {handler.total_batches}")
#             print("Changing file path and updating gridlist...")
#             handler.update(nc_file_path, df)
#             mtdt_nc_updated = handler.get_metadata()
#             for batch in handler.batch_generator():
#                 print(f"Processing batch {batch['batch_index'] + 1}/{handler.total_batches}")
#                 print(f"Batch size: {batch['batch_size']}")
#                 # Process batch['data'] and batch['metadata']
#                 # print(batch['data'])  # Example processing

#             end = time.perf_counter() - start
#             print(f"input_handler with NetCDF took {end:.2f} seconds to process {len(handler.gridlist)} stations.")

#     elif cfg.input_handler.input_type == 'bz2':
#         start = time.perf_counter()
#         handler = input_handler(input_dir, gridlist_df, batch_size=128)

#         # Process all batches
#         mtdt_bz = handler.get_metadata()
#         for batch in handler.batch_generator():
#             print(f"Processing batch {batch['batch_index']}/{handler.total_batches}")
#             print(f"Batch size: {batch['batch_size']}")
#             # Process batch['data'] and batch['metadata']
#             # print(batch['data'])  # Example processing step
#             # print(batch['metadata'])  # Example processing step

#         print("All batches processed.")
#         print(f"Total batches: {handler.total_batches}")
#         print("Changing file path and updating gridlist...")
#         handler.update(input_dir, df)
#         mtdt_bz = handler.get_metadata()
#         for batch in handler.batch_generator():
#             print(f"Processing batch {batch['batch_index']}/{handler.total_batches}")
#             print(f"Batch size: {batch['batch_size']}")
#             # Process batch['data'] and batch['metadata']
#             # print(batch['data'])  # Example processing step
#             # print(batch['metadata'])  # Example processing step
#         end = time.perf_counter() - start
#         print(f"input_handler with bz2 took {end:.2f} seconds to process {len(handler.gridlist)} stations.")
