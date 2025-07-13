#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NetCDF Handler
-------------
A class to read NetCDF files for CAETE model input.
"""

import numpy as np
import netCDF4 as nc
import polars as pl


class NetCDFHandler:
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
        Initialize the NetCDFHandler with the NetCDF file path and a pre-loaded gridlist DataFrame.

        Args:
            nc_file_path (str): Path to the NetCDF file
            gridlist_df (polars.DataFrame): Pre-loaded gridlist as a Polars DataFrame
        """
        self.nc_file = nc_file_path

        # Open the NetCDF file with optimal chunking for faster access
        try:
            self.nc_data = nc.Dataset(self.nc_file, 'r')
        except Exception as e:
            raise IOError(f"Error opening NetCDF file: {e}")

        # Store the gridlist DataFrame
        if not isinstance(gridlist_df, pl.DataFrame):
            raise TypeError("gridlist_df must be a Polars DataFrame")
        self.gridlist = gridlist_df

        # Store variable names for quick reference
        self.time_vars = ['hurs', 'tas', 'ps', 'pr', 'rsds', 'sfcwind', 'vpd']
        self.station_vars = ['tn', 'tp', 'ap', 'ip', 'op']

        # Pre-load station indices and prepare for data loading
        self._station_indices = self._map_station_indices()

    def _map_station_indices(self):
        """
        Create a mapping between station names and their indices in the netCDF file.

        Returns:
            dict: Dictionary with station names as keys and netCDF indices as values
        """
        # Get station names from gridlist
        station_names = self.gridlist.select('station_name').to_series().to_list()

        # Check if station_id exists in gridlist to map to netCDF indices
        if 'station_id' in self.gridlist.columns:
            # Create mapping from station name to station_id (which should match netCDF index)
            station_indices = dict(zip(
                station_names,
                self.gridlist.select('station_id').to_series().to_list()
            ))
        else:
            # If no station_id, assume the order in gridlist matches netCDF
            # but verify using station_name if it exists in netCDF
            station_indices = {}

            # Try to read station_name from netCDF for validation
            if 'station_name' in self.nc_data.variables:
                nc_station_names = self._get_netcdf_station_names()

                for station_name in station_names:
                    try:
                        # Find the index of this station in the netCDF file
                        idx = nc_station_names.index(station_name)
                        station_indices[station_name] = idx
                    except ValueError:
                        # Station not found in netCDF, skip it
                        continue
            else:
                # If no station name in netCDF, assume row index matches
                for i, station_name in enumerate(station_names):
                    station_indices[station_name] = i

        return station_indices

    def load_data(self):
        """
        Load data from the NetCDF file into a dictionary.

        Returns:
            dict: A dictionary with station names as keys and variable data as values

        Note:
            This method optimizes performance by loading all data at once from the netCDF file
            and then distributing it to individual station dictionaries. This avoids repeated
            file access operations which are slow.
        """
        data_dict = {}
        station_names = list(self._station_indices.keys())

        if not station_names:
            return data_dict

        # Get all the indices we need to extract from the netCDF file
        indices = np.array(list(self._station_indices.values()))

        # Pre-load all time-series variables into memory at once
        time_vars_data = {}
        for var_name in self.time_vars:
            if var_name in self.nc_data.variables:
                # Read all station data at once using fancy indexing
                # This is much faster than reading each station individually
                var_data = self.nc_data.variables[var_name][:, indices].astype(np.float32)

                # Convert masked arrays to regular arrays if needed
                if isinstance(var_data, np.ma.MaskedArray):
                    var_data = var_data.filled()

                # Store for later distribution
                time_vars_data[var_name] = var_data

        # Pre-load all station variables into memory at once
        station_vars_data = {}
        for var_name in self.station_vars:
            if var_name in self.nc_data.variables:
                # Read all station data at once
                var_data = self.nc_data.variables[var_name][indices].astype(np.float32)

                # Convert masked arrays to regular arrays if needed
                if isinstance(var_data, np.ma.MaskedArray):
                    var_data = var_data.filled()

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

    def _get_netcdf_station_names(self):
        """
        Extract and convert station names from netCDF file to a list of strings.
        Caches the result for better performance on repeated calls.

        Returns:
            list: List of station names from netCDF file
        """
        # Check if we've already cached the station names
        if hasattr(self, '_station_names_cache'):
            return self._station_names_cache

        if 'station_name' not in self.nc_data.variables:
            self._station_names_cache = []
            return []

        # Get the station_name variable
        station_name_var = self.nc_data.variables['station_name']

        # NetCDF stores character arrays as multi-dimensional arrays
        # Convert to a list of strings
        if len(station_name_var.shape) == 2:
            # For 2D character arrays (common in netCDF)
            # Load entire array at once for better performance
            char_array = station_name_var[:].astype(str)

            stations = []
            for i in range(char_array.shape[0]):
                # Convert char array to string and strip any padding
                name = ''.join(char_array[i, :]).strip()
                stations.append(name)

            self._station_names_cache = stations
            return stations
        else:
            # For 1D string arrays (less common)
            names = [str(name).strip() for name in station_name_var[:]]
            self._station_names_cache = names
            return names

    def load_metadata(self):
        """
        Load metadata from the NetCDF file.

        Returns:
            tuple: A tuple with three dictionaries containing time, latitude, and longitude metadata
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

        # Extract latitude metadata
        lat_meta = {}
        lat_var = self.nc_data.variables['lat']

        # Load latitude data all at once for better performance
        lat_data = lat_var[:]

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

        # Extract longitude metadata
        lon_meta = {}
        lon_var = self.nc_data.variables['lon']

        # Load longitude data all at once for better performance
        lon_data = lon_var[:]

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

    def close(self):
        """
        Close the NetCDF file.
        """
        if hasattr(self, 'nc_data'):
            self.nc_data.close()

    def __enter__(self):
        """
        Support for context manager protocol.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol.
        """
        self.close()

    def __len__(self):
        """
        Return the number of stations in the gridlist.
        """
        return len(self.gridlist)

    def __getitem__(self, key):
        """
        Allow dictionary-like access to station data.

        Args:
            key (str): Station name or list of station names

        Returns:
            dict: Data for the specified station(s)
        """
        # Handle a list of station names
        if isinstance(key, list):
            return {k: self[k] for k in key if k in self._station_indices}

        # Handle a single station name
        if key not in self._station_indices:
            raise KeyError(f"Station '{key}' not found in gridlist")

        nc_idx = self._station_indices[key]
        station_data = {}

        # Bulk load all time variables for this station at once
        # First create a list of variables we need to load
        time_vars_to_load = [var for var in self.time_vars if var in self.nc_data.variables]

        if time_vars_to_load:
            # Use a more efficient approach by loading all variables at once
            # This reduces file access operations
            for var_name in time_vars_to_load:
                var_data = self.nc_data.variables[var_name][:, nc_idx].astype(np.float32)
                if isinstance(var_data, np.ma.MaskedArray):
                    var_data = var_data.filled()
                station_data[var_name] = var_data

        # Bulk load all station variables for this station at once
        station_vars_to_load = [var for var in self.station_vars if var in self.nc_data.variables]

        if station_vars_to_load:
            # Load all station variables for this index at once if possible
            for var_name in station_vars_to_load:
                var_data = self.nc_data.variables[var_name][nc_idx]
                if isinstance(var_data, np.ma.MaskedArray):
                    var_data = float(var_data.filled())
                else:
                    var_data = float(var_data)
                station_data[var_name] = var_data

        return station_data




if __name__ == "__main__":
    nc_file_path = '../input/20CRv3-ERA5/obsclim/caete_input_20CRv3-ERA5_obsclim.nc'
    gridlist_path = '../input/20CRv3-ERA5/obsclim/gridlist_caete.csv'

    # Load gridlist using Polars
    gridlist_df = pl.read_csv(gridlist_path)
    gridlist_df = gridlist_df.slice(0, 50)[::2]  # Limit to first 100 rows for testing

    # Example of how you might preprocess the gridlist before passing it to the handler
    # For example, filter to only include certain stations:
    # gridlist_df = gridlist_df.filter(pl.col("lat") > 0)

    with NetCDFHandler(nc_file_path, gridlist_df) as handler:
        # Get data for all stations
        data = handler.load_data()
        time_meta, lat_meta, lon_meta = handler.load_metadata()

        print("Data loaded successfully.")
        print(f"Loaded {len(data)} stations.")
        print("Time Metadata:", time_meta)
