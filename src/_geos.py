# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """

from typing import Tuple


import numpy as np

from numba import jit # type: ignore
import pyproj

try:
    from config import fetch_config
except ImportError:
    from .config import fetch_config

config = fetch_config()
datum = config.crs.datum # type: ignore

#TODO:  find_indices and find_coordinates should raise errors in case of improper inputs. 

# WARN: find_indices and find_coordinates precision is limited to 7 decimal places.

def calculate_area(center_lat:float, center_lon:float, dx:float=0.5, dy:float=0.5, datum: str = datum)->float: # type: ignore
    """Calculates the area of a cell on the Earth's surface given the center coordinates and the cell resolution
    using a geographic coordinate system with the datum provided.

    Args:
        center_lat (float): Center latitude, degrees North
        center_lon (float): Center longitude, degrees East
        dx (float): Cell resolution in the x-direction, degrees
        dy (float): Cell resolution in the y-direction, degrees
        datum (str, optional): Datum used for the calculation. Defaults to "WGS84"

    Returns:
        float: Area of the grid cell in square meters
    """

    # do not allow negative dx and dy and crappy center_lat and center_lon
    assert dx > 0 and dy > 0, "dx and dy must be positive"
    assert center_lat >= -90 + (dy/2) and center_lat <= 90 - (dy/2), f"center_lat must match the resolution. Expected range: {-90+(dy/2)},{90-(dy/2)}"
    assert center_lon >= -180 + (dx/2) and center_lon <= 180 - (dx/2), f"center_lon must match the resolution. Expected range:{-180+(dx/2)},{180-(dx/2)}"

    # Define a geographic coordinate system with WGS84 datum
    geod = pyproj.Geod(ellps=datum)

    # Define the lower left and upper right coordinates
    ll_lon, ll_lat = center_lon - dx / 2, center_lat - dy / 2
    ur_lon, ur_lat = center_lon + dx / 2, center_lat + dy / 2

    # Calculate the length of the top and bottom edges of the cell
    _, _, top_edge_length = geod.inv(ll_lon, ur_lat, ur_lon, ur_lat)
    _, _, bottom_edge_length = geod.inv(ll_lon, ll_lat, ur_lon, ll_lat)

    # Calculate the length of the left and right edges of the cell
    _, _, left_edge_length = geod.inv(ll_lon, ll_lat, ll_lon, ur_lat)
    _, _, right_edge_length = geod.inv(ur_lon, ll_lat, ur_lon, ur_lat)

    # Calculate the area of the cell
    area = 0.25 * (top_edge_length + bottom_edge_length) * (left_edge_length + right_edge_length)

    return area

@jit(nopython=True, cache=True)
def calc_min_rounding_log(res: float) -> int:
    """Calculate minimum rounding digits needed for a given resolution using logarithms."""
    if res >= 1.0:
        return 2  # minimum for coordinate work
    else:
        # Use logarithms to find decimal places
        # log10(res) gives us the order of magnitude
        # For res = 0.1, log10(0.1) = -1, so we need 1 decimal place
        # For res = 0.01, log10(0.01) = -2, so we need 2 decimal places
        decimal_places = max(1, int(-np.log10(res)) + 1)

        # Add safety margin based on resolution magnitude
        if res < 0.01:
            safety_margin = 6  # Very fine resolutions
        elif res < 0.1:
            safety_margin = 5  # Fine resolutions  
        else:
            safety_margin = 4  # Standard resolutions
            
        return max(2, decimal_places + safety_margin)  # +2 for safety margin
    

@jit(nopython=True, cache=True) # type: ignore
def find_indices_xy(N: float, W: float, res_y: float = 0.5, res_x: float = 0.5, rounding: int = 2) -> Tuple[int, int]:
    """
    It finds the indices for a given latitude and longitude in a planar grid.
    WARNING Feeding the function with lat/long outside the boundaries
    (-180 - 180; -90 - 90) will cause the function to return invalid indices.

    Warning: This function assumes that the latitude (N) is given in decimal degrees north
    and longitude (W) in decimal degrees east.

    Find coordinates snaps to cell center.

    Args:
        N (float): latitude in decimal degrees north
        W (float): longitude in decimal degrees west
        res_y (float, optional): grid resolution for y-axis. Defaults to 0.5 degrees.
        res_x (float, optional): grid resolution for x-axis. Defaults to 0.5 degrees.
        rounding (int, optional): decimal significant digits. Defaults to 2.

    Returns:
        tuple[int, int]: (y, x) indices for the given latitude and longitude
        in the grid. (0,0) is the upper left corner.
    """

    if res_y <= 0 or res_x <= 0:
        return -1, -1
    
    # # Calculate minimum required rounding and use the maximum
    # min_rounding = max(calc_min_rounding_log(res_y), calc_min_rounding_log(res_x))
    # effective_rounding = max(rounding, min_rounding)

    rounding_y = calc_min_rounding_log(res_y)
    rounding_x = calc_min_rounding_log(res_x)

    effective_rounding_y = max(rounding, rounding_y)
    effective_rounding_x = max(rounding, rounding_x)


    Yc:float = round(N, effective_rounding_y)
    Xc:float = round(W, effective_rounding_x)


    half_res_y:float = res_y / 2
    half_res_x:float = res_x / 2
    Ymin:float = -90 + half_res_y
    Xmin:float = -180 + half_res_x

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res_x) # type: ignore
    lat = np.arange(Ymin, 90, res_y) # type: ignore

    # Find indices for Yc and Xc using bisection
    # Yc is negative because our origin (90 degrees north) is in the upper left corner of the matrix.
    # 
    Yind:int = np.searchsorted(lat, -Yc - half_res_y, side='left') #type: ignore
    Xind:int = np.searchsorted(lon, Xc - half_res_x, side='left') #type: ignore

    if Yc > 90 or Yc < -90:
        Yind = -1
    if Xc < -180 or Xc > 180:
        Xind = -1

    return Yind, Xind


@jit(nopython=True, cache=True) # type: ignore
def find_indices(N: float, W: float, res: float = 0.5, rounding: int = 2) -> Tuple[int, int]:
    """
    It finds the indices for a given latitude and longitude in a planar grid.
    WARNING Feeding the function with lat/long outside the boundaries
    (-180 - 180; -90 - 90) will cause the function to return invalid indices.

    Args:
        N (float): latitude in decimal degrees north
        W (float): longitude in decimal degrees west
        res (float, optional): grid resolution. Defaults to 0.5 degrees.
        rounding (int, optional): decimal significant digits. Defaults to 2.

    Returns:
        tuple[int, int]: (y, x) indices for the given latitude and longitude
        in the grid. (0,0) is the upper left corner.
    """
    if res <= 0:
        return -1, -1
    
    # Calculate minimum required rounding and use the maximum
    min_rounding = calc_min_rounding_log(res)
    effective_rounding = max(rounding, min_rounding)
    
    Yc:float = round(N, effective_rounding)
    Xc:float = round(W, effective_rounding)
    half_res:float = res / 2
    Ymin:float = -90 + half_res
    Xmin:float = -180 + half_res

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res) # type: ignore
    lat = np.arange(Ymin, 90, res) # type: ignore


    # Find indices for Yc and Xc using searchsorted
    Yind:int = np.searchsorted(lat, -Yc - half_res, side='left') #type: ignore
    Xind:int = np.searchsorted(lon, Xc - half_res, side='left') #type: ignore

    if Yc > 90 or Yc < -90:
        Yind = -1
    if Xc < -180 or Xc > 180:
        Xind = -1

    return Yind, Xind


@jit(nopython=True, cache=True) # type: ignore
def find_coordinates_xy(Yind: int, Xind: int, res_y: float = 0.5, res_x: float = 0.5, rounding: int = 2) -> Tuple[float, float]:
    """
    It finds the latitude and longitude (cell center) for given (Yind, Xind) indices in a planar grid.
    WARNING: This function do not check if the indices are valid.

    Args:
        Yind (int): y index in the grid
        Xind (int): x index in the grid
        res_y (float, optional): grid resolution for y-axis. Defaults to 0.5 degrees.
        res_x (float, optional): grid resolution for x-axis. Defaults to 0.5 degrees.
        rounding (int, optional): decimal significant digits. Defaults to 2.

    Returns:
        tuple[float, float]: (N, W) latitude and longitude for the given indices.
        Feeding the function with indices outside the boundaries of the grid
        will cause the function to return invalid latitude and longitude.
    """

    half_res_y:float = res_y / 2
    half_res_x:float = res_x / 2
    Ymin:float = -90 + half_res_y
    Xmin:float = -180 + half_res_x

    # min_rounding = min(calc_min_rounding_log(res_y), calc_min_rounding_log(res_x))
    rounding_y = calc_min_rounding_log(res_y)
    rounding_x = calc_min_rounding_log(res_x)

    effective_rounding_y = max(rounding, rounding_y)
    effective_rounding_x = max(rounding, rounding_x)

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res_x) # type: ignore
    lat = np.arange(Ymin, 90, res_y)[::-1] # type: ignore

    # Find latitude and longitude for Yind and Xind
    N = round(lat[Yind], effective_rounding_y)
    W = round(lon[Xind], effective_rounding_x)

    return N, W


@jit(nopython=True, cache=True) # type: ignore
def find_coordinates(Yind: int, Xind: int, res: float = 0.5, rounding: int = 2) -> Tuple[float, float]:
    """It finds the latitude and longitude (cell center) for given (Yind, Xind) indices in a planar grid
    WARNING: This function do not check if the indices are valid.

    Args:
        Yind (int): y index in the grid
        Xind (int): x index in the grid
        res (float, optional): grid resolution. Defaults to 0.5 degrees.
        rounding (int, optional): decimal significant digits. Defaults to 2.

    Returns:
        tuple[float, float]: (N, W) latitude and longitude for the given indices.
        Feeding the function with indices outside the boundaries of the grid
        will cause the function to return invalid latitude and longitude.
    """
    min_rounding = calc_min_rounding_log(res)
    effective_rounding = max(rounding, min_rounding)
    half_res:float = res / 2
    Ymin:float = -90 + half_res
    Xmin:float = -180 + half_res

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res) # type: ignore
    lat = np.arange(Ymin, 90, res)[::-1] # type: ignore

    # Find latitude and longitude for Yind and Xind

    # Find latitude and longitude for Yind and Xind
    N = round(lat[Yind], effective_rounding)
    W = round(lon[Xind], effective_rounding)

    return N, W


def define_region(north:float, south:float,
                  west:float,  east:float,
                  res_y:float=0.5,
                  res_x:float=0.5,
                  rounding:int=2):
    """define a bounding box for a given region of interest

    Args:
        north (float): Northernmost latitude. Units: degrees north.
        south (float): Southernmost latitude. Units: degrees north.
        west (float): Westernmost longitude. Units: degrees east.
        east (float): Easternmost longitude. Units: degrees east.
        res (float, optional): grid resolution. Defaults to 0.5 decimal degrees.
        rounding (int, optional): by default, coordinates are rounded.
        This sets the number of decimal digits. Defaults to 2.

    Returns:
        dict: bbox dictionary with keys ymin, ymax, xmin, xmax
    """
    ymin, xmin = find_indices_xy(north, west, res_y, res_x, rounding)
    ymax, xmax = find_indices_xy(south, east, res_y, res_x, rounding)

    return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


def get_region(region): # type: ignore
    """
    Get bounding box for a region of interest
    """
    return region["ymin"], region["ymax"], region["xmin"], region["xmax"] # type: ignore


# def latitude_axis(north: float | int, south: float | int, yres: float | int) -> np.ndarray: # type: ignore
#     """
#     Generate a latitude axis for a given region
#     """
#     return np.arange(north, south, -yres) + yres / 2 # type: ignore


# def longitude_axis(west: float | int, east: float | int, xres: float | int) -> np.ndarray: # type: ignore
#     """
#     Generate a longitude axis for a given region
#     """
#     return np.arange(west, east, xres) + xres / 2 # type: ignore

# def get_axis(bbox): # type: ignore
#     """
#     Get latitude and longitude axis for a region of interest
#     """
#     lat = latitude_axis(bbox["north"], bbox["south"], bbox["res_y"]) # type: ignore
#     lon = longitude_axis(bbox["west"], bbox["east"], bbox["res_x"]) # type: ignore
#     return lat, lon # type: ignore


# Bbox
pan_amazon_bbox = {"north":10.5, # type: ignore
                   "south":-21.5,
                   "west":-80.0,
                   "east":-43.0,
                   "res_y":config.crs.yres, #type: ignore
                   "res_x":config.crs.xres, #type: ignore
                   "rounding":2}  # type: ignore

global_bbox = {"north":90.0,  # type: ignore
               "south":-90.0,
               "west":-180.0,
               "east":180.0,
                "res_y":config.crs.yres, #type: ignore
                "res_x":config.crs.xres, #type: ignore
                "rounding":2}

# Regions
global_region = define_region(**global_bbox)  # type: ignore
pan_amazon_region = define_region(**pan_amazon_bbox) # type: ignore


if __name__ == "__main__":
    import unittest

    class TestFunctions(unittest.TestCase):
        def test_calculate_area(self):
            # Test with known values
            # Compare with some external data source:
            # https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
            # Look at the first answer

            self.assertAlmostEqual(calculate_area(0, 0, 0.5, 0.5) * 1e-6, 3077.2300079, delta=0.1)
            self.assertAlmostEqual(calculate_area(0.75, 0, 0.5, 0.5) * 1e-6, 3077.0019391, delta=0.1)
            self.assertAlmostEqual(calculate_area(-0.75, 0, 0.5, 0.5) * 1e-6, 3077.0019391, delta=0.1)
            self.assertAlmostEqual(calculate_area(-89.75, 0.0, 0.5, 0.5) * 1e-6, 13.608615243, delta=0.1)
            self.assertAlmostEqual(calculate_area(0, 0, 0.25, 0.25) * 1e-6, 3077.2300079/4, delta=0.1)
            self.assertAlmostEqual(calculate_area(0.75, 0, 0.25, 0.25) * 1e-6, 3077.0019391/4, delta=0.1)
            self.assertAlmostEqual(calculate_area(-0.75, 0, 0.25, 0.25) * 1e-6, 3077.0019391/4, delta=0.1)
            self.assertAlmostEqual(calculate_area(-89.75, 0.0, 0.25, 0.25) * 1e-6, 13.608615243/4, delta=0.1)

            # Test with invalid inputs
            with self.assertRaises(AssertionError):
                calculate_area(-91, 0, 1, 1)
                calculate_area(91, 0, 1, 1)
                calculate_area(0, -181, 1, 1)
                calculate_area(0, 181, 1, 1)
                calculate_area(0, 0, -1, 1)
                calculate_area(0, 0, 1, -1)


        def test_find_indices(self):

            # Test with invalid inputs
            # Resolution must be positive and lat/lon must be within the boundaries
            self.assertEqual(find_indices_xy(91, 0, 1, 1), (-1, 179))
            self.assertEqual(find_indices_xy(-91, 0, 1, 1), (-1, 179))
            self.assertEqual(find_indices_xy(0, 181, 1, 1), (89, -1))
            self.assertEqual(find_indices_xy(0, -181, 1, 1), (89, -1))
            self.assertEqual(find_indices_xy(0, 0, -1, 1), (-1, -1))
            self.assertEqual(find_indices_xy(0, 0, 1, -1), (-1, -1))
            self.assertEqual(find_indices_xy(0, 0, 0, 0), (-1, -1))
            
            # XY versions must match the non-XY versions
            # find_indices and find_indices_xy should return the same values
            self.assertEqual(find_indices_xy(45, 45, 1, 1), find_indices(45, 45, 1))
            self.assertEqual(find_indices_xy(0, 0, 0.25, 0.25), find_indices(0, 0, 0.25))
            self.assertEqual(find_indices_xy(45, 45, 0.25, 0.25), find_indices(45, 45, 0.25))
            self.assertEqual(find_indices_xy(0, 0, 0.5, 0.5), find_indices(0, 0, 0.5))
            self.assertEqual(find_indices_xy(-2.63, 0, 0.5, 0.5), find_indices(-2.63, 0, 0.5))
            self.assertEqual(find_indices_xy(-2.63, 0, 1/12, 1/12), find_indices(-2.63, 0, 1/12))
            self.assertEqual(find_indices_xy(-2.63, 0, 1/1200, 1/1200), find_indices(-2.63, 0, 1/1200))
            
            # the same for find_coordinates and find_coordinates_xy
            self.assertEqual(find_coordinates_xy(0, 0, 1, 1), find_coordinates(0, 0, 1))
            self.assertEqual(find_coordinates_xy(180, 45, 1, 1), find_coordinates(180, 45, 1))
            self.assertEqual(find_coordinates_xy(30, 0, 1, 1), find_coordinates(30, 0, 1))
            self.assertEqual(find_coordinates_xy(45, 45, 1/12, 1/12), find_coordinates(45, 45, 1/12))
            self.assertEqual(find_coordinates_xy(45, 45, 1/120, 1/120), find_coordinates(45, 45, 1/120))
            self.assertEqual(find_coordinates_xy(45, 45, 1/1200, 1/1200), find_coordinates(45, 45, 1/1200))
            
            # find coordinates and find coordinates_xy cross test
            self.assertEqual(find_coordinates_xy(*find_indices_xy(0, 0, 0.5, 0.5), 0.5, 0.5),
                             find_coordinates(*find_indices(0, 0, 0.5), 0.5))
            
            # find coordinates and find coordinates_xy must return the same values
            self.assertEqual(find_coordinates_xy(*find_indices_xy(45, 45, 1, 1), 1, 1),
                             find_coordinates(*find_indices(45, 45, 1), 1))
            

            # outputs of find_coordinates feeded with the outputs of find_indices should be the same as the inputs of find_indices
            # Find coordinates snaps to cell center, so we use res/2 as delta
            def check_crossed_results(lat=90, lon=180, res=0.5):  # type: ignore
                y, x = find_indices(lat, lon, res)
                lat1, lon1 = find_coordinates(y, x, res)
                # Find coordinates snaps to cell center, so we use res/2 as delta
                self.assertAlmostEqual(lat1, lat, delta=res/2.0)
                self.assertAlmostEqual(lon1, lon, delta=res/2.0)

            check_crossed_results()
            check_crossed_results(0, 0, 0.5)
            check_crossed_results(45, 45, 1)
            check_crossed_results(0, 0, 0.25)
            check_crossed_results(45, 45, 0.25)
            check_crossed_results(90, 180, 0.5)
            check_crossed_results(-90, -180, 0.5)
            check_crossed_results(-90, 180, 0.5)
            check_crossed_results(90, -180, 0.01)

            def check_crossed_results_xy(lat=90, lon=180, res=0.5):  # type: ignore
                y, x = find_indices_xy(lat, lon, res, res)
                lat1, lon1 = find_coordinates_xy(y, x, res, res)
                # Find coordinates snaps to cell center, so we use res/2 as delta
                self.assertAlmostEqual(lat1, lat, delta=res/2.0)
                self.assertAlmostEqual(lon1, lon, delta=res/2.0)

            check_crossed_results_xy()
            check_crossed_results_xy(0, 0, 0.5)
            check_crossed_results_xy(45, 45, 1)
            check_crossed_results_xy(0, 0, 0.25)
            check_crossed_results_xy(45, 45, 0.25)
            check_crossed_results_xy(90, 180, 0.5)
            check_crossed_results_xy(-90, -180, 0.5)
            check_crossed_results_xy(-90, 180, 0.5)
            check_crossed_results_xy(90, -180, 0.01)


            def check_crossed_results_crossed_xy(lat=90, lon=180, res=0.5):  # type: ignore
                y, x = find_indices(lat, lon, res)
                lat1, lon1 = find_coordinates_xy(y, x, res, res)
                # Find coordinates snaps to cell center, so we use res/2 as delta
                self.assertAlmostEqual(lat1, lat, delta=res/2.0)
                self.assertAlmostEqual(lon1, lon, delta=res/2.0)

            check_crossed_results_crossed_xy()
            check_crossed_results_crossed_xy(0, 0, 0.5)
            check_crossed_results_crossed_xy(45, 45, 1)
            check_crossed_results_crossed_xy(0, 0, 0.25)
            check_crossed_results_crossed_xy(45, 45, 0.25)
            check_crossed_results_crossed_xy(90, 180, 0.5)
            check_crossed_results_crossed_xy(-90, -180, 0.5)
            check_crossed_results_crossed_xy(-90, 180, 0.5)
            check_crossed_results_crossed_xy(90, -180, 0.01)

            def check_crossed_results_crossed_xy_2(lat=90, lon=180, res=0.5):  # type: ignore
                y, x = find_indices_xy(lat, lon, res, res)
                lat1, lon1 = find_coordinates(y, x, res)
                # Find coordinates snaps to cell center, so we use res/2 as delta
                self.assertAlmostEqual(lat1, lat, delta=res/2.0)
                self.assertAlmostEqual(lon1, lon, delta=res/2.0)

            check_crossed_results_crossed_xy_2()
            check_crossed_results_crossed_xy_2(0, 0, 0.5)
            check_crossed_results_crossed_xy_2(45, 45, 1)
            check_crossed_results_crossed_xy_2(0, 0, 0.25)
            check_crossed_results_crossed_xy_2(45, 45, 0.25)
            check_crossed_results_crossed_xy_2(90, 180, 0.5)
            check_crossed_results_crossed_xy_2(-90, -180, 0.5)
            check_crossed_results_crossed_xy_2(-90, 180, 0.5)
            check_crossed_results_crossed_xy_2(90, -180, 0.01)


        def test_find_coordinates(self):

            # Test with known values
            res = 0.5
            rounding = calc_min_rounding_log(res)
            lat = 90 - 0.5/2
            lon = -180 + 0.5/2
            y, x = find_indices_xy(lat, lon, res, res, rounding)
            self.assertAlmostEqual(find_coordinates_xy(y, x, res, res, rounding), (lat, lon), delta=0.001)

            xres = 0.25
            yres = 0.5
            rounding = 2
            lat = 90 - yres/2
            lon = 180 - xres/2
            y, x = find_indices_xy(lat, lon, yres, xres, rounding)
            self.assertAlmostEqual(find_coordinates_xy(y, x, yres, xres, rounding), (lat, lon))

            res =  0.5
            rounding = calc_min_rounding_log(res)
            lat = 90 - 0.5/2
            lon = 180 - 0.5/2
            y, x = find_indices(lat, lon, res, rounding)
            self.assertAlmostEqual(find_coordinates(y, x, res, rounding), (lat, lon))

            res = 0.25
            rounding = calc_min_rounding_log(res)
            lat = 90 - 0.25/2
            lon = -180 + 0.25/2
            y, x = find_indices(lat, lon, res, rounding)
            self.assertAlmostEqual(find_coordinates(y, x, res, rounding), (lat, lon))

            res = 0.25
            rounding = calc_min_rounding_log(res)
            lat = 90 - 0.25/2
            lon = 180 - 0.25/2
            y, x = find_indices(lat, lon, res, rounding)
            self.assertAlmostEqual(find_coordinates(y, x, res, rounding), (lat, lon))

            # Extended tests with larger range of resolutions
            # Test with various square resolutions
            for res in [1/360, 1/90, 1/30, 1/12, 0.0025, 0.008, 0.05, 0.1, 0.125, 0.2, 1.0, 2.0, 5.0]:
                rounding = calc_min_rounding_log(res)
                # Test corner coordinates (top-left)
                lat = 90 - res/2
                lon = -180 + res/2
                y, x = find_indices(lat, lon, res, rounding)
                a, b = find_coordinates(y, x, res, rounding)
                self.assertAlmostEqual(a, lat, delta=0.0000001)
                self.assertAlmostEqual(b, lon, delta=0.0000001)
                
                # Test corner coordinates (bottom-right)  
                lat = -90 + res/2
                lon = 180 - res/2
                y, x = find_indices(lat, lon, res, rounding)
                a, b = find_coordinates(y, x, res, rounding)
                self.assertAlmostEqual(a, lat, delta=0.0001)
                self.assertAlmostEqual(b, lon, delta=0.0001)


            # Test with various rectangular resolutions
            resolution_pairs = [
                (0.1, 0.2), (0.125, 0.25), (0.2, 0.4), (0.5, 1.0), 
                (1.0, 0.5), (2.0, 1.0), (1.0, 2.0), (5.0, 2.5)
            ]
            
            for yres, xres in resolution_pairs:
                rounding = max(calc_min_rounding_log(yres), calc_min_rounding_log(xres))
                # Test corner coordinates (top-left)
                lat = 90 - yres/2
                lon = -180 + xres/2
                y, x = find_indices_xy(lat, lon, yres, xres, rounding)
                self.assertAlmostEqual(find_coordinates_xy(y, x, yres, xres, rounding), (lat, lon))
                
                # Test corner coordinates (bottom-right)
                lat = -90 + yres/2  
                lon = 180 - xres/2
                y, x = find_indices_xy(lat, lon, yres, xres, rounding)
                self.assertAlmostEqual(find_coordinates_xy(y, x, yres, xres, rounding), (lat, lon))

            # Test with very fine resolutions
            for res in [0.01, 0.025, 0.05]:
                rounding = calc_min_rounding_log(res)
                # Test center coordinates
                lat = 0.0
                lon = 0.0
                y, x = find_indices(lat, lon, res, rounding)
                recovered_lat, recovered_lon = find_coordinates(y, x, res, rounding)
                self.assertAlmostEqual(recovered_lat, lat, delta=res/2)
                self.assertAlmostEqual(recovered_lon, lon, delta=res/2)

            # Test with coarse resolutions
            for res in [10.0, 15.0, 30.0]:
                rounding = calc_min_rounding_log(res)
                # Test coordinates that align with grid
                lat = 90 - res/2
                lon = -180 + res/2
                y, x = find_indices(lat, lon, res, rounding)
                self.assertAlmostEqual(find_coordinates(y, x, res, rounding), (lat, lon))

    unittest.main()
