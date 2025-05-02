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

import unittest
import numpy as np

from numba import jit # type: ignore
import pyproj
from config import fetch_config

# config_file = Path("../src/caete.toml").resolve()

config = fetch_config()
datum = config.crs.datum # type: ignore


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

@jit(nopython=True, cache=True) # type: ignore
def find_indices_xy(N: float, W: float, res_y: float = 0.5, res_x: float = 0.5, rounding: int = 2) -> Tuple[int, int]:
    """
    It finds the indices for a given latitude and longitude in a planar grid.
    WARNING Feeding the function with lat/long outside the boundaries
    (-180 - 180; -90 - 90) will cause the function to return invalid indices.

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

    Yc:float = round(N, rounding)
    Xc:float = round(W, rounding)

    half_res_y:float = res_y / 2
    half_res_x:float = res_x / 2
    Ymin:float = -90 + half_res_y
    Xmin:float = -180 + half_res_x

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res_x) # type: ignore
    lat = np.arange(Ymin, 90, res_y) # type: ignore

    # Find indices for Yc and Xc using bisection
    # Yc is negative because our origin (90 degrees north) is in the upper left corner of the matrix.
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
    Yc:float = round(N, rounding)
    Xc:float = round(W, rounding)

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
def find_coordinates_xy(Yind: int, Xind: int, res_y: float = 0.5, res_x: float = 0.5) -> Tuple[float, float]:
    """
    It finds the latitude and longitude (cell center) for given (Yind, Xind) indices in a planar grid.
    WARNING: This function do not check if the indices are valid.

    Args:
        Yind (int): y index in the grid
        Xind (int): x index in the grid
        res_y (float, optional): grid resolution for y-axis. Defaults to 0.5 degrees.
        res_x (float, optional): grid resolution for x-axis. Defaults to 0.5 degrees.

    Returns:
        tuple[float, float]: (N, W) latitude and longitude for the given indices.
        Feeding the function with indices outside the boundaries of the grid
        will cause the function to return invalid latitude and longitude.
    """

    half_res_y:float = res_y / 2
    half_res_x:float = res_x / 2
    Ymin:float = -90 + half_res_y
    Xmin:float = -180 + half_res_x

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res_x) # type: ignore
    lat = np.arange(Ymin, 90, res_y)[::-1] # type: ignore

    # Find latitude and longitude for Yind and Xind
    N = lat[Yind]
    W = lon[Xind]

    return N, W


@jit(nopython=True, cache=True) # type: ignore
def find_coordinates(Yind: int, Xind: int, res: float = 0.5) -> Tuple[float, float]:
    """It finds the latitude and longitude (cell center) for given (Yind, Xind) indices in a planar grid
    WARNING: This function do not check if the indices are valid.

    Args:
        Yind (int): y index in the grid
        Xind (int): x index in the grid
        res (float, optional): grid resolution. Defaults to 0.5 degrees.

    Returns:
        tuple[float, float]: (N, W) latitude and longitude for the given indices.
        Feeding the function with indices outside the boundaries of the grid
        will cause the function to return invalid latitude and longitude.
    """

    half_res:float = res / 2
    Ymin:float = -90 + half_res
    Xmin:float = -180 + half_res

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res) # type: ignore
    lat = np.arange(Ymin, 90, res)[::-1] # type: ignore

    # Find latitude and longitude for Yind and Xind

    N = lat[Yind]
    W = lon[Xind]

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


def latitude_axis(north: float | int, south: float | int, yres: float | int) -> np.ndarray: # type: ignore
    """
    Generate a latitude axis for a given region
    """
    return np.arange(north, south, -yres) + yres / 2 # type: ignore


def longitude_axis(west: float | int, east: float | int, xres: float | int) -> np.ndarray: # type: ignore
    """
    Generate a longitude axis for a given region
    """
    return np.arange(west, east, xres) + xres / 2 # type: ignore

def get_axis(bbox): # type: ignore
    """
    Get latitude and longitude axis for a region of interest
    """
    lat = latitude_axis(bbox["north"], bbox["south"], bbox["res_y"]) # type: ignore
    lon = longitude_axis(bbox["west"], bbox["east"], bbox["res_x"]) # type: ignore
    return lat, lon # type: ignore


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

            self.assertEqual(find_indices_xy(0, 0, 0.5, 0.5), find_indices(0, 0, 0.5))
            self.assertEqual(find_indices_xy(45, 45, 1, 1), find_indices(45, 45, 1))

            # Test with invalid inputs
            # Resolution must be positive and lat/lon must be within the boundaries
            self.assertEqual(find_indices_xy(91, 0, 1, 1), (-1, 179))
            self.assertEqual(find_indices_xy(-91, 0, 1, 1), (-1, 179))
            self.assertEqual(find_indices_xy(0, 181, 1, 1), (89, -1))
            self.assertEqual(find_indices_xy(0, -181, 1, 1), (89, -1))
            self.assertEqual(find_indices_xy(0, 0, -1, 1), (-1, -1))
            self.assertEqual(find_indices_xy(0, 0, 1, -1), (-1, -1))
            self.assertEqual(find_indices_xy(0, 0, 0, 0), (-1, -1))

            # find_indices and find_indices_xy should return the same values
            self.assertEqual(find_indices_xy(0, 0, 0.5, 0.5), find_indices(0, 0, 0.5))
            self.assertEqual(find_indices_xy(45, 45, 1, 1), find_indices(45, 45, 1))
            self.assertEqual(find_indices_xy(0, 0, 0.25, 0.25), find_indices(0, 0, 0.25))
            self.assertEqual(find_indices_xy(45, 45, 0.25, 0.25), find_indices(45, 45, 0.25))

            # outputs of find_coordinates feeded with the outputs of find_indices should be the same as the inputs of find_indices
            def check_crossed_results(lat=90, lon=180, res=0.5):  # type: ignore
                y, x = find_indices(lat, lon, res)
                lat1, lon1 = find_coordinates(y, x, res)
                self.assertAlmostEqual(lat1, lat, delta=res/2)
                self.assertAlmostEqual(lon1, lon, delta=res/2)

            check_crossed_results()
            check_crossed_results(0, 0, 0.5)
            check_crossed_results(45, 45, 1)
            check_crossed_results(0, 0, 0.25)
            check_crossed_results(45, 45, 0.25)
            check_crossed_results(90, 180, 0.5)
            check_crossed_results(-90, -180, 0.5)
            check_crossed_results(-90, 180, 0.5)
            check_crossed_results(90, -180, 0.01)


        def test_find_coordinates(self):
            # Test with known values
            res = 0.5
            lat = 90 - 0.5/2
            lon = -180 + 0.5/2
            y, x = find_indices_xy(lat, lon, res, res)
            self.assertEqual(find_coordinates_xy(y, x, res, res), (lat, lon))

            xres = 0.25
            yres = 0.5
            lat = 90 - yres/2
            lon = 180 - xres/2
            y, x = find_indices_xy(lat, lon, yres, xres)
            self.assertEqual(find_coordinates_xy(y, x, yres, xres), (lat, lon))

            res =  0.5
            lat = 90 - 0.5/2
            lon = 180 - 0.5/2
            y, x = find_indices(lat, lon, res)
            self.assertEqual(find_coordinates(y, x, res), (lat, lon))

            res = 0.25
            lat = 90 - 0.25/2
            lon = -180 + 0.25/2
            y, x = find_indices(lat, lon, res)
            self.assertEqual(find_coordinates(y, x, res), (lat, lon))

            res = 0.25
            lat = 90 - 0.25/2
            lon = 180 - 0.25/2
            y, x = find_indices(lat, lon, res)
            self.assertEqual(find_coordinates(y, x, res), (lat, lon))

    unittest.main()
