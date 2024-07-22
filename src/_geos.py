from typing import Tuple
import numpy as np
from numba import jit
import pyproj

def calculate_area(center_lat:float, center_lon:float, dx:float=0.5, dy:float=0.5)->float:
    """Calculates the area of a cell on the Earth's surface given the center coordinates and the cell resolution
    using a geographic coordinate system with the WGS84 datum.

    Args:
        center_lat (float): Center latitude, degrees North
        center_lon (float): Center longitude, degrees East
        dx (float): Cell resolution in the x-direction, degrees
        dy (float): Cell resolution in the y-direction, degrees

    Returns:
        float: Area of the grid cell in square meters
    """

    # do not allow negative dx and dy and crappy center_lat and center_lon
    assert dx > 0 and dy > 0, "dx and dy must be positive"
    assert center_lat >= -90 + (dy/2) and center_lat <= 90 - (dy/2), f"center_lat must match the resolution. Expected range: {-90+(dy/2)},{90-(dy/2)}"
    assert center_lon >= -180 + (dx/2) and center_lon <= 180 - (dx/2), f"center_lon must match the resolution. Expected range:{-180+(dx/2)},{180-(dx/2)}"

    # Define a geographic coordinate system with WGS84 datum
    geod = pyproj.Geod(ellps='WGS84')

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


@jit(nopython=True)
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

    Yc:float = round(N, rounding)
    Xc:float = round(W, rounding)

    half_res:float = res / 2
    Ymin:float = -90 + half_res
    Xmin:float = -180 + half_res

    # Generate longitude and latitude arrays (cell center coordinates)
    lon = np.arange(Xmin, 180, res)
    lat = np.arange(Ymin, 90, res)


    # Find indices for Yc and Xc using searchsorted
    # Yc is negative because our origin (90 degrees north) is in the upper left corner of the matrix.
    Yind = np.searchsorted(lat, -Yc - half_res, side='left')
    Xind = np.searchsorted(lon, Xc - half_res, side='left')

    if Yc > 90:
        Yind = -1
    if Xc < -180:
        Xind = -1

    return Yind, Xind


@jit(nopython=True)
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
    lon = np.arange(Xmin, 180, res)
    # Latitude is reversed because our upper left latitude origin (90 degrees north) is in the upper left corner.
    lat = np.arange(Ymin, 90, res)[::-1]

    # Find latitude and longitude for Yind and Xind

    N = lat[Yind]
    W = lon[Xind]

    return N, W



if __name__ == "__main__":
    center_lat = -89.75  # latitude of the center grid point
    center_lon = 179.75  # longitude of the center grid point
    dx = 0.5  # resolution in degrees (longitude)
    dy = 0.5  # resolution in degrees (latitude)
    area = calculate_area(center_lat, center_lon, dx, dy)
    print(area)

    N = -89.75  # latitude in decimal degrees north
    W = -179.75  # longitude in decimal degrees west
    res = 0.5  # grid resolution in degrees
    rounding = 2  # decimal significant digits
    Yind, Xind = find_indices(N, W, res, rounding)
    print(Yind, Xind)
    print(find_indices(N, W, res, rounding))
    print(find_coordinates(Yind, Xind, res))
