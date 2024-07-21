from typing import Tuple

import numpy as np
from numba import jit


@jit(nopython=True)
def find_coord(N: float, W: float, res: float = 0.5, rounding: int = 2) -> Tuple[int, int]:
    """It finds the indices for a given latitude and longitude in a planar grid

    Args:
        N (float): latitude in decimal degrees north
        W (float): longitude in decimal degrees west
        res (float, optional): grid resolution. Defaults to 0.5 degrees.
        rounding (int, optional): decimal significant digits. Defaults to 2.

    Returns:
        tuple[int, int]: (y, x) indices for the given latitude and longitude
        in the grid (0,0) is the upper left corner. Feeding the function with
        lat/long outside the boundaries(-180 - 180; -90 - 90) of the geographic coordinates
        will cause the function to return invalid indices in the grid.
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
    # Yc is negative because our origin (-90 degrees north) is in the upper left corner.
    Yind = np.searchsorted(lat, -Yc - half_res, side='left')
    Xind = np.searchsorted(lon, Xc - half_res, side='left')

    if Yc > 90:
        Yind = -1
    if Xc < -180:
        Xind = -1

    return Yind, Xind
