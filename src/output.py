# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from numpy.typing import NDArray
import numpy as np

class budget_output:
    """ Helper class to store the output of the daily_budget function.
    """
    evavg: NDArray[np.float64]
    epavg: NDArray[np.float64]
    phavg: NDArray[np.float64]
    aravg: NDArray[np.float64]
    nppavg: NDArray[np.float64]
    laiavg: NDArray[np.float64]
    rcavg: NDArray[np.float64]
    f5avg: NDArray[np.float64]
    rmavg: NDArray[np.float64]
    rgavg: NDArray[np.float64]
    cleafavg_pft: NDArray[np.float64]
    cawoodavg_pft: NDArray[np.float64]
    cfrootavg_pft: NDArray[np.float64]
    stodbg: NDArray[np.float64]
    ocpavg: NDArray[np.float64]
    wueavg: NDArray[np.float64]
    cueavg: NDArray[np.float64]
    c_defavg: NDArray[np.float64]
    vcmax: NDArray[np.float64]
    specific_la: NDArray[np.float64]
    nupt: NDArray[np.float64]
    pupt: NDArray[np.float64]
    litter_l: NDArray[np.float64]
    cwd: NDArray[np.float64]
    litter_fr: NDArray[np.float64]
    npp2pay: NDArray[np.float64]
    lnc: NDArray[np.float64]
    limitation_status: NDArray [np.int16]
    uptk_strat: NDArray[np.int32]
    cp: NDArray[np.float64]
    c_cost_cwm: NDArray[np.float64]
    rnpp_out: NDArray[np.float64]

    def __init__(self, *args):

        fields = ["evavg", "epavg", "phavg", "aravg", "nppavg",
                  "laiavg", "rcavg", "f5avg", "rmavg", "rgavg",
                  "cleafavg_pft", "cawoodavg_pft", "cfrootavg_pft",
                  "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg",
                  "vcmax", "specific_la", "nupt", "pupt", "litter_l",
                  "cwd", "litter_fr", "npp2pay", "lnc", "limitation_status",
                  "uptk_strat", 'cp', 'c_cost_cwm', 'rnpp_out']

        for field, value in zip(fields, args):
            setattr(self, field, value)
