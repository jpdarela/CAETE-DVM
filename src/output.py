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

from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Union, List


class DailyBudget:
    """
    Helper class to store and update daily budget outputs.
    
    """
    
    __slots__ = [
        'evavg', 'epavg', 'phavg', 'aravg', 'nppavg', 'laiavg', 'rcavg', 
        'f5avg', 'rmavg', 'rgavg', 'cleafavg_pft', 'cawoodavg_pft', 
        'cfrootavg_pft', 'stodbg', 'ocpavg', 'wueavg', 'cueavg', 'c_defavg',
        'vcmax', 'specific_la', 'nupt', 'pupt', 'litter_l', 'cwd', 
        'litter_fr', 'npp2pay', 'lnc', 'limitation_status', 'uptk_strat', 
        'cp', 'c_cost_cwm', 'rnpp_out'
    ]
    
    def __init__(self):
        """Initialize DailyBudget with None values - will be set on first update."""
        for slot in self.__slots__:
            setattr(self, slot, None)
    
    
    def update(self, out: Tuple[Union[NDArray, str, List, int, float]]) -> None:
        """Update all slots with values from tuple.
        
        Args:
            out: Tuple containing output values in the same order as defined in daily_budget 
                 subroutine from the python extension caete_module.
                 Expected order: [evavg, epavg, phavg, aravg, nppavg, laiavg, rcavg, 
                 f5avg, rmavg, rgavg, cleafavg_pft, cawoodavg_pft, cfrootavg_pft,
                 stodbg, ocpavg, wueavg, cueavg, c_defavg, vcmax, specific_la, 
                 nupt, pupt, litter_l, cwd, litter_fr, npp2pay, lnc, 
                 limitation_status, uptk_strat, cp, c_cost_cwm, rnpp_out]
        """
        (self.evavg, self.epavg, self.phavg, self.aravg, self.nppavg,
         self.laiavg, self.rcavg, self.f5avg, self.rmavg, self.rgavg,
         self.cleafavg_pft, self.cawoodavg_pft, self.cfrootavg_pft,
         self.stodbg, self.ocpavg, self.wueavg, self.cueavg, self.c_defavg,
         self.vcmax, self.specific_la, self.nupt, self.pupt, self.litter_l,
         self.cwd, self.litter_fr, self.npp2pay, self.lnc, self.limitation_status,
         self.uptk_strat, self.cp, self.c_cost_cwm, self.rnpp_out) = out


# class budget_output:
#     """ Helper class to store the output of the daily_budget function.
#     """
#     evavg: NDArray[np.float64]
#     epavg: NDArray[np.float64]
#     phavg: NDArray[np.float64]
#     aravg: NDArray[np.float64]
#     nppavg: NDArray[np.float64]
#     laiavg: NDArray[np.float64]
#     rcavg: NDArray[np.float64]
#     f5avg: NDArray[np.float64]
#     rmavg: NDArray[np.float64]
#     rgavg: NDArray[np.float64]
#     cleafavg_pft: NDArray[np.float64]
#     cawoodavg_pft: NDArray[np.float64]
#     cfrootavg_pft: NDArray[np.float64]
#     stodbg: NDArray[np.float64]
#     ocpavg: NDArray[np.float64]
#     wueavg: NDArray[np.float64]
#     cueavg: NDArray[np.float64]
#     c_defavg: NDArray[np.float64]
#     vcmax: NDArray[np.float64]
#     specific_la: NDArray[np.float64]
#     nupt: NDArray[np.float64]
#     pupt: NDArray[np.float64]
#     litter_l: NDArray[np.float64]
#     cwd: NDArray[np.float64]
#     litter_fr: NDArray[np.float64]
#     npp2pay: NDArray[np.float64]
#     lnc: NDArray[np.float64]
#     limitation_status: NDArray [np.int16]
#     uptk_strat: NDArray[np.int32]
#     cp: NDArray[np.float64]
#     c_cost_cwm: NDArray[np.float64]
#     rnpp_out: NDArray[np.float64]


#     def __init__(self, *args):

#         fields = ["evavg", "epavg", "phavg", "aravg", "nppavg",
#                   "laiavg", "rcavg", "f5avg", "rmavg", "rgavg",
#                   "cleafavg_pft", "cawoodavg_pft", "cfrootavg_pft",
#                   "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg",
#                   "vcmax", "specific_la", "nupt", "pupt", "litter_l",
#                   "cwd", "litter_fr", "npp2pay", "lnc", "limitation_status",
#                   "uptk_strat", 'cp', 'c_cost_cwm', 'rnpp_out']

#         for field, value in zip(fields, args):
#             setattr(self, field, value)


# class budget_output2:
#     """ Helper class to store the output of the daily_budget function.
#     """
#     evavg: NDArray[np.float64]
#     epavg: NDArray[np.float64]
#     phavg: NDArray[np.float64]
#     aravg: NDArray[np.float64]
#     nppavg: NDArray[np.float64]
#     laiavg: NDArray[np.float64]
#     rcavg: NDArray[np.float64]
#     f5avg: NDArray[np.float64]
#     rmavg: NDArray[np.float64]
#     rgavg: NDArray[np.float64]
#     cleafavg_pft: NDArray[np.float64]
#     cawoodavg_pft: NDArray[np.float64]
#     cfrootavg_pft: NDArray[np.float64]
#     stodbg: NDArray[np.float64]
#     ocpavg: NDArray[np.float64]
#     wueavg: NDArray[np.float64]
#     cueavg: NDArray[np.float64]
#     c_defavg: NDArray[np.float64]
#     vcmax: NDArray[np.float64]
#     specific_la: NDArray[np.float64]
#     nupt: NDArray[np.float64]
#     pupt: NDArray[np.float64]
#     litter_l: NDArray[np.float64]
#     cwd: NDArray[np.float64]
#     litter_fr: NDArray[np.float64]
#     npp2pay: NDArray[np.float64]
#     lnc: NDArray[np.float64]
#     limitation_status: NDArray [np.int16]
#     uptk_strat: NDArray[np.int32]
#     cp: NDArray[np.float64]
#     c_cost_cwm: NDArray[np.float64]
#     rnpp_out: NDArray[np.float64]


#     def __init__(self):
#         """Init a budget_output object with empty arrays.
#         """
#         self.evavg = np.array([])
#         self.epavg = np.array([])
#         self.phavg = np.array([])
#         self.aravg = np.array([])
#         self.nppavg = np.array([])
#         self.laiavg = np.array([])
#         self.rcavg = np.array([])
#         self.f5avg = np.array([])
#         self.rmavg = np.array([])
#         self.rgavg = np.array([])
#         self.cleafavg_pft = np.array([])
#         self.cawoodavg_pft = np.array([])
#         self.cfrootavg_pft = np.array([])
#         self.stodbg = np.array([])
#         self.ocpavg = np.array([])
#         self.wueavg = np.array([])
#         self.cueavg = np.array([])
#         self.c_defavg = np.array([])
#         self.vcmax = np.array([])
#         self.specific_la = np.array([])
#         self.nupt = np.array([])
#         self.pupt = np.array([])
#         self.litter_l = np.array([])
#         self.cwd = np.array([])
#         self.litter_fr = np.array([])
#         self.npp2pay = np.array([])
#         self.lnc = np.array([])
#         self.limitation_status = np.array([])
#         self.uptk_strat = np.array([])
#         self.cp = np.array([])
#         self.c_cost_cwm = np.array([])
#         self.rnpp_out = np.array([])


# def update(self, *args):
#     """Update the budget_output object with new values."""
#     for field, value in zip(self.__annotations__.keys(), args):
#         setattr(self, field, value)
