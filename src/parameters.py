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

"""DEFINE SOME MODEL PARAMETERS FOR CAETÊ EXPERIMENTS"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


# Soil Parameters
# Water saturation, field capacity & wilting point
# Topsoil
map_ws: NDArray[np.float64] = np.load("../input/soil/ws.npy")
map_fc: NDArray[np.float64] = np.load('../input/soil/fc.npy')
map_wp: NDArray[np.float64] = np.load('../input/soil/wp.npy')

# Subsoil
map_subws: NDArray[np.float64] = np.load("../input/soil/sws.npy")
map_subfc: NDArray[np.float64] = np.load("../input/soil/sfc.npy")
map_subwp: NDArray[np.float64] = np.load("../input/soil/swp.npy")

tsoil:Tuple[NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]] = (map_ws, map_fc, map_wp)

ssoil:Tuple[NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]] = (map_subws, map_subfc, map_subwp)

# Soil Hydraulics // Not used in the model. Needs to pull Gabriela's modifications.
theta_sat: NDArray[np.float64] = np.load("../input/hydra/theta_sat.npy")
psi_sat: NDArray[np.float64]  = np.load("../input/hydra/psi_sat.npy")
soil_texture: NDArray[np.float64] = np.load("../input/hydra/soil_text.npy")

hsoil:Tuple[NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]] = (theta_sat, psi_sat, soil_texture)