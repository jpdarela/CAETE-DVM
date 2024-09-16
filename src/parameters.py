# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

_ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

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

# Running in sombrero:
# These variables point to the directory where the input data is stored.
input_in_sombrero: Path = Path("/home/amazonfaceme/shared_data/").resolve()
local_input: Path = Path("../input").resolve()

#masks
# Pan-Amazon mask. Only forest dominated gridcells are considered. MAPBIOMAS 2000
mask_am: NDArray = np.load("../input/mask/pan_amazon_05d_FORESTS_MAPBIOMASS_2000.npy")

# Pan-Amazon mask.
mask_pan_am: NDArray = np.load("../input/mask/mask_raisg-360-720.npy")


# Name of the base historical observed run.
# Used in the old version scripts
BASE_RUN: str = 'HISTORICAL-RUN' #"HISTORICAL-RUN" <- in sombrero this is the
                  # STANDARD name for the historical observed run.

ATTR_FILENAME: str = "pls_attrs-2000.csv"
START_COND_FILENAME: str = f"CAETE_STATE_START_{BASE_RUN}_.pkz"

output_path: Path = Path(f"../outputs").resolve()
run_path: Path = Path(f"../outputs/{BASE_RUN}/{START_COND_FILENAME}")
pls_path: Path = Path(f"./{ATTR_FILENAME}")

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

# Hydraulics
theta_sat: NDArray[np.float64] = np.load("../input/hydra/theta_sat.npy")
psi_sat: NDArray[np.float64]  = np.load("../input/hydra/psi_sat.npy")
soil_texture: NDArray[np.float64] = np.load("../input/hydra/soil_text.npy")

hsoil:Tuple[NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]] = (theta_sat, psi_sat, soil_texture)