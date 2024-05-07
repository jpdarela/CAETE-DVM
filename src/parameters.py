"""DEFINE SOME PARAMETERS FOR CAETÊ EXPERIMENTS"""
from pathlib import Path
import numpy as np

# Name of the base historical observed run.
BASE_RUN = 'HISTORICAL-RUN' #"HISTORICAL-RUN" <- in sombrero this is the
                  # STANDARD name for the historical observed run

ATTR_FILENAME = "pls_attrs-2000.csv"
START_COND_FILENAME = f"CAETE_STATE_START_{BASE_RUN}_.pkz"

run_path = Path(f"../outputs/{BASE_RUN}/{START_COND_FILENAME}")
pls_path = Path(f"./{ATTR_FILENAME}")


# Soil Parameters
# Water saturation, field capacity & wilting point
# Topsoil
map_ws = np.load("../input/soil/ws.npy")
map_fc = np.load('../input/soil/fc.npy')
map_wp = np.load('../input/soil/wp.npy')

# Subsoil
map_subws = np.load("../input/soil/sws.npy")
map_subfc = np.load("../input/soil/sfc.npy")
map_subwp = np.load("../input/soil/swp.npy")

tsoil = (map_ws, map_fc, map_wp)
ssoil = (map_subws, map_subfc, map_subwp)

# Hydraulics
theta_sat = np.load("../input/hydra/theta_sat.npy")
psi_sat = np.load("../input/hydra/psi_sat.npy")
soil_texture = np.load("../input/hydra/soil_text.npy")

hsoil = (theta_sat, psi_sat, soil_texture)