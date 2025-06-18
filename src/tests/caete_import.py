import sys
import numpy as np
import os

sys.path.append("../")
# current_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    from config import update_sys_pathlib, fortran_runtime
    update_sys_pathlib(fortran_runtime)

import caete_module as model
from caete_module import soil_dec
from plsgen import *
from metacommunity import pls_table