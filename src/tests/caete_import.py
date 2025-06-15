import sys
import numpy as np
import os

sys.path.append("../")
current_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    try:
        os.add_dll_directory(os.environ["FC_RUNTIME"])
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

import caete_module as model
from caete_module import soil_dec
from plsgen import *
from metacommunity import pls_table