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


# create_plsbin.py
import os
import sys

import plsgen as pls
import numpy as np



if sys.platform == "win32":
    from config import fortran_runtime
    try:
        os.add_dll_directory(fortran_runtime)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

import caete_module
a = pls.table_gen(caete_module.global_par.npls)
np.savetxt("pls_ex.txt", a.T) # type: ignore
