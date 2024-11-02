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

# Build cython extensions. Not used yet. Need to find processings that can be improved by cython.
# For now, no improvement gains in speed.

# from setuptools import setup
# from Cython.Build import cythonize
# import numpy

# # No improvement gains is speed for now.
# # Need to check if its worth to convert the bottleneck code to C or not.

# setup(
#     ext_modules=cythonize(["metacommunity.py",
#                            "community.py",
#                            "hydro_caete.py",
#                            "output.py",],
#         compiler_directives={'language_level': "3"}),
#     include_dirs=[numpy.get_include()]
# )