from setuptools import setup
from Cython.Build import cythonize
import numpy


# No improvement gains is speed for now.
# Need to convert the bottleneck functions to C or not.

setup(
    ext_modules=cythonize(["metacommunity.py",
                           "community.py",
                           "hydro_caete.py",
                           "output.py",],
        compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)