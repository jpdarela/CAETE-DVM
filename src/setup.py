from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["region.pyx",
                           "caete.pyx",
                           "worker.pyx",
                           "metacommunity.pyx",
                           "community.pyx",
                           "hydro_caete.pyx",
                           "output.pyx",],
        compiler_directives={'language_level': "3"}),
    include_dirs=[numpy.get_include()]
)