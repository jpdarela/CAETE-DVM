import os
import sys
import tomllib as tl


class Config:
    def __init__(self, d=None) -> None:
        if d is not None:
            self.__dict__ = d
            for k, v in d.items():
                setattr(self, k, v)

def get_parameters(config):
    """ Get parameters from the pls_gen.toml file """

    with open(config, 'rb') as f:
        data = tl.load(f)
    return data


# path to the fortran compiler dlls, used in windows systems.
fortran_compiler_dlls = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin"

if sys.platform == "win32":
      try:
         os.add_dll_directory(fortran_compiler_dlls)
      except:
         raise ImportError("Could not add the DLL directory to the PATH")



