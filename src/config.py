# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho
"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from pathlib import Path
from pprint import pformat
from typing import Union, Dict , Any, Optional

import os
import sys
import tomllib


"""This file contains some parameters that are used in the code.
   Thehe is a class that read parameters stored in a toml file.
   The configurations can be accessed using the fetch_config function."""

# path to the fortran compiler dlls, used in windows systems.
fortran_runtime_fallback = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin"
fortran_runtime: Path = Path(os.environ.get("FC_RUNTIME", fortran_runtime_fallback)).resolve()

if sys.platform == "win32":
    # Check if the fortran runtime path exists
    if not fortran_runtime.exists():
        print(f"Path to fortran runtime dll does not exist. Please set the environment variable FC_RUNTIME to the path of the fortran compiler dlls.")
        print(f"\n HINT: the fortran runtime path is set to a path like:\n {fortran_runtime}.\n Please check if this is correct.")
        sys.exit(1)

config_file = Path(__file__).parent / "caete.toml"

def get_fortran_runtime() -> Path:
    """Get the path to the fortran compiler dlls."""
    return Path(fortran_runtime).resolve()

def update_sys_pathlib(lib) -> None:
    if sys.platform == "win32":
        try:
            os.add_dll_directory(lib)
        except:
            raise ImportError("Could not add the DLL directory to the PATH")


class Config:
    """
    Class to store the parameters from a toml file.
    Reads nested dictionaries as Config objects
    All the parameters are stored as attributes of the object
    Types are stored in the __annotations__ attribute
    """
    def __init__(self, d: Optional[Dict[str, Any]] = None) -> None:
        self.__annotations__ = {}
        if d is not None:
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                    self.__annotations__[k] = Config
                else:
                    setattr(self, k, v)
                    self.__annotations__[k] = type(v)


    def __repr__(self) -> str:
        return f"Config(\n{pformat(self.__dict__)})\n"


def _fetch_config_parameters(config: Union[str, Path]) -> Dict[str, Any]:
    """ Get parameters from the a toml file rerturning a dictionary."""

    cfg = Path(config)
    assert cfg.exists(), f"File {cfg} does not exist."
    assert cfg.suffix == '.toml', f"File {cfg} is not a toml file."
    assert cfg.is_file(), f"{cfg} is not a file."

    with open(config, 'rb') as f:
        # Works only with python 3.11 and above
        data = tomllib.load(f)
    return data

# Can be used in the code to get the parameters any the toml file
def fetch_config(config: Union[str, Path] = config_file) -> Config:
    """ Get parameters from the a toml file.
    Returns a Config object"""
    return Config(_fetch_config_parameters(config))
