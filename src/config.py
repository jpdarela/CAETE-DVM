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
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
from typing import Union, Literal, List
import tomllib

import os
import sys
import tomllib #TODO: this lib only available in the standard library in python 3.11 and later.


"""This file contains some parameters that are used in the code.
   There is a class that read parameters stored in a toml file.
   The configurations can be accessed using the fetch_config function."""

# TODO: add handling for gfortran/gcc runtime assistance in windows systems.
# To do that I installed gfortran using the mingw64 distribution from WinLibs.
# Add the bin folder (where gcc/gfortran are) to the PATH environment variable.
# In windows, the fortran-python extension module (.pyd) needs the gfortran
# runtime dlls to be in the same folder as the .pyd file.
# See the build_caete.bat file for more information.

# Path to the configuration file
# This is the default path to the caete.toml file.
config_file = Path(__file__).parent / "caete.toml"

# Path to the caete_module/.libs folder (when using gfortran in windows systems)
caete_libs_path = Path(__file__).parent / "caete_module" / ".libs"
if sys.platform != "win32":
    caete_libs_path = None

# Path to the fortran runtime
# This is used to import the caete_module in windows systems.
fortran_runtime: Path | None = None
# IN windows systems, the fortran runtime (OneAPI) is needed to import the caete_module.
# Find path to the fortran compiler root, used in windows systems.
# With gfortran/gcc in windows, you will need to transfer the libraries to the src folder.
if sys.platform == "win32":
    ifort_compilers_env = [ "CMPLR_ROOT",
                            "ONEAPI_ROOT",
                            "IFORT_COMPILER25",
                            "IFORT_COMPILER24",
                            "IFORT_COMPILER23",
                            "FORTRAN_COMPILER",
                            "IFORT_COMPILER",
                            'IFORT_COMPILER_DIR',
                            'IFORT_COMPILER_ROOT']
    #
    # Check if any of the environment variables are set
    # This env var should point to the root directory of the fortran compiler
    # e.g. C:\Program Files (x86)\Intel\oneAPI\compiler\latest
    for env_var in ifort_compilers_env:
        if env_var in os.environ:
            if env_var == "ONEAPI_ROOT":
                # If ONEAPI_ROOT is set, use it as the fortran runtime path
                # This should point to the root directory of the oneAPI installation
                # e.g. C:\Program Files (x86)\Intel\oneAPI\compiler\latest
                fortran_runtime = Path(os.environ[env_var]) / "compiler" / "latest" / "bin"
            else:
                fortran_runtime = Path(os.environ[env_var]) / "bin"

            fortran_runtime = fortran_runtime.resolve()
            # print(f"Using Fortran runtime from environment variable {env_var}: {fortran_runtime}")

            if not fortran_runtime.exists():
                # fallback to the next environment variable
                continue
            break
    else:
        if "FC_RUNTIME" in os.environ:
            # If FC_RUNTIME is set, use it as the fortran runtime path
            # THis should point to the directory containing the fortran compiler dlls
            # e.g. C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin
            fortran_runtime = Path(os.environ["FC_RUNTIME"]).resolve()
        else:
            # If none of the environment variables are set, use a default path
            # This is the default path for Intel oneAPI Fortran compiler runtime
            # Adjust this path according to your installation
            fortran_runtime = Path(r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin").resolve()
            # print(f"Using default Fortran runtime path (last): {fortran_runtime}")


if sys.platform == "win32":
    # Check if the fortran runtime path exists
    if not fortran_runtime.exists():
        print(f"Path to OneAPI fortran runtime dll does not exist. Please set the environment variable FC_RUNTIME to the path of the fortran compiler dlls.")
        print(f"\n HINT: the fortran runtime path is set to a path like:\n {fortran_runtime}.\n Please check if this is correct.")
        sys.exit(1)


def get_fortran_runtime() -> Path | None:
    """Get the path to the fortran compiler dlls."""
    if sys.platform == "win32":
        if fortran_runtime is not None:
            return Path(fortran_runtime).resolve()
        else:
            raise ValueError("Fortran runtime path is not set.")
    else:
        return None

# Keep track of added directories
_added_dll_directories = set()

def update_sys_pathlib(lib) -> None:
    global _added_dll_directories
    if sys.platform != "win32":
        # On non-Windows systems, we don't need to add DLL directories
        return
    if isinstance(lib, str):
        lib_path = lib
    elif isinstance(lib, Path):
        lib_path = str(lib)
    else:
        # Handle other types recursively...
        if isinstance(lib, (list, tuple, set)):
            for l in lib:
                update_sys_pathlib(l)
            return
        elif isinstance(lib, dict):
            for k, v in lib.items():
                update_sys_pathlib(v)
            return
    if lib_path in _added_dll_directories:
        # print(f"Directory {lib_path} already added to the PATH.")
        return  # Already added, skip
    if not Path(lib_path).exists():
        return  # Path does not exist, skip
    try:
        os.add_dll_directory(lib_path)
        _added_dll_directories.add(lib_path)
        # print(f"Added DLL directory: {lib_path}")
    except OSError as e:
        if "directory has already been added" in str(e).lower():
            _added_dll_directories.add(lib_path)  # Track it anyway
            return  # Already added by another process/module
        else:
            raise ImportError(f"Could not add the DLL directory to the PATH: {e}")




if sys.platform == "win32":
    def update_runtime_gcc_gfortran(dll_path="caete_module/.libs"):
        """Update the system path to include the gfortran/gcc runtime libraries."""
        
        # Example path where gfortran/gcc dlls are located
        # You need to adjust this path according to your f2py wrapper build setup
        # We assume that the dlls are in the .libs folder inside the caete_module folder
        if caete_libs_path is None:
            return
        dll_src = Path(dll_path).resolve()
        dll_exists = False
        if dll_src.exists():
            dll_exists = True
        if dll_exists and caete_libs_path.exists():
            update_sys_pathlib(caete_libs_path.resolve())
    
    def update_runtime_oneapi():
        """Update the system path to include the OneAPI fortran runtime libraries."""
        update_sys_pathlib(get_fortran_runtime())

    
    update_runtime_oneapi()
    update_runtime_gcc_gfortran()



# CONFIG class using Pydantic ----


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    output_dir: Path = Field(
        "../outputs",
        description="Directory where the output files will be saved."
    )

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure the output directory exists."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v.resolve()


class CompressionConfig(BaseModel):
    """Configuration for compression settings."""
    compressor: Literal["lzma", "bz2", "lz4", "gzip", "xz"] = Field(
        "lzma",
        description="Compressor to use for joblib.dump"
    )
    rate: int = Field(
        9,
        ge=0,
        le=9,
        description="Compression rate for joblib.dump. 0-9, where 0 is no compression and 9 is maximum compression."
    )

    @field_validator('compressor')
    @classmethod
    def validate_compressor(cls, v: str) -> str:
        """Validate the compressor type."""
        valid_compressors = ["lzma", "bz2", "lz4", "gzip", "xz"]
        if v not in valid_compressors:
            raise ValueError(f"Invalid compressor: {v}. Must be one of {valid_compressors}.")
        if v == "lz4":
            try:
                import lz4
            except ImportError:
                raise ImportError("lz4 compression is not available. Install lz4 package to use it with joblib.")
        return v

class InputHandlerConfig(BaseModel):
    """Configuration for input data handling."""

    input_method: Literal["legacy", "ih"] = Field(
        "legacy",
        description="Input method to use. Options: 'legacy' or 'ih'"
    )

    input_type: Literal["netcdf", "bz2"] = Field(
        "netcdf",
        description="Type of input data. Options: 'netcdf' or 'bz2'"
    )
    mp: bool = Field(
        False,
        description="Use MPI for reading netcdf files"
    )

class MultiprocessingConfig(BaseModel):
    """Configuration for multiprocessing and parallelization."""
    nprocs: int = Field(
        16,
        gt=0,
        description="Threads used to post process the output. Not used by the model itself"
    )
    max_processes: int = Field(
        128,
        gt=0,
        description="Number of python processes used to run the model"
    )
    omp_num_threads: int = Field(
        1,
        ge=0,
        description="Number of threads used by OpenMP"
    )

class ConversionFactorsIsimipConfig(BaseModel):
    """Unit conversion factors for ISIMIP input data."""
    tas: float = Field(
        273.15,
        description="K to degC (sub) [input(K) to model(°C)]"
    )
    pr: float = Field(
        86400.0,
        description="kg m-2 s-1 to mm day-1 (mult) [input(kg m-2 s-1) to model(mm day-1)]"
    )
    ps: float = Field(
        0.01,
        description="Pa to hPa (mult) [input(Pa) to model(hPa)]"
    )
    rhs: float = Field(
        0.01,
        description="% to kg kg-1 (mult) [input(%) to model(kg kg-1)]"
    )
    rsds: float = Field(
        0.198,
        description="W m-2 to mol(photons) m-2 day-1 conversion factor"
    )

class MetacommConfig(BaseModel):
    """Metacommunity configuration parameters."""
    n: int = Field(
        1,
        ge=1,
        le=30,
        description="Number of communities in the metacommunity"
    )
    npls_max: int = Field(
        50,
        gt=0,
        description="Maximum number of PLS per community"
    )
    ntraits: int = Field(
        17,
        gt=0,
        description="Number of traits in a PLS"
    )



class CrsConfig(BaseModel):
    """Coordinate Reference System configuration."""
    res: float = Field(0.5, gt=0, description="Grid resolution")
    xres: float = Field(0.5, gt=0, description="X resolution")
    yres: float = Field(0.5, gt=0, description="Y resolution")
    epsg_id: int = Field(4326, description="EPSG code")
    datum: str = Field("WGS84", description="Geodetic datum")
    proj4: str = Field(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        description="PROJ4 string"
    )
    epsg_long_name: str = Field(
        "World Geodetic System 1984",
        description="Full EPSG name"
    )
    lat_units: str = Field("degrees_north", description="Latitude units")
    lon_units: str = Field("degrees_east", description="Longitude units")
    lat_zero: float = Field(90, description="Latitude zero point")
    lon_zero: float = Field(-180, description="Longitude zero point")

class FertilizationConfig(BaseModel):
    """Fertilization experiment configuration."""
    afex_mode: Literal["N", "P", "NP"] = Field(
        "N",
        description="Fertilization mode: Nitrogen, Phosphorus, or both"
    )
    n: float = Field(
        12.5,
        ge=0,
        description="Nitrogen fertilization rate (g m-2 y-1)"
    )
    p: float = Field(
        5.0,
        ge=0,
        description="Phosphorus fertilization rate (g m-2 y-1)"
    )

class Config(BaseModel):
    """
    Main CAETE model configuration.

    This class replaces the original dynamic Config class with a typed Pydantic model
    that provides compile-time type checking and runtime validation.
    """

    model_config = ConfigDict(
        extra='forbid',  # Prevent unknown fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True  # Use enum values in serialization
    )

    # Day of year for pls sampling
    doy_months: List[int] = Field(
        [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        min_length=0,
        max_length=36,
        description="Day of year for pls sampling"
    )

    output: OutputConfig = Field(
        default_factory=OutputConfig,  # type: ignore
        description="Output configuration for the model"
    )

    compression: CompressionConfig = Field(
        default_factory=CompressionConfig,  # type: ignore
        description="Compression settings for joblib.dump"
    )
    input_handler: InputHandlerConfig = Field(
        default_factory=InputHandlerConfig, # type: ignore
        description="Input data handling configuration"
    )
    multiprocessing: MultiprocessingConfig = Field(
        default_factory=MultiprocessingConfig, # type: ignore
        description="Multiprocessing configuration"
    )
    conversion_factors_isimip: ConversionFactorsIsimipConfig = Field(
        default_factory=ConversionFactorsIsimipConfig, # type: ignore
        description="Unit conversion factors for ISIMIP data"
    )
    metacomm: MetacommConfig = Field(
        default_factory=MetacommConfig, # type: ignore
        description="Metacommunity configuration"
    )
    crs: CrsConfig = Field(
        default_factory=CrsConfig, # type: ignore
        description="Coordinate Reference System configuration"
    )
    fertilization: FertilizationConfig = Field(
        default_factory=FertilizationConfig, # type: ignore
        description="Fertilization experiment configuration"
    )

    @field_validator('doy_months')
    @classmethod
    def validate_doy_months(cls, v):
        """Validate day of year values are reasonable."""
        if not v:
            return v  # Allow empty list
        if any(day < 1 or day > 365 for day in v):
            raise ValueError("Day of year must be between 1 and 365")
        if v != sorted(v):
            raise ValueError("Day of year values must be in ascending order")
        return v

    @field_validator('multiprocessing')
    @classmethod
    def validate_multiprocessing(cls, v):
        """Validate multiprocessing configuration."""
        if v.omp_num_threads > 3:
            import warnings
            warnings.warn(
                "Setting omp_num_threads > 2 may degrade performance due to thread creation overhead",
                UserWarning
            )
        return v

    # def __repr__(self) -> str:
    #     """Maintain compatibility with original Config.__repr__."""
    #     return f"Config(\n{pformat(self.model_dump())})\n"


def _fetch_config_parameters(config: Union[str, Path]) -> dict:
    """Get parameters from a TOML file returning a dictionary."""
    cfg = Path(config)
    assert cfg.exists(), f"File {cfg} does not exist."
    assert cfg.suffix == '.toml', f"File {cfg} is not a TOML file."
    assert cfg.is_file(), f"{cfg} is not a file."

    with open(config, 'rb') as f:
        data = tomllib.load(f)
    return data


def fetch_config(config: Union[str, Path] = config_file) -> Config:
    """
    Get parameters from the caete.toml file.

    Returns a validated Config object with full type support.

    Args:
        config: Path to the TOML configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    data = _fetch_config_parameters(config)
    return Config(**data)


# Convenience functions for accessing nested config values
# NOT used in the code yet. Just examples for now.
def get_input_type(cfg: Config) -> str:
    """Get the input data type from configuration."""
    return cfg.input_handler.input_type

def get_batch_size(cfg: Config) -> int:
    """Get batch size from configuration or return default. Batch size must match max processes."""
    return get_max_processes(cfg)

def get_max_processes(cfg: Config) -> int:
    """Get maximum number of processes from configuration."""
    return cfg.multiprocessing.max_processes

def get_conversion_factor(cfg: Config, variable: str) -> float:
    """Get unit conversion factor for a specific variable."""
    return getattr(cfg.conversion_factors_isimip, variable)



## OLD config class and functions
# class Config:
#     """
#     Class to store the parameters from a toml file.
#     Reads nested dictionaries as Config objects
#     All the parameters are stored as attributes of the object
#     Types are stored in the __annotations__ attribute
#     """
#     def __init__(self, d: Optional[Dict[str, Any]] = None) -> None:
#         self.__annotations__ = {}
#         if d is not None:
#             for k, v in d.items():
#                 if isinstance(v, dict):
#                     setattr(self, k, Config(v))
#                     self.__annotations__[k] = Config
#                 else:
#                     setattr(self, k, v)
#                     self.__annotations__[k] = type(v)


#     def __repr__(self) -> str:
#         return f"Config(\n{pformat(self.__dict__)})\n"


# def _fetch_config_parameters(config: Union[str, Path]) -> Dict[str, Any]:
#     """ Get parameters from the a toml file rerturning a dictionary."""

#     cfg = Path(config)
#     assert cfg.exists(), f"File {cfg} does not exist."
#     assert cfg.suffix == '.toml', f"File {cfg} is not a toml file."
#     assert cfg.is_file(), f"{cfg} is not a file."

#     with open(config, 'rb') as f:
#         # Works only with python 3.11 and above
#         data = tomllib.load(f)
#     return data

# # Can be used in the code to get the parameters any the toml file
# def fetch_config(config: Union[str, Path] = config_file) -> Config:
#     """ Get parameters from the  caete.toml file.
#     Returns a Config object"""
#     return Config(_fetch_config_parameters(config))
