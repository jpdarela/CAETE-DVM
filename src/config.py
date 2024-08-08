from pathlib import Path
from typing import Union
import tomllib as tl

# path to the fortran compiler dlls, used in windows systems.
fortran_compiler_dlls = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin"


class Config:
    """ Class to store the parameters from a toml file.
    Reads nested dictionaries as Config objects
    All the parameters are stored as attributes of the object
    """

    def __init__(self, d=None) -> None:
        if d is not None:
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)


    def __repr__(self) -> str:
        return str(self.__dict__)


    def __str__(self) -> str:
        return str(self.__dict__)


    def __getitem__(self, key):
        return getattr(self, key)


    def __setitem__(self, key, value):
        setattr(self, key, value)


    def __iter__(self):
        return iter(self.__dict__)


    def __len__(self):
        return len(self.__dict__)


    def __contains__(self, key):
        return key in self.__dict__


    def items(self):
        return self.__dict__.items()


    def keys(self):
        return self.__dict__.keys()


    def values(self):
        return self.__dict__.values()


    def get(self, key, default=None):
        return getattr(self, key, default)


    def update(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)


def fetch_config_parameters(config: Union[str, Path]) -> dict:
    """ Get parameters from the a toml file rerturning a dictionary."""

    cfg = Path(config)
    assert cfg.exists(), f"File {cfg} does not exist."
    assert cfg.suffix == '.toml', f"File {cfg} is not a toml file."
    assert cfg.is_file(), f"{cfg} is not a file."

    with open(config, 'rb') as f:
        data = tl.load(f)
    return data

def fetch_config(config: Union[str, Path]) -> Config:
    """ Get parameters from the a toml file.
    Returns a Config object"""
    return Config(fetch_config_parameters(config))
