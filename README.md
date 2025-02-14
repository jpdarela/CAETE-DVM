
# CAETÊ

The prototype version of the model is stored in the branch CAETE-DVM-v0.1

This is the implementation of the Dynamic Vegetation Model CAETÊ (CArbon and Ecosystem Trait-based Evaluation model) - including Nitrogen and Phosphorus cycling.

## Development Dependencies

### Windows (>7) setup

cl.exe, link.exe, nmake.exe -
Microsoft visual studio build tools https://visualstudio.microsoft.com/downloads/

ifx.exe -
Intel fortran compiler https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html#gs.ihbm92

[Python 3.11.6](https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe) and [windows requirements](./src/requirements_311.txt) (only tested this way).

See the [windows Makefile](./src/Makefile_win) for instructions on how to build the python extension module.

In the windows Makefile the ifx.exe compiler is called. The Makefile must be executed in a shell with the proper environment:
https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/oneapi-development-environment-setup.html

To build in windows:

Assuming that the Intel OneAPI setvars.bat script is in C:\Program Files (x86)\Intel\oneAPI\setvars.bat.

```powershell
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'

PS> nmake -f Makefile_win so

```

### Linux setup

- gnu-make

- gcc

- gfortran

- python 3.11 and [legacy requirements](./src/requirements_311.txt)
or
- python 3.12 and [linux requirements](./src/requirements_312.txt)

### General Dependencies for linux

In linux it is possible to run the model in python 3.11 and 3.12. However, the building system used to compile the fortran/python extension module differs between python versions. In python 3.12 the numpy distutils was removed from numpy. See: [Numpy distutils status](https://numpy.org/doc/stable/reference/distutils_status_migration.html#distutils-status-migration). The new (candidate) build method is based on meson. So, if you are using python 3.12 you will need also:

- meson
- ninja

Note: The new building (meson) system was only tested in linux

Python Dependencies (see the [requirements for python 3.11](./src/requirements_311.txt) and [requirements for python 3.12](./src/requirements_312.txt))

- numpy
- joblib
- netCDF4
- charset-normalizer
- lz4
- numba
- zstandard
- pyproj
- pandas

Edit the PYEXEC variable in the [linux Makefile](./src/Makefile) to match the python that you will use to run the model.

Use the make target setup_311 or setup_312 to install python libraries.

To build the extension module with python 3.12 and meson in linux:

```bash
$ make setup_312

$ make ext_mod_meson

```

To build the extension module with python 3.11 in linux:

```bash
$ make setup_311

$ make ext_mod_legacy

```

## Outdated content ->

## Running and Developing CAETÊ

This section suposses you have a working python environment and an installed fortran compiler. I strongly recommend you to do it in a LINUX/GNU operating system. Most of the fuctionality will not work on MS-windows systems because of the parallel computations realized in python.
If you are a MAC-OS user and need help to set up your environment, check the [development environment section](#development-Environment)

CAETÊ uses both Python and Fortran and uses `f2py` module to create an library of code (functions and subroutines) that can be imported and used in python scripts. This means that the Fortran code must be compiled before you can run the code.

The Makefile inside `/src` folder have useful automation to make it easier.

`clean_plsgen` - remove the cache files containing allocation combinations for PLS creation in plsgen.py. Run this will make the first run of CAETÊ in your computer slower but will make sure that your input data is ok!

`make clean` - it clear your python cache and deletes some compiled files.

`make so` - compile fortran source and creates a file named with something like `caete_module.so`, the wrapped Fortran functions and subroutines for python.

For your use we prepared some sampled input data. You probably do not want to run the model for the entire Amazon outside an HPC environment and we are not able to publicize the input dataset.

To build and run CAETÊ in you PC you can do the following:

```bash
# Use pyenv to select your python executable
CAETE-DVM$ pyenv local <define your python version (>= 3.5)

# Goto the source file
CAETE-DVM$ cd src

# Clean you cache
CAETE-DVM/src$ make clean_plsgen
# After the first clean_plsgen there is almost no need to run in anymore.
# If you change the plsgen.py file, particularly the creation
# of allocation combinations you must to re-run the above command

CAETE-DVM/src$ make clean

# Build caete_module.so
CAETE-DVM/src$ make so

# Run the model, -i will leave you in the python shell that executed model_driver.py
CAETE-DVM/src$ python -i model_driver.py

# At this point the model will ask you if it will run in sombrero
# Just say: n<enter>
# Choose the region you want to execute and wait a time.
```

We have a module that stores the data of the compressed binary model outputs in netCDF (CF compliant) files. It runs right after the model calculations are ended.
So you have two types of output data:

1 - The results are saved as compressed pickled python dictionaries for each gridcell for each each spinup realized by the `grd.run_caete` method in a numerated file with the extension `.pkz` in the folder `./ouputs/run_name/gridcellYX/spinZ.pkz` where Y and X are the geographic coordinates of the grid cell object, Z is the number of consecutive spinups made by the `grd.run_caete` method and `run_name` is a string asked to be inputed by the user before everthing starts. This is the standard output data when running the model with plot originated data. Time related metadata is saved within each .pkz file.

If you want to check your raw outputs:

```bash

# go to the ouputs folder for a specific grid cell
CAETE-DVM$ cd ouputs/run_name/gridcellYX # where Y & X are the indexes

# open the python interpreter
CAETE-DVM/outputs/run_name/gridcellYX$ python
```

```python

# To open the ouputs and check the file contents
>>> import joblib
>>> # Open the file containing the 5th spinup realized
>>> # edit the functions apply_fun() in model_driver.py script
>>> with open("spin05.pkz", 'rb') as fh:
>>>    dt = joblib.load(fh)
>>> dt.keys() # list the available keys for the ouputs
>>> # PLot some variables
>>> import matplotlib.pyplot as plt
>>> plt.plot(dt['npp'])
>>> plt.show()
>>> plt.plot(dt['area'].T)
>>> plt.show()
>>> plt.plot(dt['u_strat'][0,:,:].T, 'x')
>>> plt.show()

```

2 - In the same folder were the raw outputs are saved (`./ouputs/run_name/`) you will find the nc_outputs folder,_i.e._, the
folder containing CF compliant netCDF files with daily values for the main output variables. Some high dimensional output data is simplified. So, some valuable information is absent in the netCDF files.

## Development Environment

If you need help configuring your development environment, installing python, installing CAETÊ dependencies or setting up a debug enviornment in vscode, check the [CAETÊ starting pack tutorial](https://github.com/fmammoli/CAETE-Tutorials)

## __CONTRIBUTORS__

- Anja Rammig
- Bárbara R. Cardeli
- Bianca Rius
- Caio Fascina
- Carlos A. Quesada
- David Lapola
- Felipe Mammoli
- Gabriela M. Sophia
- Gabriel Marandola
- Helena Alves
- Katrin Fleischer
- Phillip PAPAstefanou
- Tatiana Reichert
