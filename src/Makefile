# ! Copyright 2017- LabTerra

# !     This program is free software: you can redistribute it and/or modify
# !     it under the terms of the GNU General Public License as published by
# !     the Free Software Foundation, either version 3 of the License, or
# !     (at your option) any later version.)

# !     This program is distributed in the hope that it will be useful,
# !     but WITHOUT ANY WARRANTY; without even the implied warranty of
# !     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# !     GNU General Public License for more details.

# !     You should have received a copy of the GNU General Public License
# !     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ! AUTHOR: JP Darela

# Makefile for UNIX-like systems. To be used with gnu-make + gcc + gfortran + f2py
# This Makefile is used to build the caete_module.
# It also can build a fortran executable for debugging purposes.
# This is not implemented in the windows version.

# Requires python 3.13 or lower (because of numba)
# If using python 3.12, use the ext_mod_meson target
# If using python 3.11, use the ext_mod_legacy target

PYEXEC = python # Edit this line to point to the python executable
PIP = $(PYEXEC) -m pip
F2PY = $(PYEXEC) -m numpy.f2py
MODNAME = caete_module
INTERFACES = $(MODNAME).pyf
DEBUG_PROGRAM = run_debug
MOD_DIR = ./

# RUN MODE FLAGS
MFLAG = -m
HFLAG = -h
CFLAG = -c
OVRTFLAG = --overwrite-signature

# Compiler and flags. Choose one of the options below. 

# GNU Fortran compiler with libgomp. Set FC to gfortran and OMP_LIB to -lgomp (Default) 
# Ext flags for gfortran: EXT_FLAGS = -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans

# gfortran compiler - libgomp
FC = gfortran
OMP_LIB = -lgomp
EXT_FLAGS = -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans

# # AMD Optimizing C/C++ and Fortran Compilers. Not tested with other versions of flang.
## Faster in AMD hardware 
# # flang compiler uses libomp 
# FC = flang
# OMP_LIB = -lomp
# EXT_FLAGS = -fno-unsafe-math-optimizations -frounding-math -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN

# f2py debugging flag: --debug
# Add this flag to the f2py command to enable debugging information in the extension module.
DEBUG = 


# gfrotran FLAGS for debugging (not used in the extension module build)
FCFLAGS = -g -Wall -Wextra -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic
FLFLAGS = -c -g -Wall -fcheck=all -Wextra -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic

# Choose the main source file for the budget module
# There is a maintenance problem here, as there are two versions of the budget module:
# one that uses stack allocation (budget_fixed.F90) and another that uses heap allocation (budget.F90).
# With time these functions can diverge if not maintained properly. Put both in one module in the future. 

# Use budget_fixed.F90 for stack allocation (faster)
MAIN_SUB = budget_fixed.F90

# Use budget.F90 for heap allocation (slower, but allows larger number of PLS)
# MAIN_SUB = budget.F90


# Sources for extension module creation (F2PY)
src_lib = types.f90 global.f90 photo_par.f90 funcs.f90 evap.f90 soil_dec.f90 cc.f90 allocation.f90 productivity.f90
sources = $(src_lib) $(MAIN_SUB)


# Objects for compilation of the debug executable (LINUX)
src_obj = types.o photo_par.o global.o funcs.o evap.o soil_dec.o cc.o allocation.o productivity.o
objects = $(src_obj) $(basename $(MAIN_SUB)).o debug_caete.o


# Targets
.PHONY: setup interface so so_parallel clean clean_so modules

all: ext_mod_meson_parallel #ext_mod_legacy

# Bypass the makefile. To be used with gnu-make in windows
mingw64_build:
	cmd /c build_caete.bat
	

# Python setup
setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements_3xx_last.txt

modules: $(sources)
	$(FC) -c $(sources)

# Build the extension module (caete_module) from .F90 source
# To be used when python >= 3.12
ext_mod_meson: $(sources) modules
	@$(PYEXEC) precompilation_data.py
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) -m $(MODNAME) $(OVRTFLAG) --quiet
	@export FC=$(FC) && \
	$(F2PY) $(INTERFACES) $(CFLAG) $(sources) --build-dir build --backend meson --f90flags="-Wall $(EXT_FLAGS)" $(DEBUG)


ext_mod_meson_parallel: $(sources) modules
	@$(PYEXEC) precompilation_data.py
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) -m $(MODNAME) $(OVRTFLAG) --quiet
	@export FC=$(FC) && \
	$(F2PY) $(INTERFACES) $(CFLAG) $(sources) --build-dir build --backend meson --f90flags="-Wall $(EXT_FLAGS) -fopenmp" $(OMP_LIB) $(DEBUG) --no-freethreading-compatible


# Build the extension module (caete_module) from .F90 source
# To be used when python == 3.11
ext_mod_legacy: $(sources) modules
	@$(PYEXEC) precompilation_data.py
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) -m $(MODNAME) $(OVRTFLAG) --quiet
	$(F2PY) $(INTERFACES) $(CFLAG) $(sources) --f90exec="$(FC)" --f77exec="$(FC)" --f90flags="-Wall $(EXT_FLAGS)" --backend distutils $(DEBUG)

### numpy.distutils build targets ===========================
# deprecated targets for building the extension module using numpy.distutils
# This will not work with python >= 3.12
# pyf interface
interface: $(sources)
	@echo "Creating fortran interfaces for the caete_module using f2py..."
	@$(PYEXEC) precompilation_data.py ## Set the number of PLS in the global module and validate config file
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) $(MFLAG) $(MODNAME) $(OVRTFLAG) --quiet

# Build the shared object (caete_module) from .F90 source
# Enable OpenMP
so_parallel: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90exec="$(FC)" --f77exec="$(FC)" --f90flags="-Wall $(EXT_FLAGS) \
	                                      -fopenmp " $(OMP_LIB) $(DEBUG) --backend distutils --no-freethreading-compatible
	@echo "Parallel version of caete_module created using $(FC)"
#	@echo "Compiling community module..."
#	$(PYEXEC) setup.py build_ext --inplace

# Build the shared object (caete_module) from .F90 source
# Serial version
so: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90exec="$(FC)" --f77exec="$(FC)" --f90flags="-Wall $(EXT_FLAGS) " --quiet $(DEBUG)
	@echo "Serial version of caete_module created using $(FC)"
#	@echo "Compiling community module..."
#	$(PYEXEC) setup.py build_ext --inplace
# =================================================================================================

# Build objects for the DEBUG program
debug: $(objects) so
	$(PYEXEC) create_plsbin.py
	$(FC) -o $(DEBUG_PROGRAM) $(FCFLAGS) $(objects)

global.o: global.f90
	$(FC) $(FLFLAGS) global.f90

types.mod: types.o types.f90
	$(FC) $(FLFLAGS) types.f90

global_par.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

photo_par.mod: photo_par.o photo_par.f90
	$(FC) $(FLFLAGS) photo_par.f90

funcs.o: funcs.f90
	$(FC) $(FLFLAGS) funcs.f90

photo.mod: funcs.o funcs.f90
	$(FC) $(FLFLAGS) funcs.f90

evap.o: evap.f90
	$(FC) $(FLFLAGS) evap.f90

water.mod: evap.o evap.f90
	$(FC) $(FLFLAGS) evap.f90

soil_dec.o: soil_dec.f90
	$(FC) $(FLFLAGS) soil_dec.f90

soil_dec.mod: soil_dec.o soil_dec.f90
	$(FC) $(FLFLAGS) soil_dec.f90

cc.o: cc.f90
	$(FC) $(FLFLAGS) cc.f90

carbon_costs.mod: cc.o cc.f90
	$(FC) $(FLFLAGS) cc.f90

allocation.o: allocation.f90
	$(FC) $(FLFLAGS) allocation.f90

alloc.mod: allocation.o allocation.f90
	$(FC) $(FLFLAGS) allocation.f90

productivity.o: productivity.f90
	$(FC) $(FLFLAGS) productivity.f90

productivity.mod: productivity.o productivity.f90
	$(FC) $(FLFLAGS) productivity.f90

budget.o: budget.F90
	$(FC) $(FLFLAGS) budget.F90

budget.mod: budget.o budget.F90
	$(FC) $(FLFLAGS) budget.F90

debug_caete.o: debug_caete.f90
	$(FC) $(FLFLAGS) debug_caete.f90


# CLEAN targets
clean: 
	rm -rf *.s *.o *.obj *. pls_ex.txt __pycache__ run_debug execution.log ./caete_module/ temp_annual_*/

clean_so: clean
	rm -rf *.so *.pyf *.mod *.psz *.pyd 
	rm -rf build
	clear

reinstall: clean_so ext_mod_meson
