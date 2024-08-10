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

# Build caete_module & debug programs
PYEXEC = python
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
FC = gfortran
FCFLAGS = -g -Wall -Wextra -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic
FLFLAGS = -c -g -Wall -fcheck=all -Wextra -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic

# gfrotran FLAGS for f2py
EXT_FLAGS = -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans

# Sources for compilation in run mode (F2PY)
src_lib = global.f90 funcs.f90 evap.f90 soil_dec.f90 cc.f90 allocation.f90 productivity.f90
sources = $(src_lib) budget.F90


# Objects for compilation in debug mode
src_obj = global.o funcs.o evap.o soil_dec.o cc.o allocation.o productivity.o
objects = $(src_obj) budget.o debug_caete.o

# Targets
.PHONY: setup interface so so_parallel clean clean_so modules

# Python setup
setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

# Build the interface and the shared object (caete_module) from .F90 source
interface: $(sources)
	@echo "Creating fortran interfaces for the caete_module using f2py..."
	@$(PYEXEC) set_npls.py ## Set the number of PLS in the global module
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) $(MFLAG) $(MODNAME) $(OVRTFLAG) --quiet

so_parallel: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90flags="-Wall $(EXT_FLAGS) -fopenmp " -lgomp --quiet
	@echo "Parallel version of caete_module created using gfortran"

so: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90flags="-Wall $(EXT_FLAGS) " --quiet
	@echo "Serial version of caete_module created using gfortran"


modules: $(sources)
	$(FC) -c $(sources)

# Build objects for DEBUG mode
debug: $(objects) so
	$(PYEXEC) create_plsbin.py
	$(FC) -o $(DEBUG_PROGRAM) $(FCFLAGS) $(objects)

global.o: global.f90
	$(FC) $(FLFLAGS) global.f90

types.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

global_par.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

photo_par.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

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

# CLEAN
clean:
	rm -rf *.s *.o pls_ex.txt __pycache__ run_debug logfile.log

clean_so: clean
	rm -rf *.so *.pyf *.mod
	clear
