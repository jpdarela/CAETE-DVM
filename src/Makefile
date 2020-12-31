# Build caete_module
F2PY = python3.8 -m numpy.f2py
MODNAME = caete_module
INTERFACES = $(MODNAME).pyf
CLEAN_DIR = ./clean_dir.sh

# RUN MODE FLAGS
MFLAG = -m
HFLAG = -h
CFLAG = -c
OVRTFLAG = --overwrite-signature

# DEBUG MODE
FC = gfortran
FCFLAGS = -g -Wall -Wextra -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic
FLFLAGS = -c -g -Wall -fcheck=all -Wextra -ffpe-trap=invalid,zero,overflow,underflow -fbacktrace -fbounds-check -Wconversion -pedantic

# Sources for compilation in run mode (F2PY)
sources = global.f90 utils.f90 cc.f90 funcs.f90 soil_dec.f90 allocation.f90 productivity.f90 budget.f90

# Objects for compilation in debug mode
objects = global.o utils.o funcs.o soil_dec.o debug_caete.o

# Build the interface and the shared object (caete_module) from .F90 source
interface: $(sources)
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) $(MFLAG) $(MODNAME) $(OVRTFLAG) --quiet

so: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90flags="-Wall -fopenmp " -lgomp
	printf "\n\n%s", %''

# Build objects for DEBUG mode

debug: $(objects)
	$(FC) -o run_debug $(FCFLAGS) $(objects)

global.o: global.f90
	$(FC) $(FLFLAGS) global.f90

types.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

global_par.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

photo_par.mod: global.o global.f90
	$(FC) $(FLFLAGS) global.f90

utils.o: utils.f90
	$(FC) $(FLFLAGS) utils.f90

utils.mod: utils.o utils.f90
	$(FC) $(FLFLAGS) utils.f90

funcs.o: funcs.f90
	$(FC) $(FLFLAGS) funcs.f90

photo.mod: funcs.o funcs.f90
	$(FC) $(FLFLAGS) funcs.f90

water.mod: funcs.o funcs.f90
	$(FC) $(FLFLAGS) funcs.f90

soil_dec.o: soil_dec.f90
	$(FC) $(FLFLAGS) soil_dec.f90

soil_dec.mod: soil_dec.o soil_dec.f90
	$(FC) $(FLFLAGS) soil_dec.f90

debug_caete.o: debug_caete.f90
	$(FC) $(FLFLAGS) debug_caete.f90


# CLEAN DIR
clean:
	rm -rf test *.mod *.s *.o pls_ex.txt outputs __pycache__
	$(CLEAN_DIR)
	clear

clean_plsgen:
	rm -rf wallo.npy
	rm -rf gallo.npy
