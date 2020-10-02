# Build caete_module
F2PY = python3.8 -m numpy.f2py
MODNAME = caete_module
INTERFACES = $(MODNAME).pyf
CLEAN_DIR = ./clean_dir.sh
MFLAG = -m
HFLAG = -h
CFLAG = -c
OVRTFLAG = --overwrite-signature

sources = global.f90 utils.f90 cc.f90 funcs.f90 soil_dec.f90 allocation.f90 productivity.f90 budget.f90

interface: $(sources)
	$(F2PY) $(HFLAG) $(INTERFACES) $(sources) $(MFLAG) $(MODNAME) $(OVRTFLAG) --quiet

so: $(sources) interface
	$(F2PY)  $(INTERFACES) $(CFLAG) $(sources) --f90flags="-Wall -fopenmp " -lgomp
	printf "\n\n%s", %''

clean:
	rm -rf test *.mod *.s *.o pls_ex.txt outputs __pycache__
	$(CLEAN_DIR)
	clear
