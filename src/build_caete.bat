@REM Building CAETE using gfortran and gcc in Windows

@REM Working with python 3.13 and meson already. However, the .dll file is not being saved. The extension module cannot find the dll (me neither).

REM Configuration
set MODULE_NAME=caete_module

REM Adjust the path to your Python installation.
REM This points to the folder containing Python.h header file.
set PYTHON_INCLUDE=C:\Users\darel\opt\Python-3.13.5\PCbuild\amd64

REM MinGW paths
set MINGW_BIN=C:\mingw64\bin
set MINGW_LIB=C:\mingw64\lib

REM Prepend MinGW to PATH so meson/f2py detect GCC/MinGW first (avoid MSVC selection)
set PATH=%MINGW_BIN%;%PATH%

REM Force compilers to MinGW
set CC=%MINGW_BIN%\gcc.exe
set CXX=%MINGW_BIN%\g++.exe
set FC=%MINGW_BIN%\gfortran.exe
set AR=%MINGW_BIN%\ar.exe
set RANLIB=%MINGW_BIN%\ranlib.exe

REM Source files
set SOURCE_FILES=types.f90 global.f90 photo_par.f90 funcs.f90 evap.f90 soil_dec.f90 cc.f90 allocation.f90 productivity.f90 .\budget_fixed.F90

REM Build using MinGW (use gnu95 Fortran and MinGW compiler) Adapt the python version as needed
py -3.11 -m numpy.f2py -m %MODULE_NAME%  --build-dir ./build -c %SOURCE_FILES% -I%PYTHON_INCLUDE% -L%MINGW_LIB% -lgfortran -lquadmath --fcompiler=gnu95 --compiler=mingw32