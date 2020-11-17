# CAETÊ
This is the implementation of the Dynamic version of CAETÊ - including Nitrogen and Phosphorus cycling

The code in this repository is based on an earlier version of CAETÊ that was not dynamic and did not have the N and P cycling implemented.

THis is part of my ongoing PhD research project.

# Development Dependencies
CAETÊ depends on a few packages that must be installed beforehand.
You can installed them using your favorite package manager: `apt`, `brew`, `pip`, `poetry`, etc.

General Dependencies
- make (for building automation)

Python Dependencies
- numpy
- f2py (part of numpy)
- cftime
- pandas
- matplotlib
- ipython

Make sure you have them properly installed before running the code.

# Running and Developing CAETÊ
~This section suposses you have a working python environment and an installed fortran compiler.
If you need help setting up your invornment, check the [development environment section](#development-Environment)

CAETÊ uses both Python and Fortran and uses `f2py` module to create an interface between them. This means that the Fortran code must be compiled before you can run Python code.

The Makefile inside `/src` folder have useful automation to make it easier.

`make clean` - it clear your python cache, deletes the `/output` folder and deletes all compiled fortran files, including the `.pyf` file.

`make so` - compiles all fortran code and creates the `caete_module.pyf`, the interface between Fortran and Python code.

To run CAETÊ you can do the following:
```bash
# Clean you cache
make clean

# Build caete_module.pyf
make so

# Run the model
python model_driver.py
```

You can also run it interactively inside `ipython` to have direct access to the variables or you can also run it directly from the vscode debug environment.

# Development Environment
If you need help configuring your development environment, installing python, installing CAETÊ dependencies or setting up a debug enviornment in vscode, check the [CAETÊ starting packtutorial](https://github.com/fmammoli/CAETE-Tutorials)

## __AUTHORS__:

 - Bianca Rius
 - David Lapola
 - Helena Alves
 - João Paulo Darela Filho
 - Put you name here! (labterra@unicamp.br)
