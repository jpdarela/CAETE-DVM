# Building the CAETE Source Code

[⇦ Back](./system_config.md)

Before running the model, you need to compile the Python extension module (Fortran backend). You will need:

### Requirements

1. **C Compiler**
   - Windows: MSVC (mandatory)
   - Linux/macOS: `gcc`

2. **Fortran Compiler**
   - Windows: Intel OneAPI (install after MSVC)
   - The gfortan from WinLibs (GCC 14.2.0 (with MCF threads) + MinGW-w64 12.0.0 (UCRT) - release 1) can also be used in windows along with MSVC.
   - Linux/macOS: `gfortran` (also usable on Windows)
   - The flang [AOCC](https://docs.amd.com/r/en-US/57222-AOCC-user-guide/Introduction) compiler is supported in linux. You will need to set some variables in the Makefile to use it.

3. **Python Environment**
   - Windows: `python == 3.11`
   - Linux/macOS: `python >= 3.11 <= 3.13`
   - Must support:
     - `pip` for package installation
     - `numpy.f2py` for compiling Fortran modules

Refer to the [NumPy v2.3 Manual – Three Ways to Wrap](https://numpy.org/doc/stable/f2py/getting-started.html) and try to reproduce the "quick way". If the "quick way" works in your machine, you will likely be able to compile CAETE's Python C/API extension and run the model locally.

### Python environment management

There are several options to install and manage multiple Python versions in all platforms. For our purposes, we need a python installation that meets the requirements above. Additionally, the python installation must be reachable from the terminal/shell where you will run the Makefile to build the Fortran/Python extension module. This means that the python executable must be in the PATH environment variable or can be launched via an environment manager like pyenv, uv, PIM, and et cetera. Below are some suggestions for managing python versions in different platforms. 

#### Windows
To run CAETE, it is important that you have a python version that meets the requirements (see above). The Python install Manager [PIM](https://www.python.org/downloads/release/pymanager-250/) is a windows tool to help install and manage multiple Python versions easily. The current python manager (for python <=3.13) is also an option for windows users.

#### Linux/MacOS
In linux/macOS, `pyenv` is a popular python version manager. You can find installation instructions in the [pyenv github repository](https://github.com/pyenv/pyenv)

### Caveats

Another very important aspect is that this python installation must be compatible with the C compiler installed in your system. More specifically, the python installation must be compiled with the same C runtime library as the C compiler used to build the Fortran/Python extension module. In windows, this usually means that you should use the official python installer from python.org, since it is compiled with MSVC. Other python distributions like Anaconda may use different C runtimes and may lead to issues when building the extension module with MSVC.

- On Windows, install MSVC first, then OneAPI.
- Ensure `pip` is available to install additional Python packages.
- On Linux, use `pyenv` to manage Python versions.
- On Windows, use the official Python 3.11 installer.
- Other tools like `conda` or `uv` may work but are not guaranteed.
- In windows, gfortran from [WinLibs](https://winlibs.com/#download-release) can do the job:
    Tested with: GCC 14.2.0 (with MCF threads) + MinGW-w64 12.0.0 (UCRT) - release 1

---

# Dangers of Using System Python in Linux

Using the system Python for development can compromise stability, security, and reproducibility.

### Risks

- **System Dependency Conflicts**
  - System Python is used by core utilities and package managers.
  - Modifying it can break your system.

- **Permission and Security Issues**
  - Global installs require `sudo`, increasing risk.
  - Vulnerable to shell injection if user input is unsanitized.

- **Version Lock-In**
  - You are stuck with the distribution's Python version.
  - Limits access to newer features and packages.

- **Reproducibility Problems**
  - Code may behave differently across systems.
  - Pre-installed packages vary by distribution.

---

# Double-Check Your Python Version

Ensure your Python version matches the platform requirements before compiling or running CAETE.

---

# Makefile Usage

- **Linux/macOS**: Use `Makefile` with GNU Make
- **Windows**: Use `Makefile_win` with NMake (adapted for MSVC and OneAPI)

---

# Running the Model

Once compiled and configured, you can proceed to run CAETE using the prepared input files and gridlists.

[⇦ Back](./README.md)
