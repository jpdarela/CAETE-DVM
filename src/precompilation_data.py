# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """


import os
from pathlib import Path
from config import fetch_config

caete_config = fetch_config('../src/caete.toml')
# Set the environment variable in the current PowerShell session

descrp = "This script creates a global.f90 file with parameters defined in the caete.toml file."

NTRAITS = caete_config.metacomm.ntraits # type: ignore
NPLS = caete_config.metacomm.npls_max # type: ignore # Max number of Plant Life Strategies per community
OMP_NUM_THREADS = caete_config.multiprocessing.omp_num_threads # type: ignore

global_f90 = f"""
! Copyright 2017- LabTerra

!     This program is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.

!     This program is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with this program.  If not, see <http://www.gnu.org/licenses/>.

! This program is based on the work of those that gave us the INPE-CPTEC-PVM2 model

! This file is generated by the script ask_npls.py
! If you want to change any parameter, please, edit the script and compile the code again

module global_par

   use types

   implicit none
   real(r_4),parameter,public :: q10 = 1.4
   real(r_4),parameter,public :: h = 1.0                         ! soil layer thickness (meters)
   real(r_4),parameter,public :: diffu = 1.036800e14             ! soil thermal diffusivity (m2/mes)
   real(r_4),parameter,public :: tau = (h**2)/(2.0*diffu)        ! e-folding times (months)
   real(r_4),parameter,public :: rcmax = 2000.0                  ! ResistÊncia estomática máxima s/m
   real(r_4),parameter,public :: rcmin = 100.0                   ! ResistÊncia estomática mínima s/m
   real(r_8),parameter,public :: cmin = 1.0D-3                   ! Minimum to survive kg m-2
   ! real(r_4),parameter,public :: wmax = 500.0                  ! Maximum water soil capacity (Kg m-2)

   real(r_8),parameter,public :: csru = 0.5D0                    ! Root attribute
   real(r_8),parameter,public :: alfm = 1.391D0                  ! Root attribute
   real(r_8),parameter,public :: gm = 3.0D0                      ! mm s-1
   real(r_8),parameter,public :: sapwood = 0.05D0                ! Fraction of wood tissues that are sapwood
   real(r_4),parameter,public :: ks = 0.25                       ! P Sorption
   integer(i_4),parameter,public :: npls = {NPLS}                ! Number of Plant Life Strategies-PLSs simulated (Defined at compile time)
   integer(i_4),parameter,public :: ntraits = {NTRAITS}          ! Number of traits for each PLS
   integer(i_4),parameter,public :: omp_nthreads = {OMP_NUM_THREADS} ! Number of OpenMP threads

end module global_par

"""

root = Path(os.getcwd()).resolve()
print(f"\n\n\nSetting the number of Plant Life Strategies (n PLSs = {NPLS}) to the global.f90 file\n\n\n")
with open(f"{root}/global.f90", 'w') as fh:
    fh.write(global_f90)
