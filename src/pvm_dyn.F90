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

! contacts :: David Montenegro Lapola <lapoladm ( at ) gmail.com>
! Author: JP Darela
! This program is based on the work of those that gave us the INPE-CPTEC-PVM2 model


module pvm_dyn
    implicit none
    private

    public :: run_caete

 contains

 subroutine run_caete(tas)
    use types
    use global_par
    use photo
    use water
    use soil_dec
    use budget

    real(r_4), dimension(366), intent(inout) :: tas(0:365)

    tas(:) = 0.0_r_4

 end subroutine run_caete


 end module pvm_dyn