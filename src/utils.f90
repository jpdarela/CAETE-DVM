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


! Some functions for test/debug CAETÊ
module utils
    use types
    implicit none
    private

    public :: linspace
    public :: process_id
    public :: abort_on_nan
    public :: abort_on_inf
    public :: ascii2bin
    public :: leap
    public :: cwm

    contains

     !=================================================================
     !=================================================================

    subroutine ascii2bin(file_in, file_out, nx1, ny1)
        ! DEPRECATED! Trasnform a matrix (text file) to a binary file
        use types
        !implicit none

        character(len=100), intent(in) :: file_in, file_out
        integer(i_4),intent(in) :: nx1, ny1

        integer(i_4) :: i, j

        real(r_4),allocatable,dimension(:,:) :: arr_in

        allocate(arr_in(nx1,ny1))

        open (unit=11,file=file_in,status='old',form='formatted',access='sequential',&
             action='read')


        open (unit=21,file=file_out,status='unknown',&
             form='unformatted',access='direct',recl=nx1*ny1*r_4)


        do j = 1, ny1 ! for each line do
           read(11,*) (arr_in(i,j), i=1,nx1) ! read all elements in line j (implicitly looping)
           !write(*,*) arr_in(:,j)
        end do

        write(21,rec=1) arr_in
        close(11)
        close(21)

     end subroutine ascii2bin

     !=================================================================
     !=================================================================

     function leap(year) result(is_leap)
        ! Why this is here?
        use types
        !implicit none

        integer(i_4),intent(in) :: year
        logical(l_1) :: is_leap

        logical(l_1) :: by4, by100, by400

        by4 = (mod(year,4) .eq. 0)
        by100 = (mod(year,100) .eq. 0)
        by400 = (mod(year,400) .eq. 0)

        is_leap = by4 .and. (by400 .or. (.not. by100))

     end function leap

     !=================================================================
     !=================================================================

    subroutine abort_on_nan(arg1)

        real(r_4), intent(in) :: arg1

        if (isnan(arg1))then
            print *, "Aborting in nan"
            call abort
            endif
    end subroutine abort_on_nan

     !=================================================================
     !=================================================================

    subroutine abort_on_inf(arg1)

        real(r_4), intent(inout) :: arg1

        if (arg1 .ge. arg1 - 1)then
            print *, "Aborting on inf"
            call abort()
        endif
    end subroutine abort_on_inf

     !=================================================================
     !=================================================================

    subroutine linspace(from, to, array)
    ! DO NOT USE IN PYTHON ENVIRONMENT
    ! THIS function is used only for testing and debugging

        real(r_4), intent(in) :: from, to
        real(r_4), intent(out) :: array(:)
        real(r_4) :: range
        integer :: n, i
        n = size(array)
        range = to - from

        if (n == 0) return

        if (n == 1) then
            array(1) = from
            return
        end if

        do i=1, n
            array(i) = from + range * (i - 1) / (n - 1)
        end do
end subroutine linspace

     !=================================================================
     !=================================================================

function process_id() result(ipid)
    ! Identify process number

       integer(r_4) :: ipid

       ipid = getpid()

    end function process_id

     !=================================================================
     !=================================================================

    function cwm(var_arr, area_arr) result(retval)

        use types
        use global_par

        real(kind=r_8), dimension(npls), intent(in) :: var_arr, area_arr
        real(kind=r_4) :: retval
        retval = real(sum(var_arr * area_arr, mask = .not. isnan(var_arr)), r_4)

     end function cwm

end module utils
