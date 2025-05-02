! I use prints and python scripts most of the time to test the fortran library
! Sometimes we need to step in and debug the fortran code directly,
! so this is a simple example of a fortran program
! that calls the carbon3 subroutine from the caete_module.
! You can implement your own test program to debug the fortran code

program test_caete

   use types
   use global_par
   use photo
   use water
   use soil_dec
   use budget

   implicit none

   print *, "Testing/debugging carbon3 subroutine from caete_module"

   call test_c3()

   contains

   subroutine test_c3()

      integer(i_4) :: index
      real(r_4) :: soilt=25.0, water_s=0.9
      real(r_8) :: ll=5.5, lf=5.5, lw=5.5
      real(r_8), dimension(6) :: lnc = (/2.5461449101567262D-002, 1.2789730913937092D-002, 4.1226762905716891D-002,&
                                        & 3.2206000294536350D-003, 3.1807350460439920D-003, 4.0366222383454442D-003/)
      real(r_8), dimension(4) :: cs = 0.1, cs_out = 0.1
      real(r_8), dimension(8) :: snc = 0.00001, snc_in = 0.0001
      real(r_8) :: hr
      real(r_8) :: nmin, pmin

      do index = 1,200000

         call carbon3(soilt, water_s, ll, lw, lf, lnc, cs, snc_in, cs_out, snc, hr, nmin, pmin)

         cs = cs_out
         snc_in = snc

         print *, snc,"<- snc"
         print *, hr,"<- hr"
         print *, nmin, pmin, "<- N & P"
         print *, cs,"<- cs"
      end do
   end subroutine test_c3

end program test_caete
