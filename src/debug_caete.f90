program test_caete

   use types
   use utils
   use global_par
   use photo
   use water
   use soil_dec

   implicit none


   print *,
   print *,
   print *, "Testing/debugging CARBON3"

    call test_c3()


   contains

   ! TEST CARBON3
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

   subroutine test_budg()
   
      ! real(r_8),dimension(ntraits,npls),intent(in) :: dt
      ! real(r_4),dimension(npls),intent(in) :: w1   !Initial (previous month last day) soil moisture storage (mm)
      ! real(r_4),dimension(npls),intent(in) :: g1   !Initial soil ice storage (mm)
      ! real(r_4),dimension(npls),intent(in) :: s1   !Initial overland snow storage (mm)
      ! real(r_4),intent(in) :: ts                   ! Soil temperature (oC)
      ! real(r_4),intent(in) :: temp                 ! Surface air temperature (oC)
      ! real(r_4),intent(in) :: prec                 ! Precipitation (mm/day)
      ! real(r_4),intent(in) :: p0                   ! Surface pressure (mb)
      ! real(r_4),intent(in) :: ipar                 ! Incident photosynthetic active radiation mol Photons m-2 s-1
      ! real(r_4),intent(in) :: rh                   ! Relative humidity
      ! real(r_4),intent(in) :: mineral_n            ! Solution N NOx/NaOH gm-2
      ! real(r_4),intent(in) :: labile_p             ! solution P O4P  gm-2
      ! real(r_8),intent(in) :: on, sop, op          ! Organic N, isoluble inorganic P, Organic P g m-2
      ! real(r_8),intent(in) :: catm                 ! ATM CO2 concentration ppm


      ! real(r_8),dimension(3,npls),intent(in)  :: sto_budg ! Rapid Storage Pool (C,N,P)  g m-2
      ! real(r_8),dimension(npls),intent(in) :: cl1_pft  ! initial BIOMASS cleaf compartment kgm-2
      ! real(r_8),dimension(npls),intent(in) :: cf1_pft  !                 froot
      ! real(r_8),dimension(npls),intent(in) :: ca1_pft  !                 cawood
      ! real(r_8),dimension(npls),intent(in) :: dleaf  ! CHANGE IN cVEG (DAILY BASIS) TO GROWTH RESP
      ! real(r_8),dimension(npls),intent(in) :: droot  ! k gm-2
      ! real(r_8),dimension(npls),intent(in) :: dwood  ! k gm-2
      ! real(r_8),dimension(npls),intent(in) :: uptk_costs ! g m-2

   end subroutine test_budg

end program test_caete
