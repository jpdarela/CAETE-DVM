program test_caete

   use types
   use global_par
   use photo
   use water
   use soil_dec
   use budget

   implicit none

   print *, "Testing/debugging bdgt"

   call test_c3()
   ! call test_budg()

   contains



! subroutine test_budg()

!    real(r_8),dimension(ntraits,npls) :: dt
!    real(r_8) :: w1, w2   !Initial (previous month last day) soil moisture storage (mm)
!    ! real(r_4),dimension(npls) :: g1   !Initial soil ice storage (mm)
!    ! real(r_4),dimension(npls) :: s1   !Initial overland snow storage (mm)
!    real(r_4) :: ts                   ! Soil temperature (oC)
!    real(r_4) :: temp                 ! Surface air temperature (oC)
!    real(r_4) :: p0                   ! Surface pressure (mb)
!    real(r_4) :: ipar                 ! Incident photosynthetic active radiation mol Photons m-2 s-1
!    real(r_4) :: rh                   ! Relative humidity
!    real(r_4) :: mineral_n            ! Solution N NOx/NaOH gm-2
!    real(r_4) :: labile_p             ! solution P O4P  gm-2
!    real(r_8) :: on, sop, op          ! Organic N, isoluble inorganic P, Organic P g m-2
!    real(r_8) :: catm, wmax_in                 ! ATM CO2 concentration ppm


!    real(r_8),dimension(3,npls)  :: sto_budg_in ! Rapid Storage Pool (C,N,P)  g m-2
!    real(r_8),dimension(npls) :: cl1_in  ! initial BIOMASS cleaf compartment kgm-2
!    real(r_8),dimension(npls) :: cf1_in  !                 froot
!    real(r_8),dimension(npls) :: ca1_in  !                 cawood
!    real(r_8),dimension(npls) :: dleaf_in  ! CHANGE IN cVEG (DAILY BASIS) TO GROWTH RESP
!    real(r_8),dimension(npls) :: droot_in  ! k gm-2
!    real(r_8),dimension(npls) :: dwood_in  ! k gm-2
!    real(r_8),dimension(npls) :: uptk_costs_in ! g m-2

!    !    !# out
!    real(r_4) :: epavg          !Maximum evapotranspiration (mm/day)
!    real(r_8) :: evavg          !Actual evapotranspiration Daily average (mm/day)
!    real(r_8) :: phavg          !Daily photosynthesis (Kg m-2 y-1)
!    real(r_8) :: aravg          !Daily autotrophic respiration (Kg m-2 y-1)
!    real(r_8) :: nppavg         !Daily NPP (average between PFTs)(Kg m-2 y-1)
!    real(r_8) :: laiavg         !Daily leaf19010101', '19551231 area Index m2m-2
!    real(r_8) :: rcavg          !Daily canopy resistence s/m
!    real(r_8) :: f5avg          !Daily canopy resistence s/m
!    real(r_8) :: rmavg          !maintenance/growth respiration (Kg m-2 y-1)
!    real(r_8) :: rgavg          !maintenance/growth respiration (Kg m-2 y-1)
!    real(r_8) :: wueavg         ! Water use efficiency
!    real(r_8) :: cueavg         ! [0-1]
!    real(r_8) :: vcmax_1          ! µmol m-2 s-1
!    real(r_8) :: specific_la_1    ! m2 g(C)-1
!    real(r_8) :: c_defavg       ! kg(C) m-2 Carbon deficit due to negative NPP - i.e. ph < ar
!    real(r_8) :: litter_l_1       ! g m-2
!    real(r_8) :: cwd_1            ! g m-2
!    real(r_8) :: litter_fr_1      ! g m-2
!    real(r_8),dimension(2) :: nupt_1         ! g m-2 (1) from Soluble (2) from organic
!    real(r_8),dimension(3) :: pupt_1         ! g m-2
!    real(r_8),dimension(6) :: lit_nut_content_1 ! g(Nutrient)m-2 ! Lit_nut_content variables
!    !    ! FULL OUTPUT
!    real(r_8),dimension(npls) :: cleafavg_pft   !Carbon in plant tissues (kg m-2)
!    real(r_8),dimension(npls) :: cawoodavg_pft  !
!    real(r_8),dimension(npls) :: cfrootavg_pft  !
!    real(r_8),dimension(npls) :: ocpavg         ! [0-1] Gridcell occupation
!    real(r_8),dimension(3,npls) :: delta_cveg_1
!    real(r_8),dimension(3,npls) :: storage_out_bdgt_1
!    integer(i_2),dimension(3,npls) :: limitation_status_1
!    integer(i_4),dimension(2,npls) :: uptk_strat_1
!    real(r_8),dimension(npls) ::  npp2pay_1 ! C costs of N/P uptake
!    real(r_8),dimension(4) :: cp ! Aux cp(1:3) CVEG C POOLS cp(4) Auxiliary to HR
!    real(r_8) :: c_cost_cwm

!       integer(i_4) :: i,j


!    !    ! Open pls_table

!       open (unit=11,file='pls_ex.txt',status='old',form='formatted',access='sequential',&
!       action='read')


!       do j = 1, npls ! for each line do
!          read(11,*) (dt(i,j), i=1,ntraits)
!       end do

!       close(11)

!       print*, size(dt)
!       w1 = 40.0D0
!       w2 = 280.0D0
!       ts = 23
!       temp = 25.0
!       p0   = 1000.0
!       ipar = 200.0 * 0.5 / 2.18e5
!       rh   = 0.9
!       mineral_n = 10
!       labile_p = 2.0
!       on = 3.0
!       op = 4.0
!       sop = 11.0
!       catm = 400.0
!       sto_budg_in(:,:) = 0.0D0
!       cl1_in(:) = 2.0D0
!       cf1_in(:) = 2.0D0
!       ca1_in(1) = 0.0D0
!       ca1_in(2:) = 10.0D0
!       dleaf_in(:) = 0.0D0
!       droot_in(:) = 0.0D0
!       dwood_in(:) = 0.0D0
!       uptk_costs_in(:) = 0.0D0

!       do j = 1, 1100
!          call daily_budget(dt, w1, w2, ts, temp, p0, ipar, rh&
!         &, mineral_n, labile_p, on, sop, op, catm, sto_budg_in, cl1_in, ca1_in, cf1_in, dleaf_in, dwood_in&
!         &, droot_in, uptk_costs_in, wmax_in, evavg, epavg, phavg, aravg, nppavg&
!         &, laiavg, rcavg, f5avg, rmavg, rgavg, cleafavg_pft, cawoodavg_pft&
!         &, cfrootavg_pft, storage_out_bdgt_1, ocpavg, wueavg, cueavg, c_defavg&
!         &, vcmax_1, specific_la_1, nupt_1, pupt_1, litter_l_1, cwd_1, litter_fr_1, npp2pay_1, lit_nut_content_1&
!         &, delta_cveg_1, limitation_status_1, uptk_strat_1, cp, c_cost_cwm)

!          sto_budg_in = storage_out_bdgt_1
!          cl1_in = cleafavg_pft
!          cf1_in = cfrootavg_pft
!          ca1_in = cawoodavg_pft
!          dleaf_in = delta_cveg_1(1,:)
!          droot_in = delta_cveg_1(2,:)
!          dwood_in = delta_cveg_1(3,:)
!          uptk_costs_in = npp2pay_1

!          print*, uptk_strat_1(1,:)
!          print*, uptk_strat_1(2,:)
!          print*, w1,w2
!          print*,nppavg
!       enddo

!    end subroutine test_budg


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

end program test_caete
