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

module budget
   use types
   use global_par, only: ntraits, npls, omp_nthreads
   use alloc
   use productivity
   use ieee_arithmetic
   implicit none
   private

   public :: daily_budget

contains
!> daily budget subroutine. To be called for each gridcell each day
!> @param dt PLS traits array
!


   subroutine daily_budget(dt, w1, w2, ts, temp, p0, ipar, rh&
        &, mineral_n, labile_p, on, sop, op, catm, sto_budg_in, cl1_in, ca1_in, cf1_in&
        &, uptk_costs_in, wmax_in, rnpp, evavg, epavg, phavg, aravg, nppavg, laiavg, rcavg&
        &,  f5avg, rmavg, rgavg, cleafavg_pft, cawoodavg_pft, cfrootavg_pft&
        &, storage_out_bdgt_1, ocpavg, wueavg, cueavg, c_defavg, vcmax_1&
        &, specific_la_1, nupt_1, pupt_1, litter_l_1, cwd_1, litter_fr_1, npp2pay_1, lit_nut_content_1&
        &, limitation_status_1, uptk_strat_1, cp, c_cost_cwm, rnpp_out)




#ifdef _OPENMP
   use omp_lib
#endif
      !f2py threadsafe
      use photo, only: pft_area_frac, sto_resp
      use water, only: evpot2, penman, available_energy, runoff

      !     ----------------------------INPUTS-------------------------------
      real(r_8),dimension(ntraits,npls),intent(in) :: dt
      real(r_8),intent(in) :: w1, w2   !Initial (previous month last day) soil moisture storage (mm)
      ! real(r_8),dimension(npls),intent(in) :: g1   !Initial soil ice storage (mm)
      ! real(r_8),dimension(npls),intent(in) :: s1   !Initial overland snow storage (mm)
      real(r_8),intent(in) :: ts                   ! Soil temperature (oC)
      real(r_8),intent(in) :: temp                 ! Surface air temperature (oC)
      real(r_8),intent(in) :: p0                   ! Surface pressure (mb)
      real(r_8),intent(in) :: ipar                 ! Incident photosynthetic active radiation mol Photons m-2 s-1
      real(r_8),intent(in) :: rh                   ! Relative humidity
      real(r_8),intent(in) :: mineral_n            ! Solution N NOx/NaOH gm-2
      real(r_8),intent(in) :: labile_p             ! solution P O4P  gm-2
      real(r_8),intent(in) :: on, sop, op          ! Organic N, isoluble inorganic P, Organic P g m-2
      real(r_8),intent(in) :: catm, wmax_in        ! ATM CO2 concentration ppm


      real(r_8),dimension(3,npls),intent(in)  :: sto_budg_in ! Rapid Storage Pool (C,N,P)  g m-2
      real(r_8),dimension(npls),intent(in) :: cl1_in  ! initial BIOMASS cleaf compartment kgm-2
      real(r_8),dimension(npls),intent(in) :: cf1_in  !                 froot
      real(r_8),dimension(npls),intent(in) :: ca1_in  !                 cawood
      real(r_8),dimension(npls),intent(in) :: uptk_costs_in ! g m-2
      real(r_8),dimension(npls),intent(in) :: rnpp ! NPP (g m-2 day-1) construction from previous day


      !     ----------------------------OUTPUTS------------------------------
      real(r_8),intent(out) :: epavg          !Maximum evapotranspiration (mm/day)
      real(r_8),intent(out) :: evavg          !Actual evapotranspiration Daily average (mm/day)
      real(r_8),intent(out) :: phavg          !Daily photosynthesis (Kg m-2 y-1)
      real(r_8),intent(out) :: aravg          !Daily autotrophic respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: nppavg         !Daily NPP (average between PFTs)(Kg m-2 y-1)
      real(r_8),intent(out) :: laiavg         !Daily leaf19010101', '19551231 area Index m2m-2
      real(r_8),intent(out) :: rcavg          !Daily canopy resistence s/m
      real(r_8),intent(out) :: f5avg          !Daily canopy resistence s/m
      real(r_8),intent(out) :: rmavg          !maintenance/growth respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: rgavg          !maintenance/growth respiration (Kg m-2 y-1)
      real(r_8),intent(out) :: wueavg         ! Water use efficiency
      real(r_8),intent(out) :: cueavg         ! [0-1]
      real(r_8),intent(out) :: vcmax_1          ! µmol m-2 s-1
      real(r_8),intent(out) :: specific_la_1    ! m2 g(C)-1
      real(r_8),intent(out) :: c_defavg       ! kg(C) m-2 Carbon deficit due to negative NPP - i.e. ph < ar
      real(r_8),intent(out) :: litter_l_1       ! g m-2
      real(r_8),intent(out) :: cwd_1            ! g m-2
      real(r_8),intent(out) :: litter_fr_1      ! g m-2
      real(r_8),dimension(2),intent(out) :: nupt_1         ! g m-2 (1) from Soluble (2) from organic
      real(r_8),dimension(3),intent(out) :: pupt_1         ! g m-2
      real(r_8),dimension(6),intent(out) :: lit_nut_content_1 ! g(Nutrient)m-2 ! Lit_nut_content variables         [(lln),(rln),(cwdn),(llp),(rl),(cwdp)]

      ! FULL OUTPUT
      real(r_8),dimension(npls),intent(out) :: cleafavg_pft   !Carbon in plant tissues (kg m-2)
      real(r_8),dimension(npls),intent(out) :: cawoodavg_pft  !
      real(r_8),dimension(npls),intent(out) :: cfrootavg_pft  !
      real(r_8),dimension(npls),intent(out) :: ocpavg         ! [0-1] Gridcell occupation
      real(r_8),dimension(npls),intent(out) :: rnpp_out       ! NPP (g m-2 day-1) construction

      real(r_8),dimension(3,npls),intent(out) :: storage_out_bdgt_1
      integer(i_2),dimension(3,npls),intent(out) :: limitation_status_1
      integer(i_4),dimension(2,npls),intent(out) :: uptk_strat_1
      real(r_8),dimension(npls),intent(out) ::  npp2pay_1 ! C costs of N/P uptake
      real(r_8),dimension(4),intent(out) :: cp ! Aux cp(1:3) CVEG C POOLS cp(4) Auxiliary to HR
      real(r_8),intent(out) :: c_cost_cwm
      !     -----------------------Internal Variables------------------------
      integer(i_4) :: p, counter, nlen, ri, i, j
      real(r_8),dimension(ntraits) :: dt1 ! Store one PLS attributes array (1D)
      real(r_8) :: carbon_in_storage ! Carbon stored in the storage pool
      real(r_8) :: testcdef ! Test if the carbon deficit can be compensated by stored carbon
      real(r_8) :: sr, mr_sto, growth_stoc ! Storage growth respiration, maintenance respiration of stored C, growth of storage C pool
      real(r_8),dimension(npls) :: ocp_mm  ! Biomass based abundance
      integer(i_4),dimension(npls) :: ocp_wood ! 1 if PLS is not shaded
      integer(i_4),dimension(npls) :: run ! 1 if PLS is alive


      real(r_8),dimension(npls) :: cl1_pft, cf1_pft, ca1_pft ! Initial carbon pools kg m-2
      real(r_8) :: soil_temp ! Soil temperature (oC)
      real(r_8) :: emax ! Maximum evapotranspiration (mm/day)
      real(r_8) :: w ! Soil moisture storage (mm)
      real(r_8),dimension(npls) :: construction ! Allocated NPP

      real(r_8),dimension(:),allocatable :: ocp_coeffs ! Occupation coefficients Biomass baseed abundance (Biomas-Ratio hypothesis)

      real(r_8),dimension(:),allocatable :: evap   !Actual evapotranspiration (mm/day)
      !c     Carbon Cycle
      real(r_8),dimension(:),allocatable :: ph     !Canopy gross photosynthesis (kgC/m2/yr)
      real(r_8),dimension(:),allocatable :: ar     !Autotrophic respiration (kgC/m2/yr)
      real(r_8),dimension(:),allocatable :: nppa   !Net primary productivity / auxiliar
      real(r_8),dimension(:),allocatable :: laia   !Leaf area index (m2 leaf/m2 area)
      real(r_8),dimension(:),allocatable :: rc2    !Canopy resistence (s/m)
      real(r_8),dimension(:),allocatable :: f5     !Photosynthesis (mol/m2/s)
      real(r_8),dimension(:),allocatable :: vpd    !Vapor Pressure deficit
      real(r_8),dimension(:),allocatable :: rm     !maintenance & growth a.resp
      real(r_8),dimension(:),allocatable :: rg     !maintenance & growth a.resp
      real(r_8),dimension(:),allocatable :: wue    !Water use efficiency
      real(r_8),dimension(:),allocatable :: cue    !Carbon use efficiency
      real(r_8),dimension(:),allocatable :: c_def  !Carbon deficit due to negative NPP - i.e. ph < ar
      real(r_8),dimension(:),allocatable :: cl1_int ! carbon auxiliar allocation
      real(r_8),dimension(:),allocatable :: cf1_int ! carbon auxiliar allocation
      real(r_8),dimension(:),allocatable :: ca1_int ! carbon auxiliar allocation
      real(r_8),dimension(:),allocatable :: tra    !transpiration (mm/day) or (mm/s)
      real(r_8),dimension(:),allocatable :: cl2    ! carbon pos-allocation
      real(r_8),dimension(:),allocatable :: cf2    ! carbon pos-allocation
      real(r_8),dimension(:),allocatable :: ca2    ! carbon pos-allocation
      real(r_8),dimension(:,:),allocatable :: day_storage ! D0=3 g m-2
      real(r_8),dimension(:),allocatable   :: vcmax            ! µmol m-2 s-1
      real(r_8),dimension(:),allocatable   :: specific_la      ! m2 g(C)-1
      real(r_8),dimension(:,:),allocatable :: nupt             !d0 =2      ! g m-2 (1) from Soluble (2) from organic
      real(r_8),dimension(:,:),allocatable :: pupt             !d0 =3      ! g m-2
      real(r_8),dimension(:),allocatable   :: litter_l         ! g m-2
      real(r_8),dimension(:),allocatable   :: cwd              ! g m-2
      real(r_8),dimension(:),allocatable   :: litter_fr        ! g m-2
      real(r_8),dimension(:),allocatable   :: npp2pay          ! G M-2
      real(r_8),dimension(:,:),allocatable :: lit_nut_content  ! d0=6 g(Nutrient)m-2 ! Lit_nut_content variables         [(lln),(rln),(cwdn),(llp),(rl),(cwdp)]
      real(r_8),dimension(:,:),allocatable :: storage_out_bdgt ! d0 = 3
      real(r_8),dimension(:),allocatable :: ar_fix_hr ! store plss npp (C) exchanged with N fixer µorganisms GOTO HR
      integer(i_2),dimension(:,:),allocatable   :: limitation_status ! D0=3
      integer(i_4), dimension(:, :),allocatable :: uptk_strat        ! D0=2
      INTEGER(i_4), dimension(:), allocatable :: lp ! index of living PLSs/living grasses
      real(r_8), dimension(:), allocatable :: rnpp_aux ! NPP (g m-2 day-1) construction tracking for each living PLS

      real(r_8), dimension(npls) :: awood_aux, uptk_costs, pdia_aux
      real(r_8), dimension(:, :), allocatable :: sto_budg
      real(r_8) :: soil_sat, ar_aux, total_c, c_def_amount, cl_def, ca_def, cf_def
      real(r_8), dimension(:), allocatable :: idx_grasses, idx_pdia


      allocate(sto_budg(3, npls))

      !     START
      !     --------------
      !     Grid cell area fraction 0-1
      !     ============================
      ! create copies of some input variables (arrays) - ( they are passed by reference by standard)
      do i = 1,npls
         awood_aux(i) = dt(7,i)
         pdia_aux(i) = dt(17,i)
         cl1_pft(i) = cl1_in(i)
         ca1_pft(i) = ca1_in(i)
         cf1_pft(i) = cf1_in(i)
         uptk_costs(i) = uptk_costs_in(i)
         do j = 1,3
            sto_budg(j,i) = sto_budg_in(j,i)
         enddo
      enddo

      ! find the number of grasses

      w = w1 + w2      ! soil water mm
      soil_temp = ts   ! soil temp °C
      soil_sat = wmax_in

      call pft_area_frac(cl1_pft, cf1_pft, ca1_pft, awood_aux,&
      &                  ocpavg, ocp_wood, run, ocp_mm)

      nlen = sum(run)    ! New length for the arrays in the main loop
      allocate(lp(nlen))
      allocate(ocp_coeffs(nlen))
      allocate(idx_grasses(nlen))
      allocate(idx_pdia(nlen))
      allocate(ar_fix_hr(nlen))

      idx_grasses(:) = 1.0D0
      idx_pdia(:) = 1.0D0

      ! Get only living PLSs
      counter = 1
      do p = 1,npls
         if(run(p).eq. 1) then
            lp(counter) = p
            ocp_coeffs(counter) = ocpavg(p)
            counter = counter + 1
         endif
      enddo

      ! Identify grasses
      do p = 1, nlen
         if (awood_aux(lp(p)) .le. 0.0D0) idx_grasses(p) = 0.0D0
      enddo
      ! Identify Nfixers
      do p = 1, nlen
         if (pdia_aux(lp(p)) .le. 0.0D0) idx_pdia(p) = 0.0D0
      enddo

      allocate(rnpp_aux(nlen))
      allocate(evap(nlen))
      allocate(nppa(nlen))
      allocate(ph(nlen))
      allocate(ar(nlen))
      allocate(laia(nlen))
      allocate(f5(nlen))
      allocate(vpd(nlen))
      allocate(rc2(nlen))
      allocate(rm(nlen))
      allocate(rg(nlen))
      allocate(wue(nlen))
      allocate(cue(nlen))
      allocate(c_def(nlen))
      allocate(vcmax(nlen))
      allocate(specific_la(nlen))
      allocate(storage_out_bdgt(3, nlen))
      allocate(tra(nlen))
      allocate(nupt(2, nlen))
      allocate(pupt(3, nlen))
      allocate(litter_l(nlen))
      allocate(cwd(nlen))
      allocate(litter_fr(nlen))
      allocate(lit_nut_content(6, nlen))
      allocate(npp2pay(nlen))
      allocate(limitation_status(3,nlen))
      allocate(uptk_strat(2,nlen))
      allocate(cl1_int(nlen))
      allocate(cf1_int(nlen))
      allocate(ca1_int(nlen))
      allocate(cl2(nlen))
      allocate(cf2(nlen))
      allocate(ca2(nlen))
      allocate(day_storage(3,nlen))

      ! Initialize all allocated arrays to zero
      rnpp_aux(:) = 0.0D0
      evap(:) = 0.0D0
      nppa(:) = 0.0D0
      ph(:) = 0.0D0
      ar(:) = 0.0D0
      laia(:) = 0.0D0
      f5(:) = 0.0D0
      vpd(:) = 0.0D0
      rc2(:) = 0.0D0
      rm(:) = 0.0D0
      rg(:) = 0.0D0
      wue(:) = 0.0D0
      cue(:) = 0.0D0
      c_def(:) = 0.0D0
      vcmax(:) = 0.0D0
      specific_la(:) = 0.0D0
      storage_out_bdgt(:,:) = 0.0D0
      tra(:) = 0.0D0
      nupt(:,:) = 0.0D0
      pupt(:,:) = 0.0D0
      litter_l(:) = 0.0D0
      cwd(:) = 0.0D0
      litter_fr(:) = 0.0D0
      lit_nut_content(:,:) = 0.0D0
      npp2pay(:) = 0.0D0
      limitation_status(:,:) = 0.0D0
      uptk_strat(:,:) = 0.0D0
      cl1_int(:) = 0.0D0
      cf1_int(:) = 0.0D0
      ca1_int(:) = 0.0D0
      cl2(:) = 0.0D0
      cf2(:) = 0.0D0
      ca2(:) = 0.0D0
      day_storage(:,:) = 0.0D0

      !     Maximum evapotranspiration   (emax)
      !     =================================
      emax = evpot2(p0,temp,rh,available_energy(temp))
      soil_temp = ts

      !     Productivity & Growth (ph, ALLOCATION, aresp, vpd, rc2 & etc.) for each PLS
      !     =====================make it parallel=========================
#ifdef _OPENMP
   call OMP_SET_NUM_THREADS(omp_nthreads)
   if (nlen .lt. 200) call OMP_SET_NUM_THREADS(1)
#endif

      construction = rnpp ! construction (Real NPP) of the plant tissues in the previous day. To calculate growth respiration

      !$OMP PARALLEL DO &
      !$OMP SCHEDULE(AUTO) &
      !$OMP DEFAULT(SHARED) &
      !$OMP PRIVATE(p, ri, carbon_in_storage, testcdef, sr, dt1, mr_sto, growth_stoc,&
      !$OMP         ar_aux, total_c, c_def_amount, cl_def, ca_def, cf_def)
      do p = 1,nlen
         carbon_in_storage = 0.0D0
         testcdef = 0.0D0
         sr = 0.0D0
         ar_aux = 0.0D0
         ri = lp(p)
         dt1 = dt(:,ri) ! Pick up the pls functional attributes list

         ! TODO: input storage pool to prod function; We need to calculate the respiratory costs  of storage inside
         ! use the dleaf dwwood and droot trio.
         call prod(dt1, ocp_wood(ri),catm, temp, soil_temp, p0, w, ipar, rh, emax&
               &, cl1_pft(ri), ca1_pft(ri), cf1_pft(ri), soil_sat, construction(ri)&
               &, ph(p), ar(p), nppa(p), laia(p), f5(p), vpd(p), rm(p), rg(p), rc2(p)&
               &, wue(p), c_def(p), vcmax(p), specific_la(p), tra(p))

         ! TODO: Implement the degree of coupling between the soil-vegetation-atmosphere system using stomatal conductance and VPD
         ! as in Köstner et al. 1992
         evap(p) = penman(p0,temp,rh,available_energy(temp),rc2(p)) !Actual evapotranspiration (evap, mm/day)

         ! Check if the carbon deficit can be compensated by stored carbon

         carbon_in_storage = sto_budg(1, ri)
         ! print *, "Carbon in storage: ", carbon_in_storage
         storage_out_bdgt(1, p) = carbon_in_storage
         if (c_def(p) .gt. 0.0) then
            ! print *, "Carbon deficit: ", c_def(p)
            ! print *, "====----===="
            ! print *, ""
            testcdef = c_def(p) - carbon_in_storage
            if(testcdef .lt. 0.0) then
               storage_out_bdgt(1, p) = carbon_in_storage - c_def(p)
               c_def(p) = 0.0D0
            else
               storage_out_bdgt(1, p) = 0.0D0
               c_def(p) = real(testcdef, kind=r_8)       ! testcdef is zero or positive
            endif
         endif
         carbon_in_storage = 0.0D0
         testcdef = 0.0D0

         ! calculate maintanance respirarion of stored C
         mr_sto = sto_resp(temp, storage_out_bdgt(:,p))
         if (ieee_is_nan(mr_sto)) mr_sto = 0.0D0
         if (mr_sto .gt. 0.1D2) mr_sto = 0.0D0
         storage_out_bdgt(1,p) = max(0.0D0, (storage_out_bdgt(1,p) - mr_sto))

         !     Carbon/Nitrogen/Phosphorus allocation/deallocation
         !     =====================================================
         ! print *, "Allocation, PLS:", p, " ri: ", ri
         ! print *, "STO in before allocation: ", storage_out_bdgt(:,p)
         ! print *, "STO out: ", day_storage(:,p)  ! Added print statement for storage out
         ! print *, "mr_sto: ", mr_sto
         ! print *, "c_def: ", c_def(p)


         call allocation (dt1,nppa(p),uptk_costs(ri), soil_temp, w, tra(p)&
            &, mineral_n,labile_p, on, sop, op, cl1_pft(ri),ca1_pft(ri)&
            &, cf1_pft(ri),storage_out_bdgt(:,p),day_storage(:,p),cl2(p),ca2(p)&
            &, cf2(p),litter_l(p),cwd(p), litter_fr(p),nupt(:,p),pupt(:,p)&
            &, lit_nut_content(:,p), limitation_status(:,p), npp2pay(p), uptk_strat(:, p)&
            &, ar_aux, rnpp_aux(p))

            ! print *, "STO_OUT: out from allocation ", day_storage(:,p)
            ! print *, "END Allocation, PLS:", p, " ri: ", ri
            ! print *, "------------------------------------"
         ! Estimate growth of storage C pool
         ar_fix_hr(p) = ar_aux
         growth_stoc = max( 0.0D0, (day_storage(1,p) - storage_out_bdgt(1,p)))
         if (ieee_is_nan(growth_stoc)) growth_stoc = 0.0D0
         if (growth_stoc .gt. 0.1D3) growth_stoc = 0.0D0
         storage_out_bdgt(:,p) = day_storage(:,p)

         ! Calculate storage GROWTH respiration
         sr = 0.25D0 * growth_stoc ! g m-2
         if(sr .gt. 1.0D2) sr = 0.0D0
         ar(p) = ar(p) + real(((sr + mr_sto) * 0.365242), kind=r_8) ! Convert g m-2 day-1 in kg m-2 year-1
         storage_out_bdgt(1, p) = storage_out_bdgt(1, p) - sr
         ! print *, "STO in: ", storage_out_bdgt(:,p)
         ! print *, "STO out: ", day_storage(:,p)


         growth_stoc = 0.0D0
         mr_sto = 0.0D0
         sr = 0.0D0

         ! CUE
         if(ph(p) .eq. 0.0 .or. nppa(p) .eq. 0.0) then
            cue(p) = 0.0
         else
            cue(p) = nppa(p)/ph(p)
         endif

         ! Mass Balance. If ar > A then the population is not able to maintain itself. It loses the C
         ! c_def in g m-2 and cx2 in kg m-2
         if (c_def(p) .gt. 0.0) then
            ! Calculate the total carbon available
            total_c = cl2(p) + ca2(p) + cf2(p)
            c_def_amount = c_def(p) * 1e-3
            ca_def = 0.0D0
            cf_def = 0.0D0
            cl_def = 0.0D0

            ! Check if the total carbon is less than c_def
            if (total_c .lt. c_def_amount) then
               cl1_int(p) = 0.0D0
               ca1_int(p) = 0.0D0
               cf1_int(p) = 0.0D0
            else
               ! Calculate the proportional amounts to subtract
               cl_def = c_def_amount * (cl2(p) / total_c)
               if (dt1(7) .gt. 0.0D0) ca_def = c_def_amount * (ca2(p) / total_c)
               cf_def = c_def_amount * (cf2(p) / total_c)

               ! Subtract carbon from each pool ensuring no negative values
               cl1_int(p) = max(0.0D0, cl2(p) - cl_def)
               ca1_int(p) = max(0.0D0, ca2(p) - ca_def)
               cf1_int(p) = max(0.0D0, cf2(p) - cf_def)
            endif
         else
            if (dt1(7) .gt. 0.0D0) then
               cl1_int(p) = cl2(p)
               ca1_int(p) = ca2(p)
               cf1_int(p) = cf2(p)
            else
               cl1_int(p) = cl2(p)
               ca1_int(p) = 0.0D0
               cf1_int(p) = cf2(p)
            endif
         endif

         ! Ensure no negative values
         if (cl1_int(p) .lt. 0.0D0) cl1_int(p) = 0.0D0
         if (ca1_int(p) .lt. 0.0D0) ca1_int(p) = 0.0D0
         if (cf1_int(p) .lt. 0.0D0) cf1_int(p) = 0.0D0

      enddo ! end pls_loop (p)
      !$OMP END PARALLEL DO

      epavg = emax !mm/day

      ! FILL OUTPUT DATA
      evavg = 0.0D0
      rcavg = 0.0D0
      f5avg = 0.0D0
      laiavg = 0.0D0
      phavg = 0.0D0
      aravg = 0.0D0
      nppavg = 0.0D0
      rmavg = 0.0D0
      rgavg = 0.0D0
      wueavg = 0.0D0
      cueavg = 0.0D0
      litter_l_1 = 0.0D0
      cwd_1 = 0.0D0
      litter_fr_1 = 0.0D0
      c_defavg = 0.0D0
      vcmax_1 = 0.0D0
      specific_la_1 = 0.0D0
      lit_nut_content_1(:) = 0.0D0
      nupt_1(:) = 0.0D0
      pupt_1(:) = 0.0D0

      cleafavg_pft(:) = 0.0D0
      cawoodavg_pft(:) = 0.0D0
      cfrootavg_pft(:) = 0.0D0
      storage_out_bdgt_1(:, :) = 0.0D0
      limitation_status_1(:,:) = 0
      uptk_strat_1(:,:) = 0
      npp2pay_1(:) = 0.0
      cp(:) = 0.0D0
      c_cost_cwm = 0.0D0
      rnpp_out(:) = 0.0D0

      ! CALCULATE CWM FOR ECOSYSTEM PROCESSES
      ! clean NaN values in occupation coefficients
      ! These checks are useful for model start periods
      do p = 1, nlen
         if(ieee_is_nan(ocp_coeffs(p))) ocp_coeffs(p) = 0.0D0
      enddo

      evavg = sum(real(evap, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(evap))
      phavg = sum(real(ph, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(ph))
      aravg = sum(real(ar, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(ar))
      nppavg = sum(real(nppa, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(nppa))
      laiavg = sum(laia * ocp_coeffs, mask= .not. ieee_is_nan(laia))
      rcavg = sum(real(rc2, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(rc2))
      f5avg = sum(f5 * ocp_coeffs, mask= .not. ieee_is_nan(f5))
      rmavg = sum(real(rm, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(rm))
      rgavg = sum(real(rg, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(rg))
      wueavg = sum(real(wue, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(wue))
      cueavg = sum(real(cue, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(cue))
      c_defavg = sum(real(c_def, kind=r_8) * ocp_coeffs, mask= .not. ieee_is_nan(c_def)) / 2.73791
      vcmax_1 = sum(vcmax * ocp_coeffs, mask= .not. ieee_is_nan(vcmax))
      specific_la_1 = sum(specific_la * ocp_coeffs, mask= .not. ieee_is_nan(specific_la))
      litter_l_1 = sum(litter_l * ocp_coeffs, mask= .not. ieee_is_nan(litter_l))
      cwd_1 = sum(cwd * ocp_coeffs, mask= .not. ieee_is_nan(cwd))
      litter_fr_1 = sum(litter_fr * ocp_coeffs, mask= .not. ieee_is_nan(litter_fr))
      c_cost_cwm = sum(npp2pay * ocp_coeffs, mask= .not. ieee_is_nan(npp2pay))

      cp(1) = sum(cl1_int * ocp_coeffs, mask= .not. ieee_is_nan(cl1_int))
      cp(2) = sum(ca1_int * (ocp_coeffs * idx_grasses), mask= .not. ieee_is_nan(ca1_int))
      cp(3) = sum(cf1_int * ocp_coeffs, mask= .not. ieee_is_nan(cf1_int))
      cp(4) = sum(ar_fix_hr * (ocp_coeffs * idx_pdia), mask= .not. ieee_is_nan(ar_fix_hr))

      do p = 1,2
         nupt_1(p) = sum(nupt(p,:) * ocp_coeffs)
      enddo
      do p = 1,3
         pupt_1(p) = sum(pupt(p,:) * ocp_coeffs)
      enddo

      do p = 1, 6
         lit_nut_content_1(p) = sum(lit_nut_content(p, :) * ocp_coeffs)
      enddo

      ! Copy the results from the heap to the output variables
      do p = 1, nlen
         ri = lp(p)
         cleafavg_pft(ri)  = cl1_int(p)
         cawoodavg_pft(ri) = ca1_int(p)
         cfrootavg_pft(ri) = cf1_int(p)
         storage_out_bdgt_1(:,ri) = storage_out_bdgt(:,p)
         limitation_status_1(:,ri) = limitation_status(:,p)
         uptk_strat_1(:,ri) = uptk_strat(:,p)
         npp2pay_1(ri) = npp2pay(p)
         rnpp_out(ri) = rnpp_aux(p)
      enddo

      deallocate(rnpp_aux)
      deallocate(lp)
      deallocate(evap)
      deallocate(nppa)
      deallocate(ph)
      deallocate(ar)
      deallocate(laia)
      deallocate(f5)
      deallocate(vpd)
      deallocate(rc2)
      deallocate(rm)
      deallocate(rg)
      deallocate(wue)
      deallocate(cue)
      deallocate(c_def)
      deallocate(vcmax)
      deallocate(specific_la)
      deallocate(storage_out_bdgt)
      deallocate(tra)
      deallocate(nupt)
      deallocate(pupt)
      deallocate(litter_l)
      deallocate(cwd)
      deallocate(litter_fr)
      deallocate(lit_nut_content)
      deallocate(npp2pay)
      deallocate(limitation_status)
      deallocate(uptk_strat)
      deallocate(cl1_int)
      deallocate(cf1_int)
      deallocate(ca1_int)
      deallocate(cl2)
      deallocate(cf2)
      deallocate(ca2)
      deallocate(day_storage)
      deallocate(idx_grasses)
      deallocate(idx_pdia)
      deallocate(ar_fix_hr)
      deallocate(ocp_coeffs)
      deallocate(sto_budg)

   end subroutine daily_budget

end module budget
