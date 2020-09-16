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
!             João Paulo Darela Filho <darelafilho ( at ) gmail.com>


module budget
   implicit none
   private

   public :: daily_budget

contains

   subroutine daily_budget(dt, w1, g1, s1, ts, temp, prec, p0, ipar, rh&
        &, mineral_n, labile_p, catm, sto_budg, cl1_pft, ca1_pft, cf1_pft, dleaf, dwood&
        &, droot, w2, g2, s2, smavg, ruavg, evavg, epavg, phavg, aravg, nppavg&
        &, laiavg, rcavg, f5avg, rmavg, rgavg, cleafavg_pft, cawoodavg_pft&
        &, cfrootavg_pft, storage_out_bdgt, ocpavg, wueavg, cueavg, c_defavg&
        &, vcmax, specific_la, nupt, pupt, litter_l, cwd, litter_fr, lit_nut_content&
        &, delta_cveg, mineral_n_pls, labile_p_pls, limitation_status, sto_min)


      use types
      use global_par
      use alloc
      use productivity
      use omp_lib

      use photo, only: pft_area_frac, sto_resp
      use water, only: evpot2, penman, available_energy, runoff

      !     ----------------------------INPUTS-------------------------------
      real(r_8),dimension(ntraits,npls),intent(in) :: dt
      real(r_4),dimension(npls),intent(in) :: w1   !Initial (previous month last day) soil moisture storage (mm)
      real(r_4),dimension(npls),intent(in) :: g1   !Initial soil ice storage (mm)
      real(r_4),dimension(npls),intent(in) :: s1   !Initial overland snow storage (mm)
      real(r_4),intent(in) :: ts                   ! Soil temperature (oC)
      real(r_4),intent(in) :: temp                 ! Surface air temperature (oC)
      real(r_4),intent(in) :: prec                 ! Precipitation (mm/day)
      real(r_4),intent(in) :: p0                   ! Surface pressure (mb)
      real(r_4),intent(in) :: ipar                 ! Incident photosynthetic active radiation mol Photons m-2 s-1
      real(r_4),intent(in) :: rh                   ! Relative humidity
      real(r_4),intent(in) :: mineral_n
      real(r_4),intent(in) :: labile_p
      real(r_8),intent(in) :: catm


      real(r_8),dimension(3,npls),intent(in)  :: sto_budg ! Rapid Storage Pool (C,N,P)
      real(r_8),dimension(npls),intent(in) :: cl1_pft  ! initial BIOMASS cleaf compartment
      real(r_8),dimension(npls),intent(in) :: cf1_pft  !                 froot
      real(r_8),dimension(npls),intent(in) :: ca1_pft  !                 cawood
      real(r_8),dimension(npls),intent(in) :: dleaf  ! CHANGE IN cVEG (DAILY BASIS) TO GROWTH RESP
      real(r_8),dimension(npls),intent(in) :: droot
      real(r_8),dimension(npls),intent(in) :: dwood


      !     ----------------------------OUTPUTS------------------------------
      real(r_4) ,intent(out)                :: epavg          !Maximum evapotranspiration (mm/day)
      real(r_4),dimension(npls),intent(out) :: w2             !Final (last day) soil moisture storage (mm)
      real(r_4),dimension(npls),intent(out) :: g2             !Final soil ice storage (mm)
      real(r_4),dimension(npls),intent(out) :: s2             !Final overland snow storage (mm)
      real(r_8),dimension(npls),intent(out) :: smavg          !Snowmelt Daily average (mm/day)
      real(r_8),dimension(npls),intent(out) :: ruavg          !Runoff Daily average (mm/day)
      real(r_8),dimension(npls),intent(out) :: evavg          !Actual evapotranspiration Daily average (mm/day)
      real(r_8),dimension(npls),intent(out) :: phavg          !Daily photosynthesis (Kg m-2 y-1)
      real(r_8),dimension(npls),intent(out) :: aravg          !Daily autotrophic respiration (Kg m-2 y-1)
      real(r_8),dimension(npls),intent(out) :: nppavg         !Daily NPP (average between PFTs)(Kg m-2 y-1)
      real(r_8),dimension(npls),intent(out) :: laiavg         !Daily leaf area Index m2m-2
      real(r_8),dimension(npls),intent(out) :: rcavg          !Daily canopy resistence s/m
      real(r_8),dimension(npls),intent(out) :: f5avg          !Daily canopy resistence s/m
      real(r_8),dimension(npls),intent(out) :: rmavg,rgavg    !maintenance/growth respiration (Kg m-2 y-1)
      real(r_8),dimension(npls),intent(out) :: cleafavg_pft   !Carbon in plant tissues (kg m-2)
      real(r_8),dimension(npls),intent(out) :: cawoodavg_pft  !
      real(r_8),dimension(npls),intent(out) :: cfrootavg_pft  !
      real(r_8),dimension(npls),intent(out) :: ocpavg         ! [0-1]
      real(r_8),dimension(npls),intent(out) :: wueavg         !
      real(r_8),dimension(npls),intent(out) :: cueavg         ! [0-1]
      real(r_8),dimension(npls),intent(out) :: c_defavg       ! kg(C) m-2
      real(r_8),dimension(npls),intent(out) :: vcmax          ! µmol m-2 s-1
      real(r_8),dimension(npls),intent(out) :: specific_la    ! m2 g(C)-1
      real(r_8),dimension(npls),intent(out) :: nupt           ! g m-2
      real(r_8),dimension(npls),intent(out) :: pupt           ! g m-2
      real(r_8),dimension(npls),intent(out) :: litter_l       ! g m-2
      real(r_8),dimension(npls),intent(out) :: cwd            ! g m-2
      real(r_8),dimension(npls),intent(out) :: litter_fr      ! g m-2
      ! Lit_nut_content variables         [(lln),(rln),(cwdn),(llp),(rl),(cwdp)]
      real(r_8),dimension(6,npls),intent(out) :: lit_nut_content          ! g(Nutrient)m-2
      real(r_8),dimension(3,npls),intent(out) :: delta_cveg
      real(r_8),dimension(3,npls),intent(out) :: storage_out_bdgt
      real(r_8),dimension(npls),intent(out) ::  mineral_n_pls, labile_p_pls
      integer(i_2),dimension(3,npls),intent(out) :: limitation_status
      real(r_8), dimension(2, npls), intent(out) :: sto_min
      !     -----------------------Internal Variables------------------------
      integer(i_4) :: p, numprocs
      real(r_8),dimension(ntraits) :: dt1 ! Store pls attributes array (1D)
      !     RELATED WITH GRIDCELL OCUPATION

      real(r_8),dimension(npls) :: ocp_mm
      real(r_8),dimension(npls) :: ocp_coeffs
      logical(l_1),dimension(npls) :: ocp_wood, run

      !     WBM COMMUNICATION (water balance)
      real(r_4) :: psnow                !Snowfall (mm/day)
      real(r_4) :: prain                !Rainfall (mm/day)
      real(r_4) :: emax

      real(r_4),dimension(npls) :: rimelt               !Runoff due to soil ice melting
      real(r_4),dimension(npls) :: smelt                !Snowmelt (mm/day)
      real(r_4),dimension(npls) :: w                    !Daily soil moisture storage (mm)
      real(r_4),dimension(npls) :: g                    !Daily soil ice storage (mm)
      real(r_4),dimension(npls) :: s                    !Daily overland snow storage (mm)
      real(r_4),dimension(npls) :: ds
      real(r_4),dimension(npls) :: dw
      real(r_4),dimension(npls) :: roff                 !Total runoff
      real(r_4),dimension(npls) :: evap                !Actual evapotranspiration (mm/day)


      !c     Carbon Cycle
      real(r_4),dimension(npls) ::  ph             !Canopy gross photosynthesis (kgC/m2/yr)
      real(r_4),dimension(npls) ::  ar             !Autotrophic respiration (kgC/m2/yr)
      real(r_4),dimension(npls) ::  nppa           !Net primary productivity / auxiliar
      real(r_8),dimension(npls) ::  laia           !Leaf area index (m2 leaf/m2 area)
      real(r_4),dimension(npls) ::  rc2            !Canopy resistence (s/m)
      real(r_4),dimension(npls) ::  f1             !
      real(r_8),dimension(npls) ::  f5             !Photosynthesis (mol/m2/s)
      real(r_4),dimension(npls) ::  vpd            !Vapor Pressure deficit
      real(r_4),dimension(npls) ::  rm             !maintenance & growth a.resp
      real(r_4),dimension(npls) ::  rg
      real(r_4),dimension(npls) ::  wue, cue, c_def
      real(r_8),dimension(npls) ::  cl1_int, cf1_int, ca1_int
      real(r_8),dimension(npls) ::  cl2,cf2,ca2 ! carbon pos-allocation
      real(r_8),dimension(3,npls) :: day_storage   ! g m-2
      real(r_8) :: carbon_in_storage
      real(r_8) :: testcdef
      real(r_8) :: sr, mr_sto, growth_stoc
      ! real(r_8) :: srn
      ! real(r_8) :: srp
      ! real(r_8) :: mrn
      ! real(r_8) :: mrp
      ! real(r_8) :: ston2c
      ! real(r_8) :: stop2c


      !     Precipitation
      !     =============
      psnow = 0.0
      prain = 0.0
      if (temp.lt.tsnow) then
         psnow = prec
      else
         prain = prec
      endif

      !     Initialization
      !     --------------
      epavg   = 0.0
      w       = w1     ! hidrological pools state vars
      g       = g1
      s       = s1
      smavg   = 0.0D0    !  plss vectors (outputs)
      w2(:) = 0.0
      g2(:) = 0.0
      s2(:) = 0.0
      ruavg(:) = 0.0D0
      evavg(:) = 0.0D0
      rcavg(:) = 0.0D0
      laiavg(:) = 0.0D0
      phavg(:) = 0.0D0
      aravg(:) = 0.0D0
      nppavg(:) = 0.0D0
      rmavg(:) = 0.0D0
      rgavg(:) = 0.0D0
      ocpavg(:) = 0.0D0
      wueavg(:) = 0.0D0
      cueavg(:) = 0.0D0
      ocp_mm(:) = 0.0D0
      emax  = 0.0D0
      cleafavg_pft(:) = 0.0D0
      cawoodavg_pft(:) = 0.0D0
      cfrootavg_pft(:) = 0.0D0
      ocpavg(:) = 0.0D0
      wueavg(:) = 0.0D0
      cueavg(:) = 0.0D0
      c_defavg(:) = 0.0D0
      vcmax(:) = 0.0D0
      specific_la(:) = 0.0D0
      nupt(:) = 0.0D0
      pupt(:) = 0.0D0
      litter_l(:) = 0.0D0
      cwd(:) = 0.0D0
      litter_fr(:) = 0.0D0
      lit_nut_content(:, :) = 0.0D0
      delta_cveg(:, :) = 0.0D0
      storage_out_bdgt(:, :) = 0.0D0

      nppa(:) = 0.0
      ph(:) = 0.0
      ar(:) = 0.0
      laia(:) = 0.0D0
      f5(:) = 0.0D0
      f1(:) = 0.0
      vpd(:) = 0.0
      rc2(:) = 0.0
      rm(:) = 0.0
      rg(:) = 0.0
      wue(:) = 0.0
      cue(:) = 0.0
      rc2(:) = 0.0


      !     Grid cell area fraction (%) ocp_coeffs(pft(1), pft(2), ...,pft(p))
      !     =================================================================
      ! (cleaf1, cfroot1, cawood1, awood, ocp_coeffs, ocp_wood, run_pls, c_to_soil)
      call pft_area_frac(cl1_pft, cf1_pft, ca1_pft, dt(7, :), ocp_coeffs, ocp_wood, run, ocp_mm) ! def in funcs.f90
      ocpavg = ocp_coeffs
      !     Maximum evapotranspiration   (emax)
      !     =================================
      emax = evpot2(p0,temp,rh,available_energy(temp))

      !     Productivity & Growth (ph, ALLOCATION, aresp, vpd, rc2 & etc.) for each PLS
      !     =================================
      numprocs = 2
      ! do  p = 1, npls
      !    if (run(p)) numprocs = numprocs + 1
      ! enddo
      ! numprocs = max(2, numprocs / 4)
      ! print*, numprocs
      call OMP_SET_NUM_THREADS(numprocs)

      !$OMP PARALLEL DO &
      !$OMP SCHEDULE(AUTO) &
      !$OMP DEFAULT(SHARED) &
      !$OMP PRIVATE(p, carbon_in_storage, testcdef, sr, dt1, mr_sto, growth_stoc)
      do p = 1,npls
         if (.not. run(p)) then
            cleafavg_pft(p)  = 0.0D0
            cawoodavg_pft(p) = 0.0D0
            cfrootavg_pft(p) = 0.0D0
         else
            carbon_in_storage = 0.0D0
            testcdef = 0.0D0
            sr = 0.0D0

            dt1 = dt(:,p) ! Pick up the pls functional attributes list

            call prod(dt1,ocp_wood(p),catm, temp,ts,p0,w(p),ipar,rh,emax,cl1_pft(p)&
               &,ca1_pft(p),cf1_pft(p),dleaf(p),dwood(p),droot(p)&
               &,ph(p),ar(p),nppa(p),laia(p)&
               &,f5(p),vpd(p),rm(p),rg(p),rc2(p),wue(p),c_def(p)&
               &,vcmax(p),specific_la(p))


         ! Check if the carbon deficit can be conpensated by stored carbon
            carbon_in_storage = sto_budg(1, p)
            storage_out_bdgt(1, p) = carbon_in_storage
            if (c_def(p) .gt. 0.0) then
               testcdef = c_def(p) - carbon_in_storage
               if(testcdef .lt. 0.0) then
                  storage_out_bdgt(1, p) = carbon_in_storage + testcdef ! Testcdef in negative
               else
                  storage_out_bdgt(1, p) = 0.0D0
                  c_def(p) = real(testcdef, kind=r_4)       ! testcdef is zero or positive
               endif
            endif
            carbon_in_storage = 0.0D0
            testcdef = 0.0D0

            ! calculate maintanance respirarion of stored C
            ! ston2c = 0.0D0
            ! stop2c = 0.0D0
            ! if (storage_out_bdgt(1,p) .gt. 0.0D0) then
            !    ston2c = storage_out_bdgt(2,p)/storage_out_bdgt(1,p)
            !    stop2c = storage_out_bdgt(3,p)/storage_out_bdgt(1,p)
            ! endif
            mr_sto = sto_resp(temp, storage_out_bdgt)
            !mrn = ston2c * mr_sto
            !mrp = stop2c * mr_sto

            storage_out_bdgt(1,p) = max(0.0D0, (storage_out_bdgt(1,p) - mr_sto))
            !storage_out_bdgt(2,p) = max(0.0D0, (storage_out_bdgt(2,p) - mrn))
            !storage_out_bdgt(3,p) = max(0.0D0, (storage_out_bdgt(3,p) - mrp))

            ! the root potential to extract nutrients from soil is
            ! calculated here

            ! f(fine_root_mass, fine_root_residence_time)

            ! Specific root area (fine roots)
            ! Specific root length

            ! upt_capacity <- root specific uptake capacity g (P or N) m-2(root) day-1

            ! A function like F5


            !     Carbon/Nitrogen/Phosphorus allocation/deallocation
            !     =====================================================

            call allocation (dt1,nppa(p),mineral_n,labile_p,cl1_pft(p),ca1_pft(p)&
               &, cf1_pft(p),storage_out_bdgt(:,p),day_storage(:,p),cl2(p),ca2(p)&
               &, cf2(p),litter_l(p),cwd(p), litter_fr(p),nupt(p),pupt(p)&
               &, lit_nut_content(:,p), limitation_status(:,p))

            ! Estimate growth of storage C pool
            growth_stoc = max( 0.0D0, (day_storage(1,p) - storage_out_bdgt(1,p)))

            ! ston2c = 0.0D0
            ! stop2c = 0.0D0
            ! if (day_storage(1,p) .gt. 0.0D0) then
            !    ston2c = day_storage(2,p)/day_storage(1,p)
            !    stop2c = day_storage(3,p)/day_storage(1,p)
            ! endif

            storage_out_bdgt(:,p) = day_storage(:,p)

            ! SAVE OUTPUT
            mineral_n_pls(p) = mineral_n - nupt(p)
            labile_p_pls(p) = labile_p - pupt(p)

            ! Calculate storage GROWTH  respiration
            sr = 0.75D0 * growth_stoc ! g m-2
            !srn = sr * ston2c
            !srp = sr * stop2c

            ar(p) = ar(p) + real(((sr + mr_sto) * 0.365242), kind=r_4) ! Convert g m-2 day-1 in kg m-2 year-1

            storage_out_bdgt(1, p) = storage_out_bdgt(1, p) - sr
            !storage_out_bdgt(2, p) = storage_out_bdgt(2, p) - srn
            !storage_out_bdgt(3, p) = storage_out_bdgt(3, p) - srp

            sto_min(1, p) = 0.0D0!srn + mrn
            sto_min(2, p) = 0.0D0!srp + mrp

            growth_stoc = 0.0D0
            mr_sto = 0.0D0
            sr = 0.0D0
            ! srn = 0.0D0
            ! srp = 0.0D0
            ! mrn = 0.0D0
            ! mrp = 0.0D0


            if(ph(p) .eq. 0.0 .or. nppa(p) .eq. 0.0) then
               cue(p) = 0.0
            else
               cue(p) = nppa(p)/ph(p)
            endif

            delta_cveg(1,p) = cl2(p) - cl1_pft(p)  !kg m-2

            if(dt1(4) .le. 0) then
               delta_cveg(2,p) = 0.0D0
            else
               delta_cveg(2,p) = ca2(p) - ca1_pft(p)
            endif

            delta_cveg(3,p) = cf2(p) - cf1_pft(p)

            !     Snow budget
            !     ===========
            smelt(p) = 2.63 + 2.55*temp + 0.0912*temp*prain !Snowmelt (mm/day)
            smelt(p) = amax1(smelt(p),0.)
            smelt(p) = amin1(smelt(p),s(p)+psnow)
            ds(p) = psnow - smelt(p)
            s(p) = s(p) + ds(p)

            !     Water budget
            !     ============
            if (ts.le.tice) then !Frozen soil
               g(p) = g(p) + w(p) !Soil moisture freezes
               w(p) = 0.0
               roff(p) = smelt(p) + prain !mm/day
               evap(p) = 0.0

            else                !Non-frozen soil
               w(p) = w(p) + g(p)
               g(p) = 0.0
               rimelt(p) = 0.0
               if (w(p).gt.wmax) then
                  rimelt(p) = w(p) - wmax !Runoff due to soil ice melting
                  w(p) = wmax
               endif

               roff(p) = runoff(w(p)/wmax)       !Soil moisture runoff (roff, mm/day)

               evap(p) = penman(p0,temp,rh,available_energy(temp),rc2(p)) !Actual evapotranspiration (evap, mm/day)
               dw(p) = prain + smelt(p) - evap(p) - roff(p)
               w(p) = w(p) + dw(p)
               if (w(p).gt.wmax) then
                  roff(p) = roff(p) + (w(p) - wmax)
                  w(p) = wmax
               endif
               if (w(p).lt.0.) w(p) = 0.
               roff(p) = roff(p) + rimelt(p) !Total runoff
            endif


            w2(p) = w(p)
            g2(p) = g(p)
            s2(p) = s(p)
            smavg(p) = smelt(p)
            ruavg(p) = roff(p)     ! mm day-1
            evavg(p) = evap(p)     ! mm day-1
            phavg(p) = ph(p)       !kgC/m2/day
            aravg(p) = ar(p)       !kgC/m2/year
            nppavg(p) = nppa(p)    !kgC/m2/day
            laiavg(p) = laia(p)
            rcavg(p) = rc2(p)      ! s m -1
            f5avg(p) = f5(p)
            rmavg(p) = rm(p)
            rgavg(p) = rg(p)

            wueavg(p) = wue(p)
            cueavg(p) = cue(p)
            c_defavg(p) = c_def(p) / 2.73791


            if(c_def(p) .gt. 0.0) then
               if(dt1(7) .gt. 0.0) then
                  cl1_int(p) = cl2(p) - ((c_def(p) * 1e-3) * 0.333333333)
                  ca1_int(p) = ca2(p) - ((c_def(p) * 1e-3) * 0.333333333)
                  cf1_int(p) = cf2(p) - ((c_def(p) * 1e-3) * 0.333333333)
               else
                  cl1_int(p) = cl2(p) - ((c_def(p) * 1e-3) * 0.5)
                  ca1_int(p) = 0.0
                  cf1_int(p) = cf2(p) - ((c_def(p) * 1e-3) * 0.5)
               endif
            else
               if(dt1(7) .gt. 0.0) then
                  cl1_int(p) = cl2(p)
                  ca1_int(p) = ca2(p)
                  cf1_int(p) = cf2(p)
               else
                  cl1_int(p) = cl2(p)
                  ca1_int(p) = 0.0
                  cf1_int(p) = cf2(p)
               endif
            endif

            cleafavg_pft(p)  = cl1_int(p)
            cawoodavg_pft(p) = ca1_int(p)
            cfrootavg_pft(p) = cf1_int(p)
         endif

      enddo ! end pls_loop (p)
      !$OMP END PARALLEL DO
      epavg = emax !mm/day

   end subroutine daily_budget

end module budget
