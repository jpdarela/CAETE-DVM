! Copyright 2017- LabTerra

!     This program is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.

!     This program is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR 2PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with this program.  If not, see <http://www.gnu.org/licenses/>.

! AUTHORS: *, JP Darela, Bianca Rius, Helena do Prado, David Lapola
! *This program is based on the work of those that gave us the INPE-CPTEC-PVM2 model

module photo

   ! Module defining functions related with CO2 assimilation and other processes in CAETE
   ! Some of these functions are based in CPTEC-PVM2, others are new features
   use types
   implicit none
   private

   ! functions(f) and subroutines(s) defined here
   public ::                    &
        gross_ph                    ,&  ! (f), gross photosynthesis (kgC m-2 y-1)
        leaf_area_index             ,&  ! (f), leaf area index(m2 m-2)
        f_four                      ,&  ! (f), auxiliar function (calculates f4sun or f4shade or sunlai)
        spec_leaf_area              ,&  ! (f), specific leaf area (m2 g-1)
        sla_reich                   ,&  ! (f), sla based on Reich et al. 1997 (cm2 g-1)
        leaf_nitrogen_concentration  ,& ! (f), leaf nitrogen concentration (gN gC-1)
        water_stress_modifier       ,&  ! (f), F5 - water stress modifier (dimensionless)
        photosynthesis_rate         ,&  ! (s), leaf level CO2 assimilation rate (molCO2 m-2 s-1)
        vcmax_a                     ,&  ! (f), VCmax from domingues et al. 2010 (eq.1)
        vcmax_a1                    ,&  ! (f), VCmax from domingues et al. 2010 (eq.2)
        vcmax_b                     ,&  ! (f), VCmax from domingues et al. 2010 (eq.1 Table SM)
        stomatal_resistance         ,&  ! (f), Canopy resistence (from Medlyn et al. 2011a) (s/m)
        stomatal_conductance        ,&  ! (f), IN DEVELOPMENT - return stomatal conductance
        vapor_p_deficit             ,&  ! (f), Vapor pressure defcit  (kPa)
        transpiration               ,&
        tetens                      ,&  ! (f), Maximum vapor pressure (hPa)
        m_resp                      ,&  ! (f), maintenance respiration (plants)
        sto_resp                    ,&
        realized_npp                ,&
        spinup2                     ,&  ! (s), SPINUP function for CVEG pools
        spinup3                     ,&  ! (s), SPINUP function to check the viability of Allocation/residence time combinations
        g_resp                      ,&  ! (f), growth Respiration (kg m-2 yr-1)
        pft_area_frac               ,&  ! (s), area fraction by biomass
        water_ue                    ,&
        leap                        ,&
        vec_ranging                 ,&
        resp_aux                    ,& ! respiration auxiliary functions
        f

contains
   !=================================================================
   !=================================================================
   !> leap
   !> Returns true if the year is a leap year
   !> @param year Year to be checked
   !> @return True if the year is a leap year, false otherwise
   !=================================================================
   function leap(year) result(is_leap)

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
   !> gross_ph
   !> Returns gross photosynthesis rate (kgC m-2 y-1) (GPP)
   !> @param f1 Photosynthesis rate (molCO2 m-2 s-1)
   !> @param cleaf Leaf carbon in kgC m-2
   !> @param sla Specific leaf area in m2 gC-1
   !> @return Gross photosynthesis rate in kgC m-2 y-1
   !=================================================================
   function gross_ph(f1,cleaf,sla) result(ph)
      ! Returns gross photosynthesis rate (kgC m-2 y-1) (GPP)
      !implicit none

      real(r_8),intent(in) :: f1    !molCO2 m-2 s-1
      real(r_8),intent(in) :: cleaf !kgC m-2
      real(r_8),intent(in) :: sla   !m2 gC-1
      real(r_8) :: ph

      real(r_8) :: f4sun, f1in
      real(r_8) :: f4shade

      f1in = f1
      f4sun = f_four(1,cleaf,sla)
      f4shade = f_four(2,cleaf,sla)

      ph = real((0.012D0*31557600.0D0*f1in*f4sun*f4shade), r_8)
      if(ph .lt. 0.0) ph = 0.0
   end function gross_ph

   !=================================================================
   !=================================================================
   !> leaf_area_index
   !> Returns Leaf Area Index (m2 m-2) based on leaf carbon and specific leaf area
   !> @param cleaf Leaf carbon in kgC m-2
   !> @param sla Specific leaf area in m2 gC-1
   !> @return Leaf Area Index in m2 m-2
   !=================================================================
   function leaf_area_index(cleaf, sla) result(lai)
      use photo_par, only: gap_fraction
      ! Returns Leaf Area Index m2 m-2

      !implicit none

      real(r_8),intent(in) :: cleaf !kgC m-2
      real(r_8),intent(in) :: sla   !m2 gC-1
      real(r_8) :: lai              !m2 m-2

      real(r_8), parameter :: c_petiole = 0.30D0 ! Petiole carbon fraction of total leaf carbon

      lai  = ((cleaf * (1.0D0 - c_petiole)) * 1.0D3 * sla) * (1.0D0 - gap_fraction)
      if(lai .lt. 0.0D0) lai = 0.0D0

   end function leaf_area_index

   !=================================================================
   !=================================================================
   !> spec_leaf_area
   !> Specific leaf area based on Reich et al. 1997
   !> @param tau_leaf Leaf turnover time in years
   !> @return Specific leaf area in m2 gC-1
   !=================================================================
   function spec_leaf_area(tau_leaf) result(sla)
      !implicit none

      real(r_8),intent(in) :: tau_leaf  !years
      real(r_8):: sla   !m2 gC-1 Dry mass
      real(r_8), parameter :: fc = 0.47D0  ! carbon fraction of dry mass

      sla = sla_reich(tau_leaf) / fc * 0.0001 ! Projected leaf area per carbon mass. 1e-4 convert units from cm2 gC-1 to m2 gC-1 and fc converts from dry mass to carbon mass
      ! sla = sla_reich(tau_leaf) * 0.0001 ! Withouth fc # Projected leaf area per dry mass (m2 g-1)
   end function spec_leaf_area

   !=================================================================
   !=================================================================
   !> sla_reich
   !> Specific leaf area based on Reich et al. 1997
   !> @param tau_leaf Leaf turnover time in years
   !> @return Specific leaf area in cm2 gC-1
   !=================================================================
   function sla_reich(tau_leaf) result(sla)
      ! based on Reich et al. 1997
      !implicit none

      real(r_8),intent(in) :: tau_leaf  !years
      real(r_8):: sla   !cm2 gC-1

      real(r_8) :: tl0

      tl0 = tau_leaf * 12.0D0

      sla = 278.0D0 * (tl0 ** (-0.49)) ! Intercept 278 extracted from the figure in Reich et al. 1997
      !sla = 96.68 * (tl0 ** (-0.49))
      ! sla = 85.0D0 * (tl0 ** (-0.49)) ! cm2 gC-1 Dry mass


   end function sla_reich

   !=================================================================
   !=================================================================
   !> leaf_nitrogen_concetration +++Not used in the code+++
   !> Leaf nitrogen concentration based on Reich et al. 1997
   !> @param tau_leaf Leaf turnover time in years
   !> @return Leaf nitrogen concentration in gN gC-1
   !=================================================================
   function leaf_nitrogen_concentration(tau_leaf) result(nleaf)
      ! based on Reich et al. 1997
      !implicit none
      real(r_8),intent(in) :: tau_leaf  !years
      real(r_8):: nleaf   !gN gC-1
      real(r_8) :: tl0
      real(r_8), parameter :: fc = 0.47D0  ! carbon fraction of dry mass
      real(r_8) :: delta, r
      tl0 = tau_leaf * 12.0D0
      nleaf = 42.7D0 * (tl0 ** (-0.32)) / fc  ! mgN gC-1
      nleaf = nleaf * 0.001D0 ! convert mgN gC-1 to gN gC-1
      call random_number(r)
      delta = (r - 0.5D0) * 0.01D0
      nleaf = nleaf + delta
   end function leaf_nitrogen_concentration

   !=================================================================
   !=================================================================
   !> f_four
   !> Function used to scale LAI from leaf to canopy level (2 layers)
   !> @param fs Function mode:
   !> 1  == f4sun   --->  to gross assimilation
   !> 2  == f4shade --->  too
   !> 90 == sun LAI
   !> 20 == shade LAI
   !> Any other number returns sunlai (not scaled to canopy)
   !> @param cleaf Carbon in leaf (kg m-2)
   !> @param sla Specific leaf area (m2 gC-1)
   !> @return lai_ss Leaf area index (m2 m-2)
   !> @note This function is based on de Pury & Farquhar (1997) adapted from the CPTEC-PVM2 model
   !=================================================================
   function f_four(fs,cleaf,sla) result(lai_ss)
      ! Function used to scale LAI from leaf to canopy level (2 layers)
      use photo_par, only: p26, p27
      !implicit none

      integer(i_4),intent(in) :: fs !function mode:
      ! 1  == f4sun   --->  to gross assimilation
      ! 2  == f4shade --->  too
      ! 90 == sun LAI
      ! 20 == shade LAI
      ! Any other number returns sunlai (not scaled to canopy)

      real(r_8),intent(in) :: cleaf ! carbon in leaf (kg m-2)
      real(r_8),intent(in) :: sla   ! specific leaf area (m2 gC-1)
      real(r_8) :: lai_ss           ! leaf area index (m2 m-2)

      real(r_8) :: lai
      real(r_8) :: sunlai
      real(r_8) :: shadelai

      lai = leaf_area_index(cleaf, sla)

      sunlai = (1.0D0-(dexp(-p26*lai)))/p26
      shadelai = lai - sunlai

      lai_ss = sunlai

      if (fs .eq. 90) then
         return
      endif
      if (fs .eq. 20) then
         lai_ss = shadelai
         return
      endif

      !Scaling-up to canopy level (dimensionless scaling factors)
      !------------------------------------------
      !Sun/Shade approach to canopy scaling !Based in de Pury & Farquhar (1997)
      !------------------------------------------------------------------------
      if(fs .eq. 1) then
         ! f4sun
         lai_ss = (1.0-(dexp(-p26*sunlai)))/p26 !sun decl 90 degrees
         return
      endif

      if(fs .eq. 2) then
         !f4shade
         lai_ss = (1.0-(dexp(-p27*shadelai)))/p27 !sun decl ~20 degrees
         return
      endif
   end function f_four

   !=================================================================
   !=================================================================
   !> water_stress_modifier
   !> Returns the water stress modifier (F5) based on soil water content, carbon in fine roots, canopy resistance, potential evapotranspiration and maximum soil water content
   !> @param w Soil water content in mm
   !> @param cfroot Carbon in fine roots in kg m-2
   !> @param rc Canopy resistance in s/m
   !> @param ep Potential evapotranspiration in mm s-1
   !> @param wmax Maximum soil water content in mm
   !> @return f5 Water stress modifier (dimensionless)
   !=================================================================
   function water_stress_modifier(w, cfroot, rc, ep, wmax) result(f5)
      use global_par, only: csru, alfm, gm, rcmin, rcmax
      !implicit none

      real(r_8),intent(in) :: w      !soil water mm
      real(r_8),intent(in) :: cfroot !carbon in fine roots kg m-2
      real(r_8),intent(in) :: rc     !Canopy resistence s/m)
      real(r_8),intent(in) :: ep
      real(r_8),intent(in) :: wmax     !potential evapotranspiration
      real(r_8) :: f5


      real(r_8) :: pt, rc_aux, rcmin_aux, ep_aux
      real(r_8) :: gc
      real(r_8) :: wa
      real(r_8) :: d
      real(r_8) :: f5_64

      ! print*, 'water: ', w

      wa = w/wmax
      rc_aux = real(rc, kind=r_8)
      rcmin_aux = real(rcmin, kind=r_8)
      ep_aux = real(ep, kind=r_8)
      ! if (rc .gt. rcmax) rc_aux = real(rcmax, r_8)
      ! CRSU: Specific root water uptake mm/g/day (g(C) of fine roots) = 0.5
      pt = csru*(cfroot * 1000) * wa  !(based in Pavlick et al. 2013; *1000. converts kgC/m2 to gC/m2)
      gc = max((1.0D0/rc_aux * 1000.0D0), gm) ! Canopy conductance (mm s-1)-> s m-1 to mm s-1

      !d =(ep * alfm) / (1. + gm/gc) !(based in Gerten et al. 2004)
      d = (ep_aux * alfm) / (1.0D0 + (gm/gc))
      if(d .gt. 0.0D0) then
         f5_64 = pt/d
         f5_64 = exp(-f5_64)
         f5_64 = 1.0D0 - f5_64
      else
         f5_64 = wa
      endif

      f5 = f5_64
      if (f5 .lt. wa) f5 = wa
      if (f5 .gt. 0.999D0) f5 = 1.0D0
   end function water_stress_modifier

   ! =============================================================
   ! =============================================================
   !> stomatal_resistance
   !> Returns stomatal resistance based on Medlyn et al. 2011a
   !> @param vpd_in Vapor pressure deficit in hPa
   !> @param f1_in Photosynthesis rate in molCO2 m-2 s-1
   !> @param g1 Model m (slope) in sqrt(kPa)
   !> @param ca Atmospheric CO2 concentration in ppm
   !> @return rc2_in Canopy resistance in s m-1
   !=================================================================
   function stomatal_resistance(vpd_in,f1_in,g1,ca) result(rc2_in)
      ! return stomatal resistence based on Medlyn et al. 2011a
      ! Coded by Helena Alves do Prado
      use global_par, only: rcmin, rcmax


      !implicit none

      real(r_8),intent(in) :: f1_in    !Photosynthesis (molCO2/m2/s)
      real(r_8),intent(in) :: vpd_in   !hPa
      real(r_8),intent(in) :: g1       ! model m (slope) (sqrt(kPa))
      real(r_8),intent(in) :: ca
      real(r_8) :: rc2_in              !Canopy resistence (sm-1)

      !     Internal
      !     --------
      real(r_8) :: gs       !Canopy conductance (molCO2 m-2 s-1)
      real(r_8) :: D1       !sqrt(kPA)
      real(r_8) :: vapour_p_d
      ! real(r_8):: GMIN = 2.0D-6


      vapour_p_d = vpd_in
      ! Assertions
      if(vpd_in .le. 0.01) vapour_p_d = 0.05
      if(vpd_in .gt. 8.0) vapour_p_d = 8.0

      D1 = sqrt(vapour_p_d)
      gs = 100.0D0 + 1.6D0 * (1.0D0 + (g1/D1)) * ((f1_in * 1.0D6)/ca) ! micromol m-2 s-1

      gs = gs * 1.0D-6 * 0.02520D0 ! convrt from  micromol/m²/s to m s-1
      rc2_in = real(1.0D0 / gs, r_8)  !  s m-1

      if(rc2_in .ge. rcmax) rc2_in = rcmax
      if(rc2_in .lt. rcmin) rc2_in = rcmin

   end function stomatal_resistance

   !=================================================================
   !=================================================================
   !> stomatal_conductance
   !> Returns stomatal conductance based on Medlyn et al. 2011
   !> @param vpd_in Vapor pressure deficit in hPa
   !> @param f1_in Photosynthesis rate in molCO2 m-2 s-1
   !> @param g1 Model m (slope) in sqrt(kPa)
   !> @param ca Atmospheric CO2 concentration in ppm
   !> @return gs Canopy conductance in molCO2 m-2 s-1
   !=================================================================
   !> IN DEVELOPMENT - return stomatal conductance
   function stomatal_conductance(vpd_in,f1_in,g1,ca) result(gs)
    ! return stomatal conductance based on Medlyn et al. 2011
    ! Coded by Helena Alves do Prado


    !implicit none

    real(r_8),intent(in) :: f1_in    !Photosynthesis (molCO2/m2/s)
    real(r_8),intent(in) :: vpd_in   !hPa
    real(r_8),intent(in) :: g1       ! model m (slope) (sqrt(kPa))
    real(r_8),intent(in) :: ca
    real(r_8) :: gs       !Canopy conductance (molCO2 m-2 s-1)
    !     Internal
    !     --------
    real(r_8) :: D1       !sqrt(kPA)
    real(r_8) :: vapour_p_d

    vapour_p_d = vpd_in
    ! Assertions
    if(vpd_in .le. 0.0) vapour_p_d = 0.01
    if(vpd_in .gt. 4.0) vapour_p_d = 4.0
    ! print *, 'vpd going mad in canopy_resistence'
    ! stop
    ! endif

    D1 = sqrt(vapour_p_d)
    gs = 1.6 * (1.0 + (g1/D1)) * (f1_in/ca) !mol m-2 s-1
 end function stomatal_conductance

   !=================================================================
   !=================================================================
   !> water_ue
   !> Returns water use efficiency (WUE) based on assimilation, stomatal resistance, atmospheric pressure and vapor pressure deficit
   !> @param a Assimilation rate in mol m-2 s-1
   !> @param g Stomatal resistance in s m-1
   !> @param p0 Atmospheric pressure in hPa
   !> @param vpd Vapor pressure deficit in kPa
   !> @return wue Water use efficiency in mol CO2 mol H2O-1
   !=================================================================
   function water_ue(a, g, p0, vpd) result(wue)
      real(r_8),intent(in) :: a
      real(r_8),intent(in) :: g, p0, vpd
      ! a = assimilacao; g = resistencia; p0 = pressao atm; vpd = vpd
      real(r_8) :: wue

      real(r_8) :: g_in, p0_in, e_in

      g_in = (1./g) * 40.87 ! convertendo a resistencia (s m-1) em condutancia mol m-2 s-1
      p0_in = p0 /10. ! convertendo pressao atm (mbar/hPa) em kPa
      e_in = g_in * (vpd/p0_in) ! calculando transpiracao mol H20 m-2 s-1

      if(a .eq. 0 .or. e_in .eq. 0) then
         wue = 0
      else
         wue = real(a, kind=r_8)/e_in
      endif
   end function water_ue


   !=================================================================
   !=================================================================
   !> transpiration
   !> Returns transpiration rate based on stomatal resistance, atmospheric pressure and vapor pressure deficit
   !> @param g Stomatal resistance in s m-1
   !> @param p0 Atmospheric pressure in hPa
   !> @param vpd Vapor pressure deficit in kPa
   !> @param unit Unit of measurement: 1 for mol m-2 s-1, 2 for mm s-1
   !> @return e Transpiration rate in mol m-2 s-1 or mm s-1
   !=================================================================
   function transpiration(g, p0, vpd, unit) result(e)
      !implicit none
      real(r_8),intent(in) :: g, p0, vpd
      integer(i_4), intent(in) :: unit
      ! g = resistencia estomatica s m-1; p0 = pressao atm (mbar == hPa); vpd = vpd (kPa)
      real(r_8) :: e

      real(r_8) :: g_in, p0_in, e_in

      g_in = (1./g) * 44.6 ! convertendo a resistencia (s m-1) (m s-1) em condutancia mol m-2 s-1
      p0_in = p0 / 10. ! convertendo pressao atm (mbar/hPa) em kPa

      e_in = g_in * (vpd/p0_in) ! calculando transpiracao mol H20 m-2 s-1

      if(unit .eq. 1) then
         e = e_in
         return
      else
         e = 18.0 * e_in * 1e-3    ! g m-2 s-1 * 1d-3  == Kg m-2 s-1  == mm s-1
      endif
   end function transpiration


   !=================================================================
   !=================================================================
   !> vapor_p_deficit
   !> Returns vapor pressure deficit (VPD) based on temperature and relative humidity
   !> @param t Temperature in °C
   !> @param rh Relative humidity in percentage (0-100)
   !> @return vpd_0 Vapor pressure deficit in kPa
   !=================================================================
   function vapor_p_deficit(t,rh) result(vpd_0)
      real(r_8),intent(in) :: t
      real(r_8),intent(in) :: rh

      real(r_8) :: vpd_ac
      real(r_8) :: es
      real(r_8) :: vpd_0

      ! ext func
      !real(r_8) :: tetens

      es = tetens(t)

      !     delta_e = es*(1. - ur)
      !VPD-REAL = Actual vapor pressure
      vpd_ac = es * rh       ! RESULT in hPa == mbar! we want kPa (DIVIDE by 10.)
      !Vapor Pressure Deficit
      vpd_0 = (es - vpd_ac) / 10.
   end function vapor_p_deficit

   !=================================================================
   !=================================================================
   !> realized_npp
   !> Calculates the realized NPP based on potential NPP, nutrient uptake potential and available nutrients
   !> @param pot_npp_pool Potential NPP for the pool (leaf, root or wood)
   !> @param nupt_pot Potential uptake of nutrient (N/P) for each pool
   !> @param available_n Available nutrients for growth weighted for each pool
   !> @param rnpp Realized NPP (output)
   !> @param nl Is limited? (output)
   !> @note If available nutrients are greater than or equal to potential nutrient uptake, there is no limitation
   !=================================================================
   ! subroutine realized_npp(pot_npp_pool, nupt_pot, available_n,&
   !    &  rnpp, nl)

   !    real(r_8), intent(in) :: pot_npp_pool ! POTENTIAL NPP (POOL - leaf, root or wood)
   !    real(r_8), intent(in) :: nupt_pot     ! POTENTIAL UPTAKE OF NUTRIENT(N/P)for each pool
   !    real(r_8), intent(in) :: available_n  ! AVAILABLE NUTRIENTS FOR GROWTH weighted for each pool

   !    real(r_8), intent(out) :: rnpp        ! REALIZED NPP
   !    logical(l_1), intent(out) :: nl       ! IS LIMITED?

   !    ! NUTRIENT LIMITED NPP TO(CVEGpool):
   !    if (available_n .ge. nupt_pot) then
   !       ! THere is NO LIMITATION in this case
   !       nl = .false.
   !       ! GROWTH IS ACCOMPLISHED (all npp can go to the CVEG pool)
   !       rnpp = pot_npp_pool
   !    else
   !       ! NPP OF THIS POOL IS LIMITED BY Nutrient X
   !       ! In this case the realized NPP for the pool is smaller than the Potential POOL
   !       nl = .true.
   !       ! ACOMPLISHED NPP
   !       rnpp = max( 0.0D0, (available_n * pot_npp_pool) / nupt_pot)
   !    endif

   !    end subroutine realized_npp

   ! subroutine realized_npp(pot_npp_pool, nupt_pot, available_n, rnpp, nl)
   !    real(r_8), intent(in) :: pot_npp_pool ! POTENTIAL NPP (POOL - leaf, root or wood)
   !    real(r_8), intent(in) :: nupt_pot     ! POTENTIAL UPTAKE OF NUTRIENT(N/P)for each pool
   !    real(r_8), intent(in) :: available_n  ! AVAILABLE NUTRIENTS FOR GROWTH weighted for each pool

   !    real(r_8), intent(out) :: rnpp        ! REALIZED NPP
   !    logical(l_1), intent(out) :: nl       ! IS LIMITED?

   !    ! Constants
   !    real(r_8), parameter :: ZERO = 0.0_r_8
   !    real(r_8), parameter :: EPS = 1.0e-12_r_8  ! Small tolerance for floating-point comparisons

   !    ! Check for invalid inputs
   !    if (pot_npp_pool < ZERO .or. nupt_pot < ZERO .or. available_n < ZERO) then
   !       write(*,*) "realized_npp: Negative input values are not allowed"
   !    end if

   !    ! NUTRIENT LIMITED NPP TO(CVEGpool):
   !    if (nupt_pot < EPS) then
   !       ! Handle case where nupt_pot is zero or very small
   !       nl = .false.
   !       rnpp = pot_npp_pool
   !    elseif (available_n >= nupt_pot * (1.0_r_8 - EPS)) then
   !       ! There is NO LIMITATION in this case (with tolerance for floating-point)
   !       nl = .false.
   !       ! GROWTH IS ACCOMPLISHED (all npp can go to the CVEG pool)
   !       rnpp = pot_npp_pool
   !    else
   !       ! NPP OF THIS POOL IS LIMITED BY Nutrient X
   !       nl = .true.
   !       ! ACCOMPLISHED NPP
   !       rnpp = max(ZERO, (available_n * pot_npp_pool) / nupt_pot)
   !    endif

   ! end subroutine realized_npp

   subroutine realized_npp(pot_npp_pool, nupt_pot, available_n, rnpp, nl)
      use, intrinsic :: ieee_arithmetic  ! For NaN checks
      real(r_8), intent(in) :: pot_npp_pool
      real(r_8), intent(in) :: nupt_pot
      real(r_8), intent(in) :: available_n
      real(r_8), intent(out) :: rnpp
      logical(l_1), intent(out) :: nl

      ! Constants
      real(r_8), parameter :: ZERO = 0.0_r_8
      real(r_8), parameter :: EPS = 1.0e-12_r_8
      real(r_8), parameter :: REL_EPS = 1.0e-8_r_8

      ! Initialize outputs (avoid NaN)
      rnpp = ZERO
      nl = .false.

      ! Check for NaN/Inf inputs
      if (ieee_is_nan(pot_npp_pool) .or. ieee_is_nan(nupt_pot) .or. ieee_is_nan(available_n)) then
         ! write(*,*) "realized_npp: NaN detected in inputs!"
         return
      endif

      ! Check for negative inputs (invalid case)
      if (pot_npp_pool < ZERO .or. nupt_pot < ZERO .or. available_n < ZERO) then
         ! write(*,*) "realized_npp: Negative inputs not allowed!"
         return
      endif

      ! Early exit if potential NPP is negligible
      if (pot_npp_pool <= EPS) then
         rnpp = ZERO
         nl = .false.
         return
      endif

      ! Case 1: No nutrient limitation (available_n covers demand)
      if (nupt_pot <= EPS * pot_npp_pool) then
         nl = .false.
         rnpp = pot_npp_pool
      elseif (available_n >= nupt_pot * (1.0_r_8 - REL_EPS)) then
         nl = .false.
         rnpp = pot_npp_pool
      else
         ! Case 2: Nutrient-limited growth
         nl = .true.
         if (nupt_pot > EPS * pot_npp_pool) then
            rnpp = max(ZERO, (available_n * pot_npp_pool) / nupt_pot)
         else
            rnpp = ZERO  ! Avoid division by zero
         endif
      endif

      ! Final NaN check (should not happen, but just in case)
      if (ieee_is_nan(rnpp)) then
         ! write(*,*) "realized_npp: NaN in rnpp calculation!"
         rnpp = ZERO
         nl = .false.
      endif

   end subroutine realized_npp

   !=================================================================
   !=================================================================
   !> vcmax_a
   !> Calculates Vcmax based on nitrogen and phosphorus content and specific leaf area
   !> @param npa Nitrogen content in mg g-1
   !> @param ppa Phosphorus content in mg g-1
   !> @param sla Specific leaf area in m2 g-1
   !> @return vcmaxd Vcmax in mol m-2 s-1
   !> @note This function is based on Domingues et al. 2010 (eq.1)
   !=================================================================
   function vcmax_a(npa, ppa, sla) result(vcmaxd)
      ! TESTING eq.1 / Fig 5 Domingues et al. 2010
      real(r_8), intent(in) :: npa       ! N mg g-1
      real(r_8), intent(in) :: ppa       ! P mg g-1
      real(r_8), intent(in) :: sla       ! m2(Leaf) g(C)-1


      real(r_8) :: vcmaxd !mol m⁻² s⁻¹

      real(r_8), parameter :: alpha_n = -1.16D0,&
                              nu_n    = 0.70D0,&
                              alpha_p = -0.30D0,&
                              nu_p    = 0.85D0

      real(r_8) :: ndw, pdw, lma, nlim, plim, vcmax_dw

      ndw = npa
      pdw = ppa
      ! print *, 'ndw', ndw
      ! print *, 'pdw', pdw
      lma = sla ** (-1) ! g/m2

      ! CALCULATE VCMAX
      nlim = alpha_n + nu_n * dlog10(ndw)  ! + (sigma_n * dlog10(sla))
      plim = alpha_p + nu_p * dlog10(pdw)  ! + (sigma_p * dlog10(sla))

      vcmax_dw = min(10**nlim, 10**plim) ! log10(vcmax_dw) in µmol g⁻¹ s⁻¹
      vcmaxd = vcmax_dw * lma * 1.0D-6 ! Multiply by LMA to have area values and 1d-6 to mol m-2 s-1

   end function vcmax_a

   !=================================================================
   !=================================================================
   !> vcmax_a1
   !> Calculates Vcmax based on nitrogen and phosphorus content and specific leaf area
   !> @param npa Nitrogen content in mg g-1
   !> @param ppa Phosphorus content in mg g-1
   !> @param sla Specific leaf area in m2 g-1
   !> @return vcmaxd Vcmax in mol m-2 s-1
   !> @note This function is based on Domingues et al. 2010 (eq.?)
   !=================================================================
   function vcmax_a1(npa, ppa, sla) result(vcmaxd)
      ! TESTING
      real(r_8), intent(in) :: npa   ! N g m-2
      real(r_8), intent(in) :: ppa,sla   ! P g m-2 / m2 g-1


      real(r_8) :: vcmaxd !mol m⁻² s⁻¹

      ! UNITS = LMA Domingues = cm2 g⁻¹ (SLA CAETE = m² g⁻¹)
      ! Dry weight -> mg g⁻¹

      real(r_8), parameter :: alpha_n = -1.56D0,&
                              nu_n    = 0.43D0,&
                              alpha_p = -0.80D0,&
                              nu_p    = 0.45D0,&
                              sigma_n = 0.37D0,&
                              sigma_p = 0.25D0

      real(r_8) :: ndw, pdw, lma, nlim, plim, vcmax_dw

      ndw = npa
      pdw = ppa

      lma = sla ** (-1) ! g/m2

      ! CALCULATE VCMAX
      nlim = alpha_n + nu_n * dlog10(ndw)  + (sigma_n * dlog10(sla))
      plim = alpha_p + nu_p * dlog10(pdw)  + (sigma_p * dlog10(sla))

      vcmax_dw = min(10**nlim, 10**plim) ! log10(vcmax_dw) in µmol g⁻¹ s⁻¹
      vcmaxd = vcmax_dw * lma * 1.0D-6 ! Multiply by LMA to have area values and 1d-6 to mol m-2 s-1

   end function vcmax_a1

   !=================================================================
   !=================================================================
   !> vcmax_b
   !> Calculates Vcmax based on nitrogen content using Domingues et al. 2010 (eq.?)
   !> @param npa Nitrogen content in mg g-1
   !> @return vcmaxd Vcmax in mol m-2 s-1
   !=================================================================
   function vcmax_b(npa) result(vcmaxd)
      ! TESTING Domingues f
      real(r_8), intent(in) :: npa   ! N g m-2
      ! real(r_8), intent(in) :: ppa   ! P g m-2


      real(r_8) :: vcmaxd !mol m⁻² s⁻¹

      real(r_8), parameter :: a = 1.57D0 ,&
                              b = 0.55D0
      real(r_8) :: ndw

      ! CALCULATE VCMAX
      ndw = a + (b * dlog10(npa))
      vcmaxd = 10**ndw * 1D-6


   end function vcmax_b
   !=================================================================
   !=================================================================
   !> photosynthesis_rate
   !> Calculates the photosynthesis rate based on atmospheric CO2 concentration, temperature, light intensity, nitrogen and phosphorus content, leaf turnover time, and whether the plant is C4 or not
   !> @param c_atm Atmospheric CO2 concentration in ppm
   !> @param temp Temperature in °C
   !> @param p0 Atmospheric pressure in hPa
   !> @param ipar Light intensity in mol photons m-2 s-1
   !> @param ll Is light limited? (1 for yes, 0 for no)
   !> @param c4 Is C4 photosynthesis pathway? (1 for yes, 0 for no)
   !> @param nbio Nitrogen content in mg g-1
   !> @param pbio Phosphorus content in mg g-1
   !> @param leaf_turnover Leaf turnover time in years
   !> @param f1ab Instantaneous photosynthesis rate at leaf level in mol CO2 m-2 s-1 (output)
   !> @param vm Maximum carboxylation rate (Vcmax) in mol CO2 m-2 s-1 (output)
   !> @param amax Light saturated photosynthesis rate in mol CO2 m-2 s-1 (output)
   !=================================================================
   !> @note This function is based on the Farquhar/Collatz C3 model for photosynthesis adapted from the CPTEC-PVM2 model
   !> C4 pathway is based on the model of Chen et al. 1994 Ecol. Model. 73 (63-80)
   subroutine photosynthesis_rate(c_atm,temp,p0,ipar,ll,c4,nbio,pbio,&
        & leaf_turnover,f1ab,vm,amax)

      ! f1ab SCALAR returns instantaneous photosynthesis rate at leaf level (molCO2/m2/s)
      ! vm SCALAR Returns maximum carboxilation Rate (Vcmax) (molCO2/m-2 s-1)
      use global_par
      use photo_par
      ! implicit none
      ! I
      real(r_8),intent(in) :: temp  ! temp °C
      real(r_8),intent(in) :: p0    ! atm Pressure hPa
      real(r_8),intent(in) :: ipar  ! mol Photons m-2 s-1
      real(r_8),intent(in) :: nbio, c_atm  ! mg g-1, ppm
      real(r_8),intent(in) :: pbio  ! mg g-1
      integer(i_4),intent(in) :: ll ! is light limited?
      integer(i_4),intent(in) :: c4 ! is C4 Photosynthesis pathway?
      real(r_8),intent(in) :: leaf_turnover   ! y
      ! O
      real(r_8),intent(out) :: f1ab ! Gross CO2 Assimilation Rate mol m-2 s-1
      real(r_8),intent(out) :: vm   ! PLS Vcmax mol m-2 s-1
      real(r_8),intent(out) :: amax ! light saturated PH rate

      real(r_8) :: f2,f3            !Michaelis-Menten CO2/O2 constant (Pa)
      real(r_8) :: mgama,vm_in      !Photo-respiration compensation point (Pa)
      real(r_8) :: rmax, r
      real(r_8) :: ci
      real(r_8) :: jp1
      real(r_8) :: jp2
      real(r_8) :: jp
      real(r_8) :: jc
      real(r_8) :: jl
      real(r_8) :: je,jcl
      real(r_8) :: b,c,c2,b2,es,j1,j2
      real(r_8) :: delta, delta2,aux_ipar
      real(r_8) :: f1a

      ! new vars C4 PHOTOSYNTHESIS
      real(r_8) :: ipar1
      real(r_8) :: tk           ! (K)
      real(r_8) :: t25          ! tk at 25°C (K)
      real(r_8) :: kp
      real(r_8) :: dummy0, dummy1, dummy2
      real(r_8) :: vpm, v4m
      real(r_8) :: cm, cm0, cm1, cm2

      real(r_8) :: nbio2, pbio2, dark_respiration  ! , cbio_aux
      real(r_8), parameter :: light_penalization = 0.2D0, alpha_a = 0.7D0



      ! vpd_effect = min(1.0D0, max(1.0D0 - (0.25D0 * vpd), 0.0D0))
      dark_respiration = 1.0D0 - 0.15D0 ! TODO:there is a problem upstream with the vm calculation


      nbio2 = nbio ! mg (N) g (C) -1
      pbio2 = pbio ! mg (P) g (C) -1

      vm = vcmax_a(nbio2, pbio2, spec_leaf_area(leaf_turnover)) !* vpd_effect  ! 10**vm_nutri * 1D-6
      if(vm .gt. p25) vm = p25
      vm = alpha_a * vm

      ! Rubisco Carboxilation Rate - temperature dependence
      vm_in = (vm*2.0D0**(0.1D0*(temp-25.0D0)))/(1.0D0+dexp(0.3D0*(temp-36.0D0)))

      if(vm_in .gt. p25) vm_in = p25

      if(c4 .eq. 0) then
         !====================-C3 PHOTOSYNTHESIS-===============================
         !Photo-respiration compensation point (Pa)
         mgama = p3/(p8*(p9**(p10*(temp-p11))))
         !Michaelis-Menten CO2 constant (Pa)
         f2 = p12*(p13**(p10*(temp-p11)))
         !Michaelis-Menten O2 constant (Pa)
         f3 = p14*(p15**(p10*(temp-p11)))
         !Saturation vapour pressure (hPa)
         es = real(tetens(temp), r_8)
         !Saturated mixing ratio (kg/kg)
         rmax = 0.622*(es/(p0-es))
         !Moisture deficit at leaf level (kg/kg)
         r = -0.315*rmax
         !Internal leaf CO2 partial pressure (Pa)
         ci = p19 * (1.-(r/p20)) * ((c_atm/9.901)-mgama) + mgama
         !Rubisco carboxilation limited photosynthesis rate (molCO2/m2/s)
         jc = vm_in*((ci-mgama)/(ci+(f2*(1.+(p3/f3)))))
         !Light limited photosynthesis rate (molCO2/m2/s)
         if (ll .eq. 1) then
            aux_ipar = ipar
         else
            aux_ipar = ipar - (ipar * light_penalization)
         endif
         jl = p4*(1.0-p5)*aux_ipar*((ci-mgama)/(ci+(p6*mgama)))
         amax = jl

         ! Transport limited photosynthesis rate (molCO2/m2/s) (RuBP) (re)generation
         ! ---------------------------------------------------
         je = p7*vm_in

         !Jp (minimum between jc and jl)
         !------------------------------
         b = (-1.)*(jc+jl)
         c = jc*jl
         delta = (b**2)-4.0*a*c
         jp1 = (-b-(sqrt(delta)))/(2.0*a)
         jp2 = (-b+(sqrt(delta)))/(2.0*a)
         jp = dmin1(jp1,jp2)

         !Leaf level gross photosynthesis (minimum between jc, jl and je)
         !---------------------------------------------------------------
         b2 = (-1.)*(jp+je)
         c2 = jp*je
         delta2 = (b2**2)-4.0*a2*c2
         j1 = (-b2-(sqrt(delta2)))/(2.0d0*a2)
         j2 = (-b2+(sqrt(delta2)))/(2.0d0*a2)
         f1a = dmin1(j1,j2)

         f1ab = f1a * dark_respiration
         ! f1ab = max(f1a - (vm_in * 0.10), 0.0D0)
         if(f1ab .lt. 0.0D0) f1ab = 0.0D0
         return
      else
         !===========================-C4 PHOTOSYNTHESIS-=============================
         !  USE PHOTO_PAR
         ! ! from Chen et al. 1994
         tk = temp + 273.15           ! K
         t25 = 273.15 + 25.0          ! K
         kp = kp25 * (2.1**(0.1*(tk-t25))) ! ppm

         if (ll .eq. 1) then
            aux_ipar = ipar
         else
            aux_ipar = ipar - (ipar * light_penalization)
         endif

         ipar1 = aux_ipar * 1e6  ! µmol m-2 s-1 - 1e6 converts mol to µmol

         !maximum PEPcarboxylase rate Arrhenius eq. (Dependence on temperature)
         dummy1 = 1.0 + exp((s_vpm * t25 - h_vpm)/(r_vpm * t25))
         dummy2 = 1.0 + exp((s_vpm * tk - h_vpm)/(r_vpm * tk))
         dummy0 = dummy1 / dummy2
         vpm =  vpm25 * exp((-e_vpm/r_vpm) * (1.0/tk - 1.0/t25)) * dummy0

         ! ! actual PEPcarboxylase rate under ipar conditions
         v4m = (alphap * ipar1) / sqrt(1 + alphap**2 * ipar1**2 / vpm**2)

         ! [CO2] mesophyl
         cm0 = 1.674 - 6.1294 * 10.0**(-2) * temp
         cm1 = 1.1688 * 10.0**(-3) * temp ** 2
         cm2 = 8.8741 * 10.0**(-6) * temp ** 3
         cm = 0.7 * c_atm * ((cm0 + cm1 - cm2) / 0.73)

         ! ! When light or PEP carboxylase is limiting
         ! ! FROM CHEN et al. 1994:
         jcl = ((V4m * cm) / (kp + cm)) * 1e-6   ! molCO2 m-2 s-1 / 1e-6 convets µmol 2 mol
         amax = jcl

         ! When V (RuBP regeneration) is limiting
         je = p7 * vm_in

         ! !Leaf level gross photosynthesis (minimum between jcl and je)
         ! !---------------------------------------------------------------
         b2 = (-1.)*(jcl+je)
         c2 = jcl*je
         delta2 = (b2**2)-4.0*a2*c2
         j1 = (-b2-(sqrt(delta2)))/(2.0*a2)
         j2 = (-b2+(sqrt(delta2)))/(2.0*a2)
         f1a = dmin1(j1,j2)

         f1ab = f1a * dark_respiration
         ! f1ab = max(f1a - (vm_in * 0.10), 0.0D0)
         if(f1ab .lt. 0.0D0) f1ab = 0.0D0
         return
      endif
   end subroutine photosynthesis_rate

   !=================================================================
   !=================================================================
   !> spinup3
   !> Performs a spin-up simulation for the carbon pools (leaf, root, wood) based on potential NPP and turnover rates
   !> @param nppot Potential NPP in kg m-2 yr-1
   !> @param dt Array containing turnover times and allocation percentages for leaf, root, and wood compartments
   !> @param cleafini Initial carbon content in leaf compartment (output)
   !> @param cfrootini Initial carbon content in fine root compartment (output)
   !> @param cawoodini Initial carbon content in aboveground woody biomass compartment
   subroutine spinup3(nppot,dt,cleafini,cfrootini,cawoodini)
      implicit none

      !parameters
      integer(kind=i_4),parameter :: ntl=65000

      ! inputs
      integer(kind=i_4) :: kk, k

      real(kind=r_8),intent(in) :: nppot
      real(kind=r_8),dimension(6),intent(in) :: dt
      ! intenal
      real(kind=r_8) :: sensitivity
      real(kind=r_8) :: nppot2
      ! outputs
      real(kind=r_8),intent(out) :: cleafini
      real(kind=r_8),intent(out) :: cawoodini
      real(kind=r_8),intent(out) :: cfrootini

      ! more internal
      real(kind=r_8),dimension(:), allocatable :: cleafi_aux
      real(kind=r_8),dimension(:), allocatable :: cfrooti_aux
      real(kind=r_8),dimension(:), allocatable :: cawoodi_aux

      real(kind=r_8) :: aux_leaf
      real(kind=r_8) :: aux_wood
      real(kind=r_8) :: aux_root
      real(kind=r_8) :: out_leaf
      real(kind=r_8) :: out_wood
      real(kind=r_8) :: out_root

      real(kind=r_8) :: aleaf  !npp percentage alocated to leaf compartment
      real(kind=r_8) :: aawood !npp percentage alocated to aboveground woody biomass compartment
      real(kind=r_8) :: afroot !npp percentage alocated to fine roots compartmentc
      real(kind=r_8) :: tleaf  !turnover time of the leaf compartment (yr)
      real(kind=r_8) :: tawood !turnover time of the aboveground woody biomass compartment (yr)
      real(kind=r_8) :: tfroot !turnover time of the fine roots compartment
      logical(kind=l_1) :: iswoody

      ! catch 'C turnover' traits
      tleaf  = dt(1)
      tawood = dt(2)
      tfroot = dt(3)
      aleaf  = dt(4)
      aawood = dt(5)
      afroot = dt(6)

      iswoody = aawood .gt. 0.0
      allocate(cleafi_aux(ntl))
      allocate(cfrooti_aux(ntl))
      allocate(cawoodi_aux(ntl))

      sensitivity = 1.0001
      if(nppot .le. 0.0) goto 200
      nppot2 = nppot !/real(npls,kind=r_8)
      do k=1,ntl
         if (k.eq.1) then
            cleafi_aux (k) =  aleaf * nppot2
            cawoodi_aux(k) = aawood * nppot2
            cfrooti_aux(k) = afroot * nppot2
         else
            aux_leaf = cleafi_aux(k-1) + (aleaf * nppot2)
            aux_wood = cawoodi_aux(k-1) + (aawood * nppot2)
            aux_root = cfrooti_aux(k-1) + (afroot * nppot2)

            out_leaf = aux_leaf - (cleafi_aux(k-1) / tleaf)
            out_wood = aux_wood - (cawoodi_aux(k-1) / tawood)
            out_root = aux_root - (cfrooti_aux(k-1) / tfroot)

            if(iswoody) then
               cleafi_aux(k) = max(0.0, out_leaf)
               cawoodi_aux(k) = max(0.0, out_wood)
               cfrooti_aux(k) = max(0.0, out_root)
            else
               cleafi_aux(k) = max(0.0, out_leaf)
               cfrooti_aux(k) = max(0.0, out_root)
               cawoodi_aux(k) = 0.0
            endif

            kk =  floor(k*0.66)
            if(iswoody) then
               if((cfrooti_aux(k)/cfrooti_aux(kk).lt.sensitivity).and.&
                    &(cleafi_aux(k)/cleafi_aux(kk).lt.sensitivity).and.&
                    &(cawoodi_aux(k)/cawoodi_aux(kk).lt.sensitivity)) then

                  cleafini = cleafi_aux(k) ! carbon content (kg m-2)
                  cfrootini = cfrooti_aux(k)
                  cawoodini = cawoodi_aux(k)
                  ! print *, 'woody exitet in', k
                  exit
               endif
            else
               if((cfrooti_aux(k)&
                    &/cfrooti_aux(kk).lt.sensitivity).and.&
                    &(cleafi_aux(k)/cleafi_aux(kk).lt.sensitivity)) then

                  cleafini = cleafi_aux(k) ! carbon content (kg m-2)
                  cfrootini = cfrooti_aux(k)
                  cawoodini = 0.0
                  ! print *, 'grass exitet in', k
                  exit
               endif
            endif
         endif
      enddo                 !nt
200   continue
      deallocate(cleafi_aux)
      deallocate(cfrooti_aux)
      deallocate(cawoodi_aux)
   end subroutine spinup3

   ! ===========================================================
   ! ===========================================================
   !> spinup2
   !> Performs a spin-up simulation for the carbon pools (leaf, root, wood) based on potential NPP and turnover rates
   !> @param nppot Potential NPP in kg m-2 yr-1
   !> @param dt Array containing turnover times and allocation percentages for leaf, root, and wood compartments
   !> @param cleafini Initial carbon content in leaf compartment (output)
   !> @param cfrootini Initial carbon content in fine root compartment (output)
   !> @param cawoodini Initial carbon content in aboveground woody biomass compartment
   subroutine spinup2(nppot,dt,cleafini,cfrootini,cawoodini)
      use global_par, only: ntraits,npls
      implicit none

      !parameters
      integer(kind=i_4),parameter :: ntl=36525

      ! inputs
      integer(kind=i_4) :: i6, kk, k

      real(kind=r_8),intent(in) :: nppot
      real(kind=r_8),dimension(ntraits, npls),intent(in) :: dt
      ! intenal
      real(kind=r_8) :: sensitivity
      real(kind=r_8) :: nppot2
      ! outputs
      real(kind=r_8),dimension(npls),intent(out) :: cleafini
      real(kind=r_8),dimension(npls),intent(out) :: cfrootini
      real(kind=r_8),dimension(npls),intent(out) :: cawoodini

      ! more internal
      real(kind=r_8),dimension(:), allocatable :: cleafi_aux
      real(kind=r_8),dimension(:), allocatable :: cfrooti_aux
      real(kind=r_8),dimension(:), allocatable :: cawoodi_aux

      real(kind=r_8) :: aux_leaf
      real(kind=r_8) :: aux_wood
      real(kind=r_8) :: aux_root
      real(kind=r_8) :: out_leaf
      real(kind=r_8) :: out_wood
      real(kind=r_8) :: out_root

      real(kind=r_8),dimension(npls) :: aleaf  !npp percentage alocated to leaf compartment
      real(kind=r_8),dimension(npls) :: aawood !npp percentage alocated to aboveground woody biomass compartment
      real(kind=r_8),dimension(npls) :: afroot !npp percentage alocated to fine roots compartmentc
      real(kind=r_8),dimension(npls) :: tleaf  !turnover time of the leaf compartment (yr)
      real(kind=r_8),dimension(npls) :: tawood !turnover time of the aboveground woody biomass compartment (yr)
      real(kind=r_8),dimension(npls) :: tfroot !turnover time of the fine roots compartment
      logical(kind=l_1) :: iswoody

      allocate(cleafi_aux(ntl))
      allocate(cfrooti_aux(ntl))
      allocate(cawoodi_aux(ntl))

      ! catch 'C turnover' traits
      tleaf  = dt(3,:)
      tawood = dt(4,:)
      tfroot = dt(5,:)
      aleaf  = dt(6,:)
      aawood = dt(7,:)
      afroot = dt(8,:)

      sensitivity = 1.01
      if(nppot .le. 0.0) goto 200
      nppot2 = nppot !/real(npls,kind=r_8)
      do i6=1,npls
         iswoody = ((aawood(i6) .gt. 0.0) .and. (tawood(i6) .gt. 0.0))
         do k=1,ntl
            if (k .eq. 1) then
               cleafi_aux (k) =  aleaf(i6) * nppot2
               cawoodi_aux(k) = aawood(i6) * nppot2
               cfrooti_aux(k) = afroot(i6) * nppot2

            else
               aux_leaf = cleafi_aux(k-1) + (aleaf(i6) * nppot2)
               aux_wood = cawoodi_aux(k-1) + (aawood(i6) * nppot2)
               aux_root = cfrooti_aux(k-1) + (afroot(i6) * nppot2)

               out_leaf = aux_leaf - (cleafi_aux(k-1) / tleaf(i6))
               out_wood = aux_wood - (cawoodi_aux(k-1) / tawood(i6))
               out_root = aux_root - (cfrooti_aux(k-1) / tfroot(i6))

               if(iswoody) then
                  cleafi_aux(k) = max(0.0, out_leaf)
                  cawoodi_aux(k) = max(0.0, out_wood)
                  cfrooti_aux(k) = max(0.0, out_root)
               else
                  cleafi_aux(k) = max(0.0, out_leaf)
                  cawoodi_aux(k) = 0.0
                  cfrooti_aux(k) = max(0.0, out_root)
               endif

               kk =  floor(k*0.66)
               if(iswoody) then
                  if((cfrooti_aux(k)/cfrooti_aux(kk).lt.sensitivity).and.&
                       &(cleafi_aux(k)/cleafi_aux(kk).lt.sensitivity).and.&
                       &(cawoodi_aux(k)/cawoodi_aux(kk).lt.sensitivity)) then

                    cleafini(i6) = cleafi_aux(k) ! carbon content (kg m-2)
                    cfrootini(i6) = cfrooti_aux(k)
                    cawoodini(i6) = cawoodi_aux(k)
                    exit
                  endif
               else
                  if((cfrooti_aux(k)&
                       &/cfrooti_aux(kk).lt.sensitivity).and.&
                       &(cleafi_aux(k)/cleafi_aux(kk).lt.sensitivity)) then

                    cleafini(i6) = cleafi_aux(k) ! carbon content (kg m-2)
                    cfrootini(i6) = cfrooti_aux(k)
                    cawoodini(i6) = 0.0
                    exit
                  endif
               endif
            endif
         enddo                  !nt
      enddo                     !npls
200   continue
   deallocate(cleafi_aux)
   deallocate(cfrooti_aux)
   deallocate(cawoodi_aux)
   end subroutine spinup2

  !===================================================================
  !===================================================================
  !> Based on Ryan 1991; Sitch et al. 2003; Levis et al. 2004
  !> This function calculates the temperature response of respiration
  !> @param temp Temperature in °C
  !> @return gtemp Temperature response of respiration in kgC/m2/yr
   function resp_aux(temp) result(gtemp)

  real(r_8), intent(in) :: temp
  real(r_8) :: gtemp

  if (temp .ge.  -50.0) then
     gtemp = exp(308.56 * (1.0 / 56.02 - 1.0 / (temp + 273.15 + 46.02)))
  else
     gtemp = 0.0
  endif

  end function resp_aux

  !===================================================================
  !===================================================================
  !> Deprecated function, use resp_aux instead
  !> This function calculates the temperature response of respiration
  !> @param temp Temperature in °C
  !> @return gtemp Temperature response of respiration in kgC/m2/yr
  !===================================================================
  function f(temp) result(gtemp)

  real(r_8), intent(in) :: temp
  real(r_8), parameter :: beta = 0.069
  real(r_8) :: gtemp

  if (temp .ge.  -50.0) then
     gtemp = exp(beta * (temp + 273.15))
  else
     gtemp = 0.0
  endif

  end function f

  !===================================================================
  !===================================================================
   !> This function calculates the maintenance respiration
   !> @param temp Temperature in °C
   !> @param ts Soil temperature in °C
   !> @param cl1_mr Carbon content in leaf pool (kgC/m2)
   !> @param cf1_mr Carbon content in fine root pool (kgC/m2)
   !> @param ca1_mr Carbon content in aboveground woody biomass pool (kgC/m2)
   !> @param n2cl Nitrogen content in leaf pool (kgN/m2)
   !> @param n2cw Nitrogen content in sapwood pool (kgN/m2)
   !> @param n2cf Nitrogen content in fine root pool (kgN/m2)
   !> @param aawood_mr Carbon content in aboveground woody biomass pool (kgC/m2)
   !> @return rm Maintenance respiration in kgC/m2/yr
   !> @author: JPdarela Adapted from LPJ-GUESS code
   !> @date 2023-10-01
   !> @version 1.0
   !===================================================================
   function m_resp(temp, ts,cl1_mr,cf1_mr,ca1_mr,&
        & n2cl,n2cw,n2cf,aawood_mr) result(rm)

      use global_par, only: sapwood
      !implicit none

      real(r_8), intent(in) :: temp, ts
      real(r_8), intent(in) :: cl1_mr
      real(r_8), intent(in) :: cf1_mr
      real(r_8), intent(in) :: ca1_mr
      real(r_8), intent(in) :: n2cl
      real(r_8), intent(in) :: n2cw
      real(r_8), intent(in) :: n2cf
      real(r_8), intent(in) :: aawood_mr
      real(r_8) :: rm

      real(r_8) :: csa, rm64, rml64
      real(r_8) :: rmf64, rms64
      real(r_8), parameter :: k=0.095218D0
      real(r_8), parameter :: rcoeff_leaf = 3.2D0, rcoeff_wood = 3.0D0, rcoeff_froot = 3.0D0

      !   Autothrophic respiration
      !   ========================
      !   Maintenance respiration (kgC/m2/yr)

      if(aawood_mr .gt. 0.0) then
         csa = sapwood * ca1_mr
         rms64 =  rcoeff_wood * k * csa * n2cw * resp_aux(temp)
      else
         rms64 = 0.0
      endif

      rml64 = rcoeff_leaf * k * cl1_mr * n2cl * resp_aux(temp)

      rmf64 = rcoeff_froot * k * cf1_mr * n2cf * resp_aux(ts)

      rm64 = rml64 + rmf64 + rms64 !* 1D-3

      rm = real(rm64,r_8)

      if (rm .lt. 0.0) then
         rm = 0.0
      endif

   end function m_resp

   !===================================================================
   !===================================================================
   !> Storage pool respiration
   !> @param temp Temperature in °C
   !> @param sto_mr Storage pool carbon content in kgC/m2 (3 elements: [1] = leaf, [2] = fine root, [3] = aboveground woody biomass)
   !> @return rm Storage pool respiration in kgC/m2/yr
   !====================================================================
   function sto_resp(temp, sto_mr) result(rm)
    !implicit none

      real(r_8), intent(in) :: temp
      real(r_8), dimension(3), intent(in) :: sto_mr
      real(r_8) :: rm

      real(r_8) :: stoc,ston
      real(r_8), parameter :: k=0.095218D0, rcoeff = 2.0D0

    !   Autothrophic respiration
    !   ========================

    stoc = sto_mr(1)
    ston = sto_mr(2)
   !  print*, ston

    if(stoc .le. 0.0D0) then
       rm = 0.0D0
       return
    endif

    if(ston .le. 0.0D0) then
      ston = 1.0D0/300.0D0
    else
      ston = ston/stoc
    endif

   !  rm = ((ston * stoc) * a1 * dexp(a2 * temp))
    rm = rcoeff * k  * stoc * ston * resp_aux(temp)

    if (rm .lt. 0) then
       rm = 0.0
    endif
    return
   end function sto_resp

   !====================================================================
   !====================================================================
   !> Growth respiration
   !> @param construction Construction cost in kgC/m2
   !> @return rg Growth respiration in kgC/m2/yr
   function g_resp(construction) result(rg)
      !implicit none

      real(r_8), intent(in) :: construction
      real(r_8) :: rg

      !     Autothrophic respiration
      !     Growth respiration (KgC/m2/yr)(based in Ryan 1991; Sitch et al.
      !     2003; Levis et al. 2004)
      if (construction .le. 0.0) then
         rg = 0.0
      else
         rg = real(0.25D0 * construction * 1.0D-3, kind=r_8)
      endif

   end function g_resp

   !====================================================================
   !====================================================================
   !> tetens
   !> This function calculates the saturation vapor pressure using the Arden Buck equation.
   !> @param t Temperature in °C
   !> @return es Saturation vapor pressure in hPa
   !====================================================================
   function tetens(t) result(es)
      ! returns Saturation Vapor Pressure (hPa), using Buck equation

      ! buck equation...references:
      ! http://www.hygrometers.com/wp-content/uploads/CR-1A-users-manual-2009-12.pdf
      ! Hartmann 1994 - Global Physical Climatology p.351
      ! https://en.wikipedia.org/wiki/Arden_Buck_equation#CITEREFBuck1996

      ! Buck AL (1981) New Equations for Computing Vapor Pressure and Enhancement Factor.
      !      J. Appl. Meteorol. 20:1527–1532.

      real(r_8),intent( in) :: t
      real(r_8) :: es

      if (t .ge. 0.) then
         es = 6.1121 * exp((18.729-(t/227.5))*(t/(257.87+t))) ! Arden Buck
         !es = es * 10 ! transform kPa in mbar == hPa
         return
      else
         es = 6.1115 * exp((23.036-(t/333.7))*(t/(279.82+t))) ! Arden Buck
         !es = es * 10 ! mbar == hPa ! mbar == hPa
         return
      endif

   end function tetens

   !====================================================================
   !====================================================================
   !> pft_area_frac
   !> This subroutine calculates the area fraction occupied by each PFT based on their carbon content.
   !> @param cleaf1 Carbon content in leaf pool (kg m-2)
   !> @param cfroot1 Carbon content in fine root pool (kg m-2)
   !> @param cawood1 Carbon content in aboveground woody biomass pool (kg m-2)
   !> @param awood Allocation coefficient to wood (dimensionless)
   !> @param ocp_coeffs Output occupation coefficients (area fraction) for each PFT
   !> @param ocp_wood Output occupation coefficients for wood (integer)
   !> @param run_pls Output array indicating whether each PFT is running (1) or not (0)
   !> @param c_to_soil Output array of carbon transferred to soil (not implemented in budget)
   !=======================================================================
   subroutine pft_area_frac(cleaf1, cfroot1, cawood1, awood,&
                          & ocp_coeffs, ocp_wood, run_pls, c_to_soil)

      use global_par, only: npls, cmin, sapwood
      !implicit none

      integer(kind=i_4),parameter :: npft = npls

      real(kind=r_8),dimension(npft),intent( in) :: cleaf1, cfroot1, cawood1 ! carbon content (kg m-2)
      real(kind=r_8),dimension(npft),intent( in) :: awood ! npp allocation coefficient to wood
      real(kind=r_8),dimension(npft),intent(out) :: ocp_coeffs ! occupation coefficients (area fraction)
      integer(kind=i_4),dimension(npft),intent(out) :: ocp_wood !
      integer(kind=i_4),dimension(npft),intent(out) :: run_pls
      real(kind=r_8), dimension(npls), intent(out) :: c_to_soil ! NOT IMPLEMENTED IN BUDGET
      logical(kind=l_1),dimension(npft) :: is_living
      real(kind=r_8),dimension(npft) :: cleaf, cawood, cfroot
      real(kind=r_8),dimension(npft) :: total_biomass_pft,total_w_pft
      integer(kind=i_4) :: p,i
      integer(kind=i_4),dimension(1) :: max_index
      real(kind=r_8) :: total_biomass, total_wood
      integer(kind=i_4) :: five_percent
      integer(kind=i_4) :: living_plss

      total_biomass = 0.0D0
      total_wood = 0.0D0

      cleaf = cleaf1
      cfroot = cfroot1
      cawood = cawood1

      do p = 1, npft
         if(awood(p) .le. 0.0D0) then
            cawood(p) = 0.0D0
         endif
      enddo


      do p = 1,npft
         total_w_pft(p) = 0.0D0
         total_biomass_pft(p) = 0.0D0
         ocp_coeffs(p) = 0.0D0
         ocp_wood(p) = 0
      enddo

      ! check for nan in cleaf cawood cfroot
      do p = 1,npft
         if(isnan(cleaf(p))) cleaf(p) = 0.0D0
         if(isnan(cfroot(p))) cfroot(p) = 0.0D0
         if(isnan(cawood(p))) cawood(p) = 0.0D0
      enddo

      do p = 1,npft
         if(cleaf(p) .lt. cmin .or. cfroot(p) .lt. cmin) then
            is_living(p) = .false.
            c_to_soil(p) = cleaf(p) + cawood(p) + cfroot(p)
            cleaf(p) = 0.0D0
            cawood(p) = 0.0D0
            cfroot(p) = 0.0D0
         else
            is_living(p) = .true.
            c_to_soil(p) = 0.0D0
         endif
      enddo

      do p = 1,npft
         ! total_biomass_pft(p) = cleaf(p) + cfroot(p) + (sapwood * cawood(p)) ! only sapwood?
         if (is_living(p)) then
            total_biomass_pft(p) = cleaf(p) + cfroot(p) + cawood(p)
            total_biomass = total_biomass + total_biomass_pft(p)
            total_wood = total_wood + cawood(p)
            total_w_pft(p) = cawood(p)
         endif
      enddo

      !     grid cell occupation coefficients
      if(total_biomass .gt. 0.0D0) then
         do p = 1,npft
            ocp_coeffs(p) = total_biomass_pft(p) / total_biomass
            if(ocp_coeffs(p) .lt. 0.0D0) ocp_coeffs(p) = 0.0D0

            if(ocp_coeffs(p) .gt. 0.0D0 .and. is_living(p)) then
               run_pls(p) = 1
            else
               run_pls(p) = 0
            endif
            !if(isnan(ocp_coeffs(p))) ocp_coeffs(p) = 0.0
         enddo
      else
         do p = 1,npft
            ocp_coeffs(p) = 0.0D0
            run_pls(p) = 0
         enddo
      endif

      !     gridcell pft ligth limitation by wood content
      living_plss = sum(run_pls)
      five_percent = nint(living_plss * 0.05)
      ! print*, 'five_percent', five_percent
      ! print*, 'living_plss', living_plss

      if(five_percent .eq. 0) five_percent = 1
      if(five_percent .eq. 1) then
         if(total_wood .gt. 0.0) then
            max_index = maxloc(total_w_pft)
            i = max_index(1)
            ocp_wood(i) = 1
         endif
      else
         do p = 1,five_percent
            if(total_wood .gt. 0.0D0) then
               max_index = maxloc(total_w_pft)
               ! print*, 'max_index', max_index

               i = max_index(1)
               total_w_pft(i) = 0.0D0
               ocp_wood(i) = 1
            endif
         enddo
      endif

   end subroutine pft_area_frac

   !====================================================================
   !====================================================================
   !> vec_ranging
   !> This subroutine rescales a vector of values to a new range defined by new_min and new_max.
   !> @param values Input vector of values to be rescaled
   !> @param new_min New minimum value for the rescaled range
   !> @param new_max New maximum value for the rescaled range
   !> @param output Output vector containing the rescaled values
   !====================================================================
   subroutine vec_ranging(values, new_min, new_max, output)
       implicit none
       real, dimension(:), intent(in) :: values
       real, intent(in) :: new_min, new_max
       real, dimension(size(values)), intent(out) :: output
       real :: old_min, old_max
       integer :: i

       old_min = minval(values)
       old_max = maxval(values)

       do i = 1, size(values)
           output(i) = (new_max - new_min) / (old_max - old_min) * (values(i) - old_min) + new_min
       end do
   end subroutine vec_ranging

end module photo
