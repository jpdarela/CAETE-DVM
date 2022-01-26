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

module productivity
  implicit none
  private

  public :: prod

!   public ::                     &
!         prod                   ,& ! (s), productivity 
!         ecosystem_services        ! (s), SE_module 


contains

    subroutine prod(dt,catm,temp,ts,p0,w,ipar,sla1,rh,emax,cl1_prod,&
        & ca1_prod,cf1_prod,beta_leaf,beta_awood,beta_froot,height1,max_height,wmax,ph,ar,&
        & nppa,laia,f5,vpd,rm,rg,rc,wue,c_defcit,vm_out,e)

        use types
        use global_par
        use photo_par
        use photo
        use water

    !Input
    !-----
        real(r_8),dimension(ntraits),intent(in) :: dt !PLS data
        real(r_4), intent(in) :: temp, ts             !Mean monthly temperature (oC)
        real(r_4), intent(in) :: p0                   !Mean surface pressure (hPa)
        real(r_8), intent(in) :: w                    !Soil moisture kg m-2
        real(r_8), intent(in) :: ipar                 !Incident photosynthetic active radiation (w/m2)
        real(r_4), intent(in) :: rh,emax              !Relative humidity/MAXIMUM EVAPOTRANSPIRATION
        real(r_8), intent(in) :: catm, cl1_prod, cf1_prod, ca1_prod        !Carbon in plant tissues (kg/m2)
        real(r_8), intent(in) :: beta_leaf            !npp allocation to carbon pools (kg/m2/day)
        real(r_8), intent(in) :: beta_awood
        real(r_8), intent(in) :: beta_froot, wmax
        ! logical(l_1), intent(in) :: light_limit                !True for no ligth limitation
        real(r_8), intent(in) :: height1
        real(r_8), intent(in) :: max_height
        real(r_8), intent(in) :: sla1

    !     Output
    !     ------
        real(r_4), intent(out) :: ph                   !Canopy gross photosynthesis (kgC/m2/yr)
        real(r_4), intent(out) :: rc                   !Stomatal resistence (not scaled to canopy!) (s/m)
        real(r_8), intent(out) :: laia                 !Autotrophic respiration (kgC/m2/yr)
        real(r_4), intent(out) :: ar                   !Leaf area index (m2 leaf/m2 area)
        real(r_4), intent(out) :: nppa                 !Net primary productivity (kgC/m2/yr)
        ! real(r_4), intent(out) :: npp_accu
        real(r_4), intent(out) :: vpd
        real(r_8), intent(out) :: f5                   !Water stress response modifier (unitless)
        real(r_4), intent(out) :: rm                   !autothrophic respiration (kgC/m2/day)
        real(r_4), intent(out) :: rg
        real(r_4), intent(out) :: wue
        real(r_4), intent(out) :: c_defcit     ! Carbon deficit gm-2 if it is positive, aresp was greater than npp + sto2(1)
        real(r_8), intent(out) :: e !sla       !specific leaf area (m2/kg)
        real(r_8), intent(out) :: vm_out
    !     Internal
    !     --------

        real(r_8) :: tleaf,awood            !leaf/wood turnover time (yr)
        real(r_8) :: g1
        real(r_8) :: c4

        real(r_8) :: n2cl
        real(r_8) :: n2cl_resp
        real(r_8) :: n2cw_resp
        real(r_8) :: n2cf_resp
        real(r_8) :: p2cl
        integer(i_4) :: c4_int
        real(r_8) :: jl_out

        real(r_8) :: f1       !Leaf level gross photosynthesis (molCO2/m2/s)
        real(r_8) :: f1a      !auxiliar_f1
        real(r_4) :: rc_pot, rc_aux

    !getting pls parameters


        g1  = dt(1)
        tleaf = dt(3)
        awood = dt(7)
        c4  = dt(9)
        n2cl = dt(10)
        n2cl_resp = n2cl
        n2cw_resp = dt(11)
        n2cf_resp = dt(12)
        p2cl = dt(13)


        n2cl = n2cl * (cl1_prod * 1D3) ! N in leaf g m-2
        p2cl = p2cl * (cl1_prod * 1D3) ! P in leaf g m-2

        c4_int = idnint(c4)


    !     ==============
    !     Photosynthesis
    !     ==============
    ! rate (molCO2/m2/s)

        call photosynthesis_rate(catm,temp,p0,ipar,c4_int,n2cl,&
            & p2cl,cl1_prod,sla1,ca1_prod,height1,max_height,f1a,vm_out,jl_out)


        ! VPD
        !========
        vpd = vapor_p_defcit(temp,rh)

        !Stomatal resistence
        !===================
        rc_pot = canopy_resistence(vpd, f1a, g1, catm) ! Potential RCM leaf level - s m-1

        !Water stress response modifier (dimensionless)
        !----------------------------------------------
        f5 =  water_stress_modifier(w, cf1_prod, rc_pot, emax, wmax)


    !     Photosysthesis minimum and maximum temperature
    !     ----------------------------------------------

        if ((temp.ge.-10.0).and.(temp.le.50.0)) then
        f1 = f1a * f5 ! :water stress factor ! Ancient floating-point underflow spring (from CPTEC-PVM2)
        else
        f1 = 0.0      !Temperature above/below photosynthesis windown
        endif

        rc_aux = canopy_resistence(vpd, f1, g1, catm)  ! RCM leaf level -!s m-1

        wue = water_ue(f1, rc_aux, p0, vpd)


        !     calcula a transpiração em mm/s
        e = transpiration(rc_aux, p0, vpd, 2)

        ! Leaf area index (m2/m2)
        ! recalcula rc e escalona para dossel
        ! laia = 0.2D0 * dexp((2.5D0 * f1)/p25)
        ! sla = spec_leaf_area(tleaf)  ! m2 g-1  ! Convertions made in leaf_area_index &  gross_ph + calls therein

        laia = leaf_area_index(cl1_prod, sla1)
        rc = rc_aux !* real(laia,kind=r_4) ! RCM -!s m-1 ! CANOPY SCALING --

    !     Canopy gross photosynthesis (kgC/m2/yr)
    !     =======================================x

        ph =  gross_ph(f1,cl1_prod,sla1)       ! kg m-2 year-1

    !     Autothrophic respiration
    !     ========================
    !     Maintenance respiration (kgC/m2/yr) (based in Ryan 1991)
        rm = m_resp(temp,ts,cl1_prod,cf1_prod,ca1_prod &
            &,n2cl_resp,n2cw_resp,n2cf_resp,awood)

    ! c     Growth respiration (KgC/m2/yr)(based in Ryan 1991; Sitch et al.
    ! c     2003; Levis et al. 2004)
        rg = g_resp(beta_leaf,beta_awood, beta_froot,awood)

        if (rg.lt.0) then
        rg = 0.0
        endif

    !     c Autotrophic (plant) respiration -ar- (kgC/m2/yr)
    !     Respiration minimum and maximum temperature
    !     -------------------------------------------
        if ((temp.ge.-10.0).and.(temp.le.50.0)) then
        ar = rm + rg
        else
        ar = 0.0               !Temperature above/below respiration windown
        endif
    !     Net primary productivity(kgC/m2/yr)
    !     ====================================
        nppa = ph - ar
    ! this operation affects the model mass balance
    ! If ar is bigger than ph, what is the source or respired C?

        if(ar .gt. ph) then
        c_defcit = ((ar - ph) * 2.73791) ! tranform kg m-2 year-1 in  g m-2 day-1
        nppa = 0.0
        else
        c_defcit = 0.0
        endif

        !Variable to calculates the accumulation of NPP
        ! npp_accu = accumulate_npp(nppa)

    end subroutine prod


end module productivity


module ecosystem_services

    implicit none
    private

    public :: SE_module

contains

    subroutine SE_module (cl1_se,cw1_se,cr1_se,evapot,&
        &prec,co2_abs,evap_se,water_ret)

        !Coded by: Bárbara Cardeli (2022)

        use types
        use global_par, only: npls

        integer(i_4),parameter :: npft = npls
        integer(i_4) :: p 

        !INPUTS
        real(r_8), dimension(npft),intent(in) :: cl1_se, cw1_se, cr1_se
        real(r_4), dimension(npft),intent(in) :: evapot !,rnoff
        real(r_8), intent(in) :: prec
        
        !OUTPUTS
        real(r_8), intent(out) :: co2_abs,evap_se,water_ret

        !INTERNAL
        real(r_8), dimension(npft) :: tissues_biomass
        real(r_8) :: total_biomass, total_evap


        do p = 1, npls
            !Loop to accumulate biomass
            tissues_biomass(p) = cl1_se(p) + cw1_se(p) + cr1_se(p)
            total_biomass = total_biomass + tissues_biomass(p)

            !Loop to accumulate evapotranspiration rates
            total_evap = total_evap + evapot(p)
        enddo

        !   CARBON (CO2) SEQUESTRATION (kgC/m2/yr)
        !============================================!
        !this equation calculates the absorption and 
        !sequestration of carbon dioxide (CO2) in plant tissues

        co2_abs = (total_biomass*3.67)*365.242D0

        !   EVAPOTRANSPIRATION RATES (mm/yr)
        !======================================!

        evap_se = total_evap*365.242D0

        !   WATER RETENTION IN SOIL (m3/yr)
        !======================================!

        water_ret = ((prec*6.9)*365.242D0-(total_evap))

        !*****************PRECISA ZERAR AO FINAL DE CADA ANO!!!!!**********************

    end subroutine SE_module


end module ecosystem_services
