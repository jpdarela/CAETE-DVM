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


contains

  subroutine prod(dt,light_limit,catm,temp,ts,p0,w,ipar,rh,emax,cl1_prod,&
       & ca1_prod,cf1_prod,wmax,constr,ph,ar, nppa,laia,f5,vpd,rm,rg,rc,wue,c_defcit,vm_out,sla,e)

    use types
    use global_par
    use photo_par
    use photo
    use water

!Input
!-----
    real(r_8),dimension(ntraits),intent(in) :: dt ! PLS data
    real(r_4), intent(in) :: temp, ts                 !Mean monthly temperature (oC)
    real(r_4), intent(in) :: p0                   !Mean surface pressure (hPa)
    real(r_8), intent(in) :: w                    !Soil moisture kg m-2
    real(r_4), intent(in) :: ipar                 !Incident photosynthetic active radiation (w/m2)
    real(r_4), intent(in) :: rh,emax !Relative humidity/MAXIMUM EVAPOTRANSPIRATION
    real(r_8), intent(in) :: catm, cl1_prod, cf1_prod, ca1_prod        !Carbon in plant tissues (kg/m2)

    real(r_8), intent(in) :: wmax
    integer(i_4), intent(in) :: light_limit                !True for no ligth limitation
    real(r_8), intent(in) :: constr
!     Output
!     ------
    real(r_4), intent(out) :: ph                   !Canopy gross photosynthesis (kgC/m2/yr)
    real(r_4), intent(out) :: rc                   !Stomatal resistence (not scaled to canopy!) (s/m)
    real(r_8), intent(out) :: laia                 !Autotrophic respiration (kgC/m2/yr)
    real(r_4), intent(out) :: ar                   !Leaf area index (m2 leaf/m2 area)
    real(r_4), intent(out) :: nppa                 !Net primary productivity (kgC/m2/yr)
    real(r_4), intent(out) :: vpd
    real(r_8), intent(out) :: f5                   !Water stress response modifier (unitless)
    real(r_4), intent(out) :: rm                   !autothrophic respiration (kgC/m2/day)
    real(r_4), intent(out) :: rg
    real(r_4), intent(out) :: wue
    real(r_4), intent(out) :: c_defcit     ! Carbon deficit gm-2 if it is positive, aresp was greater than npp + sto2(1)
    real(r_8), intent(out) :: sla, e        !specific leaf area (m2/kg)
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
    real(r_8) :: f1a, co2_pp, shade_lai, sun_lai, f4_sun, f4_shade      !auxiliar_f1

    real(r_4) :: rc_pot

!getting pls parameters


    g1         = dt(1)
    tleaf      = dt(3)
    awood      = dt(7)
    c4         = dt(9)
    n2cl       = dt(10)
    n2cl_resp  = n2cl
    n2cw_resp  = dt(11)
    n2cf_resp  = dt(12)
    p2cl       = dt(13)


    n2cl = n2cl * 1.0D3 ! N in leaf mg g-1
    p2cl = p2cl * 1.0D3 ! P in leaf mg g-1

    c4_int = idnint(c4)


!     ==============
!     Photosynthesis
!     ==============
!     (molCO2/m2/s)

    ! VPD
    !========
    vpd = vapor_p_deficit(temp, rh)

    call photosynthesis_rate(catm,temp,p0,ipar,light_limit,c4_int,n2cl,&
         & p2cl,tleaf,f1a,vm_out,jl_out)

    !Stomatal resistence
    !===================
    co2_pp = (catm * (p0 * 1.0D2) / 1.0D6) /  (p0 * 1.0D2) ! term used in the medlyn model
    rc_pot = stomatal_resistance(vpd, f1a, g1, co2_pp) ! Potential RCM leaf level - s m-1
    if (rc_pot .gt. rcmax) rc_pot = rcmax
    if (rc_pot .lt. rcmin) rc_pot = rcmin

    !Water stress response modifier (dimensionless)
    !----------------------------------------------
    f5 =  water_stress_modifier(w, cf1_prod, rc_pot, emax, wmax)

!     Photosysthesis minimum and maximum temperature
!     ----------------------------------------------

    if ((temp.ge.-10.0).and.(temp.le.50.0)) then
       f1 = f1a * f5 ! :water stress factor
    else
       f1 = 0.0      !Temperature above/below photosynthesis windown
    endif

    ! rc_aux = stomatal_resistance(vpd, f1, g1, co2_pp)  ! RCM leaf level -!s m-1
    ! if (rc_aux .gt. rcmax) rc_aux = rcmax
    ! if (rc_aux .lt. rcmin) rc_aux = rcmin

    wue = water_ue(f1, rc_pot, p0, vpd)

    !     calcula a transpiração em mm/s
    e = transpiration(rc_pot, p0, vpd, 2)

    ! Leaf area index (m2/m2)
    ! recalcula rc e escalona para dossel
    ! laia = 0.2D0 * dexp((2.5D0 * f1)/p25)
    sla = spec_leaf_area(tleaf)  ! m2 g-1  ! Convertions made in leaf_area_index &  gross_ph + calls therein

    ! laia = f_four(90, cl1_prod, sla)
    shade_lai = f_four(20, cl1_prod, sla)
    sun_lai = f_four(90, cl1_prod, sla)
    f4_shade = f_four(2, cl1_prod, sla) ! 20% of the canopy is shade
    f4_sun = f_four(1, cl1_prod, sla) ! 80% of the canopy is sun
    ! laia = leaf_area_index(cl1_prod, sla)
    laia = shade_lai + sun_lai


    rc = rc_pot / (f4_shade * f4_sun) !/ real(laia,kind=r_4) ! RCM -!s m-1 ! CANOPY SCALING --
    ! rc = stomatal_conductance(vpd, f1, g1, catm) * laia
!     Canopy gross photosynthesis (kgC/m2/yr)
!     =======================================x

    ph =  gross_ph(f1,cl1_prod,sla)       ! kg m-2 year-1

!     Autothrophic respiration
!     ========================
!     Maintenance respiration (kgC/m2/yr) (based in Ryan 1991)
    rm = m_resp(temp,ts,cl1_prod,cf1_prod,ca1_prod &
         &,n2cl_resp,n2cw_resp,n2cf_resp,awood)

! c     Growth respiration (KgC/m2/yr)(based in Ryan 1991; Sitch et al.
! c     2003; Levis et al. 2004)
    ! rg = g_resp(beta_leaf,beta_awood, beta_froot,awood)
    rg = g_resp(constr)

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
       ph = 0.0
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

  end subroutine prod

end module productivity
