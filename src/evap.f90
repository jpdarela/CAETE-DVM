module water

    ! this module defines functions related to surface water balance
    implicit none
    private

    ! functions defined here:

    public ::              &
         wtt              ,&
         soil_temp        ,&
         soil_temp_sub    ,&
         penman           ,&
         evpot2           ,&
         available_energy ,&
         runoff


  contains


    function wtt(t) result(es)
      ! returns Saturation Vapor Pressure (hPa), using Buck equation

      ! buck equation...references:
      ! http://www.hygrometers.com/wp-content/uploads/CR-1A-users-manual-2009-12.pdf
      ! Hartmann 1994 - Global Physical Climatology p.351
      ! https://en.wikipedia.org/wiki/Arden_Buck_equation#CITEREFBuck1996

      ! Buck AL (1981) New Equations for Computing Vapor Pressure and Enhancement Factor.
      !      J. Appl. Meteorol. 20:1527–1532.

      use types, only: r_8
      !implicit none

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

    end function wtt

    !====================================================================
    !====================================================================


    subroutine soil_temp_sub(temp, tsoil)
      ! Calcula a temperatura do solo. Aqui vamos mudar no futuro!
      ! a tsoil deve ter relacao com a et realizada...
      ! a profundidade do solo (H) e o coef de difusao (DIFFU) devem ser
      ! variaveis (MAPA DE SOLO?; agua no solo?)
      use types
      use global_par
      !implicit none
      integer(i_4),parameter :: m = 1095

      real(r_8),dimension(m), intent( in) :: temp ! future __ make temps an allocatable array
      real(r_8), intent(out) :: tsoil

      ! internal vars

      integer(i_4) :: n, k
      real(r_8) :: t0 = 0.0
      real(r_8) :: t1 = 0.0

      tsoil = -9999.0

      do n=1,m !run to attain equilibrium
        k = mod(n,12)
        if (k.eq.0) k = 12
        t1 = (t0*exp(-1.0/tau) + (1.0 - exp(-1.0/tau)))*temp(k)
        tsoil = (t0 + t1)/2.0
        t0 = t1
      enddo
    end subroutine soil_temp_sub

    !=================================================================
    !=================================================================

    function soil_temp(t0,temp) result(tsoil)
      use types
      use global_par, only: h, tau, diffu
      !implicit none

      real(r_8),intent( in) :: temp
      real(r_8),intent( in) :: t0
      real(r_8) :: tsoil

      real(r_8) :: t1 = 0.0

      t1 = (t0*exp(-1.0/tau) + (1.0 - exp(-1.0/tau)))*temp
      tsoil = (t0 + t1)/2.0
    end function soil_temp

    !=================================================================
    !=================================================================

    function penman (spre,temp,ur,rn,rc2) result(evap)
      use types, only: r_8
      use global_par, only: rcmin, rcmax
      !implicit none


      real(r_8),intent(in) :: spre                 !Surface pressure (mbar)
      real(r_8),intent(in) :: temp                 !Temperature (°C)
      real(r_8),intent(in) :: ur                   !Relative humidity (0-1)
      real(r_8),intent(in) :: rn                   !Radiation balance (W/m2)
      real(r_8),intent(in) :: rc2                  !Canopy resistance (s/m)

      real(r_8) :: evap                            !Evapotranspiration (mm/day)
      !     Parameters
      !     ----------
      real(r_8) :: ra, h5, t1, t2, es, es1, es2, delta_e, delta
      real(r_8) :: gama, gama2


      ra = rcmin
      h5 = 0.0275               !mb-1

      !     Delta
      !     -----
      t1 = temp + 1.
      t2 = temp - 1.
      es1 = wtt(t1)       !Saturation partial pressure of water vapour at temperature T
      es2 = wtt(t2)

      delta = (es1-es2)/(t1-t2) !mbar/oC
      !
      !     Delta_e
      !     -------
      es = wtt (temp)
      delta_e = es*(1. - ur)    !mbar

      if ((delta_e.ge.(1./h5)-0.5).or.(rc2.ge.rcmax)) evap = 0.
      if ((delta_e.lt.(1./h5)-0.5).or.(rc2.lt.rcmax)) then
         !     Gama and gama2
         !     --------------
         gama  = spre*(1004.)/(2.45e6*0.622) ! Psycometric constant
         gama2 = gama*(ra + rc2)/ra ! Incorporate resistance

         !     Real evapotranspiration
         !     -----------------------
         ! LH
         evap = (delta* rn + (1.20*1004./ra)*delta_e)/(delta+gama2) !W/m2
         ! H2O MASS
         evap = evap*(86400./2.45e6) !mm/day
         evap = max(evap,0.)  !Eliminates condensation
      endif
    end function penman

    !=================================================================
    !=================================================================

    function available_energy(temp) result(ae)
      use types, only: r_8
      !implicit none

      real(r_8),intent(in) :: temp
      real(r_8) :: ae

      ae = 2.895 * temp + 52.326 !from NCEP-NCAR Reanalysis data
    end function  available_energy

    !=================================================================
    !=================================================================

    function runoff(wa) result(roff)
      use types, only: r_8
      !implicit none

      real(r_8),intent(in) :: wa
      real(r_8):: roff

      !  roff = 38.*((w/wmax)**11.) ! [Eq. 10]
      roff = 11.5*((wa)**6.6) !from NCEP-NCAR Reanalysis data
    end function  runoff

    !=================================================================
    !=================================================================

    function evpot2 (spre,temp,ur,rn) result(evap)
      use types, only: r_8
      use global_par, only: rcmin, rcmax
      !implicit none

      !Commments from CPTEC-PVM2 code
      !    c Entradas
      !c --------
      !c spre   = pressao aa supeficie (mb)
      !c temp   = temperatura (oC)
      !c ur     = umidade relativa  (0-1,adimensional)
      !c rn     = saldo de radiacao (W m-2)
      !c
      !c Saida
      !c -----
      !c evap  = evapotranspiracao potencial sem estresse (mm/dia)

        !     Inputs

      real(r_8),intent(in) :: spre                 !Surface pressure (mb)
      real(r_8),intent(in) :: temp                 !Temperature (oC)
      real(r_8),intent(in) :: ur                   !Relative humidity (0-1,dimensionless)
      real(r_8),intent(in) :: rn                   !Radiation balance (W/m2)
      !     Output
      !     ------
      !
      real(r_8) :: evap                 !Evapotranspiration (mm/day)
      !     Parameters
      !     ----------
      real(r_8) :: ra, t1, t2, es, es1, es2, delta_e, delta
      real(r_8) :: gama, gama2, rc

      ra = rcmin            !s/m

      !     Delta

      t1 = temp + 1.
      t2 = temp - 1.
      es1 = wtt(t1)
      es2 = wtt(t2)
      delta = (es1-es2)/(t1-t2) !mb/oC (slope of the vpc)

      !     Delta_e
      !     -------

      es = wtt (temp)
      delta_e = es*(1. - ur)    !mb VPD

      !     Stomatal Conductance
      !     --------------------

      rc = rcmin

      !     Gama and gama2
      !     --------------

      gama  = spre*(1004.)/(2.45e6*0.622) ! Psycometric constant
      gama2 = gama*(ra + rc)/ra ! Incorporate resistance

      !     Potencial evapotranspiration (without stress)
      !     ---------------------------------------------

      evap =(delta*rn + (1.20*1004./ra)*delta_e)/(delta+gama2) !W/m2
      evap = evap*(86400./2.45e6) !mm/day
      evap = max(evap,0.)     !Eliminates condensation
    end function evpot2

    !=================================================================
    !=================================================================

  end module water
