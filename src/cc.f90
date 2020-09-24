module carbon_costs
   use types
   use global_par
   use utils

   implicit none
   private

   public :: abrt                   ,&
           & calc_passive_uptk1     ,&
           & cc_active              ,&
           & cc_retran              ,&
           & cc_fix                 ,&
           & passive_uptake         ,&
           & fixed_n                ,&
           & retran_nutri_cost      ,&
           & active_cost            ,&
           & ap_actvity1            ,&
           & ezc_prod               ,&
           & active_nutri_gain      ,&
           & n_invest_p             ,&
           & select_active_strategy

   ! real(r_8), dimension(100) :: temps, out
   ! integer :: j


   ! call linspace(-100.0D0,600.00D0,temps)

   ! print*, temps
   ! do j = 1, 100
   !    out(j) = cc_fix(temps(j))
   ! enddo

   ! open (unit=11,file='cfix.txt',status='replace',form='formatted',access='sequential',&
   ! action='write')

   ! write(11,*) out
   ! close(11)

   contains
   ! HElPERS
   subroutine abrt(arg1)

      character(*), intent(in) :: arg1
      print *, arg1
      call abort()
   end subroutine abrt


   ! CONVERSIONS -- LOOK eq 150 onwards in Reis 2020
   ! Calculations of passive uptake
   subroutine calc_passive_uptk1(nsoil, et, sd, uptk)
      ! From Fisher et al. 2010
      real(r_8), intent(in) :: nsoil  ! (ML⁻²)
      real(r_8), intent(in) :: et ! Transpiration (ML⁻²T⁻¹)
      real(r_8), intent(in) :: sd ! soil water depht (ML⁻²)  ( 1 Kg(H2O) m⁻² == 1 mm )
      real(r_8), intent(out) :: uptk
      ! Mass units of et and sd must be the same
      uptk = nsoil * (et/sd)      !  !(ML⁻²T⁻¹)

   end subroutine calc_passive_uptk1


   function cc_active(k1, d1, k2, d2) result (c_active)

      real(r_8), intent(in) :: k1 ! CONSTANT
      real(r_8), intent(in) :: d1 ! AVAILABLE NUTRIENT POOL Kg Nutrient m-2
      real(r_8), intent(in) :: k2 ! CONSTANT
      real(r_8), intent(in) :: d2 ! CARBON IN ROOTS kg C m-2

      real(r_8) :: c_active   ! g(C) g(Nutrient)⁻¹

      if(d1 .le. 0.0D0) call abrt("division by 0 in cc_acive - d1")
      if(d2 .le. 0.0D0) call abrt("division by 0 in cc_acive - d2")

      c_active = (k1 / d1) * (k2 / d2)

   end function cc_active


   function cc_retran(k1, d1) result(c_retran)

      real(r_8), intent(in) :: k1 ! Constant
      real(r_8), intent(in) :: d1 ! Nutrient in litter ML-2
      real(r_8) :: c_retran       ! Carbon cost of nut. resorbed MC MN-1

      if(d1 .le. 0.0D0) call abrt("division by 0 in cc_acive - d1")
      c_retran = k1/d1

   end function cc_retran


   function cc_fix(ts) result (c_fix)

      real(r_8), intent(in) :: ts ! soil temp °C
      real(r_8) :: c_fix ! C Cost of Nitrogen M(N) M(C)⁻¹
      c_fix = -6.25D0 * (exp(-3.62D0 + (0.27D0 * ts) &
      &       * (1.0D0 - (0.5D0 * (ts / 25.15D0)))) - 2.0D0)

   end function cc_fix


   ! ESTIMATE PASSIVE UPTAKE OF NUTRIENTS
   subroutine passive_uptake (w, av_n, av_p, nupt, pupt,&
                & e, topay_upt, to_storage)


      real(r_8), intent(in) :: w    ! Water soil depht Kg(H2O) m-2 == (mm)
      real(r_8), intent(in) :: av_n ! available N (soluble) g m-2
      real(r_8), intent(in) :: av_p ! available P (i soluble) g m-2
      real(r_8), intent(in) :: e    ! Trannspiration (mm/day) == kg(H2O) m-2 day-1
      real(r_8), intent(in) :: nupt, pupt ! g m-2
      real(r_8), dimension(2), intent(out) :: topay_upt !(N, P)gm⁻² Remaining uptake to be paid if applicable
      real(r_8), dimension(2), intent(out) :: to_storage!(N, P)gm⁻² Passively uptaken nutrient if applicable

      real(r_8) :: passive_n,&
                 & passive_p,&
                 & ruptn,&
                 & ruptp

      call calc_passive_uptk1(av_n, e, w, passive_n)
      call calc_passive_uptk1(av_p, e, w, passive_p)

      ruptn = passive_n - nupt
      ruptp = passive_p - pupt

      if(ruptn .ge. 0.0D0) then
         topay_upt(1) = 0.0D0
         to_storage(1) = ruptn
      else
         topay_upt(1) = abs(ruptn)
         to_storage(1) = 0.0D0
      endif

      if(ruptp .ge. 0.0D0) then
         topay_upt(2) = 0.0D0
         to_storage(2) = ruptp
      else
         topay_upt(2) = abs(ruptp)
         to_storage(2) = 0.0D0
      endif

   end subroutine passive_uptake


   function fixed_n(c, ts) result(fn)
      ! Given the C available calculate the fixed N
      real(r_8), intent(in) :: c ! g m-2 day-1 % of NPP destinated to diazotroph partners
      real(r_8), intent(in) :: ts ! soil tem °C
      real(r_8) :: fn

      fn = c * cc_fix(ts)

   end function fixed_n


   function retran_nutri_cost(littern, resorbed_n, nut) result(c_cost_nr)
      ! Calculate the costs of realized resorption
      ! ! Costs paid in the succeding timestep
      real(r_8), intent(in) :: littern ! Nutrient in litter g m-2
      real(r_8), intent(in) :: resorbed_n ! Resorbed Nutrient g m-2
      integer(i_4), intent(in) :: nut     ! 1 for N else P
      real(r_8), parameter :: krn =  0.025D0, krp = 0.02D0
      real(r_8) :: c_cost_nr ! gm-2

      if(nut .eq. 1) then ! Nitrogen
         c_cost_nr = resorbed_n * cc_retran(krn, littern)
      else ! Phosphotus
         c_cost_nr = resorbed_n * cc_retran(krp, littern)
      endif

   end function retran_nutri_cost


   subroutine active_cost(amp, av_n, av_p, croot, cc)

      real(r_8), intent(in) :: amp
      real(r_8), intent(in) :: av_n, av_p
      real(r_8), intent(in) :: croot
      real(r_8), dimension(2,4), intent(out) :: cc ! [Nccnma, Nccnme, Nccam, Nccem]
                                                   ! [Pccnma, Pccnme, Pccam, Pccem]

      real(r_8), parameter :: kn = 0.15D0    ,&
                            & kp = 0.08D0    ,&
                            & kcn = 0.15D0   ,&
                            & kcp = 0.03D0   ,& ! PArameters from FUN3.0 source code
                            & kan = 0.0D0    ,& ! AkN<-0.0
                            & ken = 0.025D0  ,& ! EkN<-0.025
                            & kanc = 0.025D0 ,& ! AkC<-0.025
                            & kenc = 0.15D0  ,& ! EkC<-0.15
                            & kap = 0.1D0    ,& ! AkP<-0.1 #AM cost
                            & kapc = 0.5D0   ,& ! AkCp<-0.5 #AM cost
                            & kep = 0.05D0   ,& ! EkP<-0.05 #ECM cost
                            & kepc = 1.0D0      ! EkCp<-1.0 #ECM cost

      integer(i_4), parameter :: N = 1, P = 2, nma = 1, nme = 2, am = 3, em = 4

      real(r_8) :: ecp, av_n_vam, av_n_ecm, av_p_vam, av_p_ecm

      ecp = 1.0D0 - amp

      av_n_vam = av_n * amp
      av_n_ecm = av_n * ecp
      av_p_vam = av_p * amp
      av_p_ecm = av_p * ecp

      ! Costs of active Non Mycorrhizal uptake of Nitrogen
      cc(N,nma) = cc_active(kn, av_n_vam, kcn, croot)
      cc(N,nme) = cc_active(kn, av_n_ecm, kcn, croot)

      ! Costs of active Mycorrhizal uptake of Nitrogen
      cc(N,am) = 1D20! cc_active(kan, av_n_vam, kanc, croot)
      cc(N,em) = cc_active(ken, av_n_ecm, kenc, croot)

      !Costs of active Non Mycorrhizal uptake of P
      cc(P,nma) = cc_active(kp, av_p_vam, kcp, croot)
      cc(P,nme) = cc_active(kp, av_p_ecm, kcp, croot)

      !Costs of active Mycorrhizal uptake of P
      cc(P,am) = cc_active(kap, av_p_vam, kapc, croot)
      cc(P,em) = cc_active(kep, av_p_ecm, kepc, croot)

   end subroutine active_cost

   subroutine select_active_strategy(amp, av_n, av_p, croot, cc, strategy)
      real(r_8), intent(in) :: amp
      real(r_8), intent(in) :: av_n, av_p
      real(r_8), intent(in) :: croot
      real(r_8), dimension(2), intent(out) :: cc
      integer(i_4), dimension(2), intent(out) :: strategy

      real(r_8), dimension(2,4) :: costs_array
      integer(i_4), parameter :: N = 1, P = 2
      integer(i_4), dimension(1) :: minindex
      real(r_8), dimension(1) :: minvalue
      ! [Nccnma, Nccnme, Nccam, Nccem]
      ! [Pccnma, Pccnme, Pccam, Pccem]
      !

      call active_cost(amp, av_n, av_p, croot, costs_array)

      minvalue = minval(costs_array(N, :))
      minindex = minloc(costs_array(N, :))

      cc(N) = minvalue(1)
      strategy(N) = minindex(1)

      minvalue = minval(costs_array(P, :))
      minindex = minloc(costs_array(P, :))

      cc(P) = minvalue(1)
      strategy(P) = minindex(1)

   end subroutine select_active_strategy


   subroutine ap_actvity1(c_xm, strat, cc_array, ezc_ap)
      real(r_8), intent(in) :: c_xm ! g m-2 C expended on P uptake
      integer(i_4), intent(in) :: strat ! index for the active costs array
      real(r_8), dimension(2,4), intent(in) :: cc_array
      real(r_8), intent(out) :: ezc_ap   ! Carbon that will be
                                         ! converted in enzymes
      ! Calculate how much C is allocated to produce
      ! enzime by an strategy given the amount of c_xm(g(C)m-2)
      integer(i_4), parameter :: P = 2

      ezc_ap = c_xm / cc_array(P, strat)

   end subroutine ap_actvity1


   subroutine ezc_prod(c_ezc, strat, cc_array, enzyme_conc)

      ! Calculate the enzyme concentration for
      ! a given nutri for a given strategy.
      ! g m-2

      real(r_8), intent(in) :: c_ezc ! g m-2 C expended on Nx uptake
      integer(i_4), intent(in) :: strat ! index for the active costs array
      real(r_8), dimension(2,4), intent(in) :: cc_array
      real(r_8), intent(out) :: enzyme_conc

      real(r_8), parameter :: c_to_enzyme = 2.0D0
      integer(i_4), parameter :: P = 2

      enzyme_conc = c_ezc * cc_array(P, strat) * c_to_enzyme ! g m-2

   end subroutine ezc_prod


   function active_nutri_gain(enzyme_conc) result(nutri_out)
      ! Calculate the amount of nutrient that is
      real(r_8), intent(in) :: enzyme_conc
      real(r_8) :: nutri_out
      real(r_8), parameter :: vcmax = 1800.0D0 ! mol(P) g[enzyme]-1 day-1

      nutri_out = vcmax * enzyme_conc  ! mol NUtrient/m2/day

   end function active_nutri_gain

   function n_invest_p(c_ezc) result(nmass)
      ! Calculate the invested n in AP activity. N added to N demand

      real(r_8), intent(in) :: c_ezc ! enzme mass g m-2
      real(r_8) :: nmass ! gm -2
      real(r_8), parameter :: n_enzime = 0.16D0

      nmass = c_ezc * n_enzime
   end function n_invest_p

end module carbon_costs
