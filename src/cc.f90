module carbon_costs
   use types
   use global_par
   use utils

   implicit none
   private

   public :: abrt                   ,&
           & calc_passive_uptk1     ,&
           & passive_uptake         ,&
           & cc_active              ,&
           & active_cost            ,&
           & active_costN           ,&
           & active_costP           ,&
           & cc_fix                 ,&
           & fixed_n                ,&
           & cc_retran              ,&
           & retran_nutri_cost      ,&
           & ap_actvity1            ,&
           & ezc_prod               ,&
           & active_nutri_gain      ,&
           & n_invest_p             ,&
           & select_active_strategy ,&
           & prep_out_n             ,&
           & prep_out_p

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
      if (uptk .gt. (0.5D0 * nsoil)) uptk = 0.5D0 * nsoil
   end subroutine calc_passive_uptk1

   ! ESTIMATE PASSIVE UPTAKE OF NUTRIENTS
   subroutine passive_uptake (w, av_n, av_p, nupt, pupt,&
      & e, topay_upt, to_storage, passive_upt)
      real(r_8), intent(in) :: w    ! Water soil depht Kg(H2O) m-2 == (mm)
      real(r_8), intent(in) :: av_n ! available N (soluble) g m-2
      real(r_8), intent(in) :: av_p ! available P (i soluble) g m-2
      real(r_8), intent(in) :: e    ! Trannspiration (mm/day) == kg(H2O) m-2 day-1
      real(r_8), intent(in) :: nupt, pupt ! g m-2
      real(r_8), dimension(2), intent(out) :: topay_upt !(N, P)gm⁻² Remaining uptake to be paid if applicable
      real(r_8), dimension(2), intent(out) :: to_storage!(N, P)gm⁻² Passively uptaken nutrient if applicable
      real(r_8), dimension(2), intent(out) :: passive_upt
      real(r_8) :: passive_n,&
            & passive_p,&
            & ruptn,&
            & ruptp

      call calc_passive_uptk1(av_n, e, w, passive_n)
      call calc_passive_uptk1(av_p, e, w, passive_p)

      passive_upt(1) = passive_n
      passive_upt(2) = passive_p

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


   function cc_active(k1, d1, k2, d2) result (c_active)
      real(r_8), intent(in) :: k1 ! CONSTANT
      real(r_8), intent(in) :: d1 ! AVAILABLE NUTRIENT POOL Kg Nutrient m-2
      real(r_8), intent(in) :: k2 ! CONSTANT
      real(r_8), intent(in) :: d2 ! CARBON IN ROOTS kg C m-2

      real(r_8) :: c_active   ! g(C) g(Nutrient)⁻¹

      if(d1 .le. 0.0D0) call abrt("division by 0 in cc_active - d1")
      if(d2 .le. 0.0D0) call abrt("division by 0 in cc_active - d2")

      c_active = (k1 / d1) * (k2 / d2)
   end function cc_active


   subroutine active_cost(amp, av_n, av_p, croot, cc)
      ! Standard Model FUN3.0
      real(r_8), intent(in) :: amp
      real(r_8), intent(in) :: av_n, av_p
      real(r_8), intent(in) :: croot
      real(r_8), dimension(2,4), intent(out) :: cc ! [Nccnma, Nccnme, Nccam, Nccem]
                                                   ! [Pccnma, Pccnme, Pccam, Pccem]

      real(r_8), parameter :: kn   = 0.5D0         ,& ! change parameter s to reflect Fisher et al 2010
                            & kcn  = 1.0D0 / kn    ,& ! assertion that the product of kc and kx == 1
                            & kp   = 0.7D0         ,&
                            & kcp  = 1.0D0 / kp    ,& ! PArameters from FUN3.0 source code
                            & kan  = 1.0D0         ,& ! AkN<-0.0?
                            & kanc = 1.0D0         ,& ! AkC<-0.025
                            & ken  = 0.15D0        ,& ! EkN<-0.025
                            & kenc = 0.75D0 / ken   ,& ! EkC<-0.15
                            & kap  = 0.8D0        ,& ! AkP<-0.1 #AM cost
                            & kapc = 1.0D0 / kap   ,& ! AkCp<-0.5 #AM cost
                            & kep  = 0.3D0         ,& ! EkP<-0.05 #ECM cost
                            & kepc = 1.15D0 / kep      ! EkCp<-1.0 #ECM cost

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
      cc(N,am) = cc_active(kan, av_n_vam, kanc, croot)
      cc(N,em) = cc_active(ken, av_n_ecm, kenc, croot)

      !Costs of active Non Mycorrhizal uptake of P
      cc(P,nma) = cc_active(kp, av_p_vam, kcp, croot)
      cc(P,nme) = cc_active(kp, av_p_ecm, kcp, croot)

      !Costs of active Mycorrhizal uptake of P
      cc(P,am) = cc_active(kap, av_p_vam, kapc, croot)
      cc(P,em) = cc_active(kep, av_p_ecm, kepc, croot)
   end subroutine active_cost


   subroutine active_costn(amp, av_n, on, croot, ccn)
      ! Adapted model - FUN-POOL
      real(r_8), intent(in) :: amp
      real(r_8), intent(in) :: av_n, on
      real(r_8), intent(in) :: croot
      real(r_8), dimension(6), intent(out) :: ccn

      real(r_8), parameter :: kn   = 0.5D0  ,& !
                            & kcn  = 1.0D0  ,& !
                            & kan  = 0.9D0  ,& !
                            & kanc = 1.4D0  ,& !
                            & ken  = 0.15D0 ,& !
                            & kenc = 0.75D0

      integer(i_4), parameter :: N = 1

      integer(i_4), parameter :: nma = 1  ,&  ! ROOT AM  active uptake in N soluble inorganic (NSI)
                               & nme = 2  ,&  ! ROOT EM  active uptake in NSI
                               & am  = 3  ,&  ! AM in NSI - active uptake via hyphae surface
                               & em  = 4  ,&  ! EM in NSI - ...
                               & AM0 = 5  ,&  ! AM Nitrogenase activity on organic N ? NO
                               & em0 = 6      ! EM Nitrogenase activity on organic Nitrogen

      real(r_8) :: ecm

      ecm = 1.0D0 - amp

      ccn(nma) = cc_active(kn, amp * av_n, kcn, amp * croot)
      ccn(nme) = cc_active(kn, ecm * av_n, kcn, ecm * croot)
      ccn(am) = cc_active(kan, amp * av_n, kanc, amp * croot)
      ccn(em) = cc_active(ken, ecm * av_n, kenc, ecm * croot)
      ccn(Am0) = cc_active(kan, amp * on, kenc, amp * croot)
      ccn(em0) = cc_active(ken, ecm * on, kenc, ecm * croot)
   end subroutine active_costn


   subroutine active_costp(amp, av_p, sop, op, croot, ccp)
      ! Adapted model - FUN-POOL
      real(r_8), intent(in) :: amp
      real(r_8), intent(in) :: av_p, sop, op
      real(r_8), intent(in) :: croot
      real(r_8), dimension(8), intent(out) :: ccp

      !['nmam', 'nmem', 'am', 'em', 'ramAP', 'remAP', 'AMAP', 'EM0']

      real(r_8), parameter :: kp   = 0.7D0  ,&
                            & kcp  = 1.0D0  ,& ! PArameters from FUN3.0 source code (modified)
                            & kap  = 0.8D0  ,& ! AkP<-0.1 #AM cost
                            & kapc = 1.2D0  ,& ! AkCp<-0.5 #AM cost
                            & kep  = 0.9D0  ,& ! EkP<-0.05 #ECM cost
                            & kepc = 1.4D0   ! EkCp<-1.0 #ECM cost

      ! Strategies of P aquisition - ACTIVE UPTAKE
      integer(i_4), parameter :: nma   = 1 ,& ! Non Myco. AM
                              &  nme   = 2 ,& ! Non Myco. EM
                              &  am    = 3 ,& ! AM active
                              &  em    = 4 ,& ! EM active
                              &  ramAP = 5 ,& ! ROOT Non Myco. AM AP activity
                              &  remAP = 6 ,& ! ROOT Non Myco. EM AP activity
                              &  AMAP  = 7 ,& ! AM AP Activity
                              &  EM0x  = 8    ! EM exudate Activity

      real(r_8) :: ecm

      ecm = 1.0D0 - amp

      ! Costs of active Non Mycorrhizal uptake of P
      ccp(nma) = cc_active(kp, amp * av_p, kcp, amp * croot)
      ccp(nme) = cc_active(kp, ecm * av_p, kcp, ecm * croot)

      ! ! Costs of active Mycorrhizal uptake of P
      ccp(am) = cc_active(kap, amp * av_p, kapc, amp * croot)
      ccp(em) = cc_active(kep, ecm * av_p, kepc, ecm * croot)

      ! !Costs of active Non Mycorrhizal AP activity
      ccp(ramAP) = cc_active(kap, amp * op, kapc, amp * croot)
      ccp(remAP) = cc_active(kep, ecm * op, kepc, ecm * croot)

      ! !Costs of Mycorrhizal AP/exudates
      ccp(AMAP) = cc_active(kap, amp * op , kapc, amp * croot)
      ccp(EM0x) = cc_active(kep, ecm * sop, kepc, ecm * croot)
   end subroutine active_costp


   function cc_fix(ts) result (c_fix)
      real(r_8), intent(in) :: ts ! soil temp °C
      real(r_8) :: c_fix ! C Cost of Nitrogen M(N) M(C)⁻¹
      c_fix = -6.25D0 * (exp(-3.62D0 + (0.27D0 * ts) &
      &       * (1.0D0 - (0.5D0 * (ts / 25.15D0)))) - 2.0D0)
   end function cc_fix


   function fixed_n(c, ts) result(fn)
      ! Given the C available calculate the fixed N
      real(r_8), intent(in) :: c ! g m-2 day-1 % of NPP destinated to diazotroph partners
      real(r_8), intent(in) :: ts ! soil tem °C
      real(r_8) :: fn

      fn = c * cc_fix(ts)
   end function fixed_n


   function cc_retran(k1, d1) result(c_retran)
      real(r_8), intent(in) :: k1 ! Constant
      real(r_8), intent(in) :: d1 ! Nutrient in litter ML-2
      real(r_8) :: c_retran       ! Carbon cost of nut. resorbed MC MN-1

      if(d1 .le. 0.0D0) call abrt("division by 0 in cc_retran - d1")
      c_retran = k1/d1
   end function cc_retran


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


   subroutine select_active_strategy(cc_array, cc, strategy)

      real(r_8), dimension(:),intent(in) :: cc_array
      real(r_8), intent(out) :: cc
      integer(i_4), intent(out) :: strategy
      real(r_8) :: cc_strat
      integer(i_4) :: cc_size, j
      logical(l_1) :: test1, test2
      cc_size = size(cc_array)
      cc_strat = 1D17

      do j = cc_size, 1, -1
         test1 = cc_array(j) .lt. cc_strat
         if(test1) then
            cc_strat = cc_array(j)
            strategy = j
         endif

         test2 = abs(cc_array(j) - cc_strat) .lt. 1D-8
         if(test2) then
            cc_strat = cc_array(j)
            strategy = j
         endif

      enddo
      cc = cc_strat
   end subroutine select_active_strategy

   subroutine prep_out_n(nut_aqui_strat, nupt, to_pay, out_array)
      integer(i_4),intent(in) :: nut_aqui_strat
      real(r_8), intent(in) :: to_pay, nupt
      real(r_8), dimension(2), intent(out) :: out_array

      integer(i_4), parameter :: avail_n = 1, on = 2

      select case (nut_aqui_strat)
         case(1,2,3,4)
            out_array(avail_n) = nupt
            out_array(on) = 0.0D0
         case(5,6)
            out_array(avail_n) = nupt - to_pay
            out_array(on) = to_pay
         case default
            call abrt("Problem in N output case default - cc.f90 325")
      end select
      ! Soluble inorg_n_pool = (1, 2, 3, 4)
      ! Organic N pool = (5, 6)
      ! Soluble inorg_p_pool = (1, 2, 3, 4)
      ! Organic P pool = (5, 6, 7)
      ! Insoluble inorg p pool = (8)
   end subroutine prep_out_n


   subroutine prep_out_p(nut_aqui_strat, pupt, to_pay, out_array)
      integer(i_4),intent(in) :: nut_aqui_strat
      real(r_8), intent(in) :: to_pay, pupt
      real(r_8), dimension(3), intent(out) :: out_array
      integer(i_4), parameter :: avail_p = 1, sop = 2, op=3

      select case (nut_aqui_strat)
         case (1,2,3,4)
            out_array(avail_p) = pupt
            out_array(sop) = 0.0D0
            out_array(op) = 0.0D0
         case (5, 6, 7)
            out_array(avail_p) = pupt - to_pay
            out_array(sop) = 0.0D0
            out_array(op) = to_pay
         case (8)
            out_array(avail_p) = pupt - to_pay
            out_array(sop) = to_pay
            out_array(op) = 0.0D0
         case default
            call abrt("Problem in P output case default - cc.f90 362")
      end select
      ! Soluble inorg_n_pool = (1, 2, 3, 4)
      ! Organic N pool = (5, 6)
      ! Soluble inorg_p_pool = (1, 2, 3, 4)
      ! Organic P pool = (5, 6, 7)
      ! Insoluble inorg p pool = (8)
   end subroutine prep_out_p


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
      real(r_8), parameter :: vcmax = 0.003D0 ! mol(P) g[enzyme]-1 day-1

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
