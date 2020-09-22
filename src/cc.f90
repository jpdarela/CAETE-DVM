program carbon_costs
   use types
   use global_par
   use utils

   implicit none

   real(r_8), parameter :: kn = 0.15D0    ,&
                         & kp = 0.08D0    ,&
                         & kcn = 0.15D0   ,&
                         & kcp = 0.03D0   ,&
                         & krp = 0.02D0

   real(r_8), dimension(100) :: temps, out
   integer :: j


   call linspace(-100.0D0,600.00D0,temps)

   print*, temps
   do j = 1, 100
      out(j) = cc_fix(temps(j))
   enddo

   open (unit=11,file='cfix.txt',status='replace',form='formatted',access='sequential',&
   action='write')

   write(11,*) out
   close(11)

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
                & e, ccost_upt, to_storage)


      real(r_8), intent(in) :: w    ! Water soil depht Kg(H2O) m-2 == (mm)
      real(r_8), intent(in) :: av_n ! available N (soluble) g m-2
      real(r_8), intent(in) :: av_p ! available P (i soluble) g m-2
      real(r_8), intent(in) :: e    ! Trannspiration (mm/day) == kg(H2O) m-2 day-1
      real(r_8), intent(in) :: nupt, pupt ! g m-2
      real(r_8), dimension(2), intent(out) :: ccost_upt !(N, P)gm⁻² Remaining uptake to be paid if applicable
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
         ccost_upt(1) = 0.0D0
         to_storage(1) = ruptn
      else
         ccost_upt(1) = abs(ruptn)
         to_storage(1) = 0.0D0
      endif

      if(ruptp .ge. 0.0D0) then
         ccost_upt(2) = 0.0D0
         to_storage(2) = ruptp
      else
         ccost_upt(2) = abs(ruptp)
         to_storage(2) = 0.0D0
      endif

   end subroutine passive_uptake

   function fixed_n(c, ts) result(fn)
      real(r_8), intent(in) :: c ! g m-2 day-1 % of NPP destinated to diazotroph partners
      real(r_8), intent(in) :: ts ! soil tem °C
      real(r_8) :: fn

      fn = c * cc_fix(ts)

   end function fixed_n


   ! Costs paid in the next timestep
   function retran_n_cost(littern, resorbed_n) result(c_cost_nr)
      real(r_8), intent(in) :: littern, resorbed_n ! g m-2
      real(r_8), parameter :: krn =  0.025D0
      real(r_8) :: c_cost_nr ! gm-2

      c_cost_nr = resorbed_n * cc_retran(krn, littern)

   end function retran_n_cost



end program carbon_costs
      ! ecp = 1.0D0 - amp
      ! e_vam = e * amp
      ! e_ecm = e * ecp
      ! leafn_vam = leafn * amp
      ! leafn_ecm = leafn * ecp
      ! leafp_vam = leafp * amp
      ! leafp_ecm = leafp * ecp
      ! croot_vam = croot * amp
      ! croot_ecm = croot * ecp
      ! Estimate passive uptake
      ! AM ROOTS
      ! real(r_8) :: ecp, e_vam, e_ecm, leafn_vam,&
               !   & leafn_ecm, leafp_vam, leafp_ecm,&
               !   & croot_vam, croot_ec
