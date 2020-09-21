program carbon_costs
   use types
   use global_par
   use utils

   implicit none

   real(r_8), parameter :: kn = 0.15D0    ,&
                         & kp = 0.08D0    ,&
                         & kcn = 0.15D0   ,&
                         & kcp = 0.03D0   ,&
                         & krn = 0.025D0  ,&
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
      real(r_8), intent(in) :: nsoil  ! (Kg Nutrient m-2)
      real(r_8), intent(in) :: et ! Transpiration (m s-1)
      real(r_8), intent(in) :: sd ! soil water depht (m) (same of soil water content (1000 * Kg m-2))
      real(r_8), intent(out) :: uptk

      uptk = nsoil * (et/sd)      !  !(Kg N m-2 s-1)

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

      real(r_8), intent(in) :: k1
      real(r_8), intent(in) :: d1 ! Nutrient in litter
      real(r_8) :: c_retran

      if(d1 .le. 0.0D0) call abrt("division by 0 in cc_acive - d1")
      c_retran = k1/d1

   end function cc_retran


   function cc_fix(ts) result (c_fix)

      real(r_8), intent(in) :: ts
      real(r_8) :: c_fix

      c_fix = -6.25D0 * (exp(-3.62D0 + (0.27D0 * ts) &
      &       * (1.0D0 - (0.5D0 * (ts / 25.15D0)))) - 2.0D0)

   end function cc_fix

   subroutine cc (w, av_n, av_p, nupt, pupt, leafn, leafp,&
                & croot, amp, e, ccost)


      real(r_8), intent(in) :: w, av_n, av_p, leafn, leafp
      real(r_8), intent(in) :: croot, e, amp
      real(r_8), intent(in) :: nupt, pupt
      real(r_8), dimension(2), intent(out) :: ccost
      real(r_8) :: ecp, e_vam, e_ecm, leafn_vam,&
                 & leafn_ecm, leafp_vam, leafp_ecm,&
                 & croot_vam, croot_ecm
      real(r_8) :: passive_n_vam, passive_p_vam,&
                 & passive_n_ecm, passive_p_ecm
      real(r_8), dimension(2) :: to_storage

      ecp = 1.0D0 - amp
      e_vam = e * amp
      e_ecm = e * ecp
      leafn_vam = leafn * amp
      leafn_ecm = leafn * ecp
      leafp_vam = leafp * amp
      leafp_ecm = leafp * ecp
      croot_vam = croot * amp
      croot_ecm = croot * ecp
      ! Estimate passive uptake
      ! AM ROOTS

      call calc_passive_uptk1(av_n, e_vam, w, passive_n_vam)
      call calc_passive_uptk1(av_n, e_ecm, w, passive_n_ecm)
      call calc_passive_uptk1(av_p, e_vam, w, passive_p_vam)
      call calc_passive_uptk1(av_p, e_ecm, w, passive_p_ecm)

      if((passive_n_ecm + passive_n_vam) .gt. nupt) then
         ccost(1) = 0.0D0
         to_storage(1) = (passive_n_ecm + passive_n_vam) - nupt
      endif

      if((passive_p_ecm + passive_p_vam) .gt. pupt) then
         ccost(2) = 0.0D0
         to_storage(2) = (passive_p_ecm + passive_p_vam) - pupt
      endif

   end subroutine cc



end program carbon_costs
