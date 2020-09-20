program carbon_costs
   use types
   use global_par
   implicit none

   real(r_8), parameter :: kn = 0.15D0    ,&
                         & kp = 0.08D0    ,&
                         & kcn = 0.15D0   ,&
                         & kcp = 0.03D0   ,&
                         & krn = 0.025D0  ,&
                         & krp = 0.02D0

   call abrt("adnaskdjfnaslkdjfnalskdjfnadslkjdfnaslkjfdnaskljfnlkasjdnflaksjfn")

   contains
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


   ! subroutine calc_passive_uptk2(nsoil, w, uptk)
   !    ! Adapted for CAETÊ??????
   !    real(r_8), intent(in) :: nsoil  !(Kg Nutrient m-2) Solution Nutrient
   !    real(r_8), intent(in) :: w      ! Water input after hydraulic calcs kg m-2 day-1
   !    real(r_8), intent(out) :: uptk

   !    uptk = nsoil * (w/wmax)      !  !(Kg N m-2 day-1)

   ! end subroutine calc_passive_uptk2

   subroutine abrt(arg1)

      character(*), intent(in) :: arg1
      print *, arg1
      call abort()
   end subroutine abrt

   ! Calculate Carbon costs of Nutrients

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

end program carbon_costs
