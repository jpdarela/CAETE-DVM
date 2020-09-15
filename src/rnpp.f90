program rnpp
   use types

   real(r_8) :: o1, o2, o3, o4
   logical(l_1) :: l1

   call realized_npp(10.0D0, 0.08D0, 0.04D0, 0.002D0, o1, o2, o3, o4, l1)

   PRINT *, o1, o2, o3, o4, l1


   contains

      subroutine realized_npp(npp_pot, nupt_pot, available_n, pool_n2c,&
                             & upt_out, rnpp, stc, stn, nl)
         implicit none
         real(r_8), intent(in) :: npp_pot     ! POTENTIAL NPP (POOL)
         real(r_8), intent(in) :: nupt_pot    ! POTENTIAL UPTAKE OF NUTRIENT(N/P)
         real(r_8), intent(in) :: available_n ! AVAILABLE NUTRIENTS FOR GROWTH
         real(r_8), intent(in) :: pool_n2c    !

         real(r_8), intent(out) :: upt_out    ! REALIZED UPTAKE
         real(r_8), intent(out) :: rnpp       ! REALIZED NPP
         real(r_8), intent(out) :: stn, stc   ! REALIZED STORAGE - [<n .or. p>; <C>]
         logical(l_1), intent(out) :: nl      ! IS LIMITED

         ! NUTRIENT LIMITED NPP TO(CVEGpool):
         ! LEAF
         if (available_n .ge. nupt_pot) then
         ! THere is NO LIMITATION in this case
            nl = .false.
            ! GROWTH IS ACCOMPLISHED (all npp can go to the CVEG pool)
            rnpp = npp_pot
            ! ESTIMATE UPTAKE
            ! available nutrients are enough to allocate POtenial NPPleaf
            ! and possibly some nutrient for storage
            ! This is the amount of N employed in allocation of NPP(pool) (NUTRIENT UPTAKE)
            upt_out = nupt_pot + (0.25D0 * (available_n - nupt_pot))
            stn = 0.25D0 * (available_n - nupt_pot)
            stc = 0.0D0
         else
            ! NPP OF THIS POOL IS LIMITED BY Nutrient X
            ! In this case the realized NPP for the pool is smaller than the Potential POOL
            nl = .true.
            ! ACOMPLISHED NPP
            rnpp = (available_n * npp_pot) / nupt_pot

            ! REALIZED UPTAKE
            upt_out = rnpp * pool_n2c
            ! REMAINING CARBON IS STORED (and possibly some nutrient)
            stn = max(0.0D0 ,0.25D0 * (available_n - upt_out))
            upt_out = add_pool(upt_out, stn)
            stc = npp_pot - rnpp
            print *, "remaining N", available_n - stn
            print *, "STORED N", stn
         endif
      end subroutine realized_npp


   function add_pool(a1, a2) result(new_amount)

      real(r_8), intent(in) :: a1, a2
      real(r_8) :: new_amount

      if(a2 .ge. 0.0) then
         new_amount = a1 + a2
      else
         new_amount = a1
      endif
   end function add_pool

end program rnpp
