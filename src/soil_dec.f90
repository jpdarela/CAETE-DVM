! Copyright 2017- LabTerra

!     This program is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.)

!     This program is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.

!     You should have received a copy of the GNU General Public License
!     along with this program.  If not, see <http://www.gnu.org/licenses/>.

! AUTHOR: JP Darela

module soil_dec

   use types
   use global_par
   implicit none

   !=========================================================================
   ! FUNCTIONS AND SUBROUTINES DEFINED IN SOIL_DEC MODULE
   public :: carbon3          ! Subroutine that calculates the C:N:P decay dynamics
   public :: carbon_decay     ! Carbon decay function in response to Temperarute
   public :: water_effect     ! Soil water content effect on C decay
   public :: sorbed_p_equil   ! Fucntion that caculates the equilibrium between Mineralized P and Sorbed P
   public :: solution_p_equil
   public :: sorbed_n_equil   ! Fucntion that caculates the equilibrium between Mineralized N and Sorbed N
   public :: solution_n_equil
   public :: leaching
   public :: add_pool

contains

   subroutine carbon3(tsoil, water_sat, leaf_litter, coarse_wd,&
                    &        root_litter, lnc, cs, snc_in,&
                    &        cs_out, snc, hr, nmin, pmin)

      real(r_8),parameter :: clit_atm = 0.7D0
      real(r_8),parameter :: cwd_atm = 0.22D0

      ! POOLS OF LITTER AND SOIL

      integer(i_4) :: index

      !     Inputs
      !     ------
      real(r_4),intent(in) :: tsoil, water_sat   ! soil temperature (°C); soil water relative content (dimensionless)

      real(r_8),intent(in) :: leaf_litter  ! Mass of C comming from living pools g(C)m⁻²
      real(r_8),intent(in) :: coarse_wd
      real(r_8),intent(in) :: root_litter
      real(r_8),dimension(6),intent(in) :: lnc   ! g(Nutrient)m⁻2 Incoming Nutrient in litter

      real(r_8),dimension(4),intent(in) :: cs   ! Soil carbon (gC/m2)   State Variable -> The size of the carbon pools
      real(r_8),dimension(8), intent(in) :: snc_in   ! Current soil nutrient mass

      !     Output
      real(r_8),dimension(4), intent(out) :: cs_out       ! State Variable -> The size of the carbon pools
      real(r_8),dimension(8), intent(out) :: snc          ! Updated Soil pools Nutrient to C ratios
      real(r_8),intent(out) :: hr                         ! Heterotrophic (microbial) respiration (gC/m2/day)
      real(r_8),intent(out) :: nmin, pmin


      !Auxiliary variables
      real(r_8),dimension(4) :: nmass_org = 0.0 ! Mass of nutrients in ORGANIC POOLS
      real(r_8),dimension(4) :: pmass_org = 0.0
      real(r_8),dimension(4) :: het_resp, cdec
      real(r_8) :: leaf_n
      real(r_8) :: froot_n
      real(r_8) :: wood_n
      real(r_8) :: leaf_p
      real(r_8) :: froot_p
      real(r_8) :: wood_p
      real(r_8),dimension(8) :: snr_in

      real(r_4) :: water_modifier ! Multiplicator for water influence on C decay
      real(r_8) :: frac1,frac2    ! Constants for litter partitioning
      real(r_8) :: c_next_pool, n_next_pool, p_next_pool
      real(r_8) :: n_min_resp_lit, p_min_resp_lit
      real(r_8) :: incomming_c_lit, incomming_n_lit, incomming_p_lit
      real(r_8) :: update_c, update_n, update_p
      real(r_8) :: leaf_l, cwd, root_l ! Mass of C comming from living pools g(C)m⁻²

      ! Turnover Rates  == residence_time⁻¹ (years⁻¹)
      real(r_8), dimension(4) :: tr_c

      tr_c(1) = 25.0D0
      tr_c(2) = 250.0D0
      tr_c(3) = 2000.0D0
      tr_c(4) = 5000.0D0

      snc(:) = 0.0D0
      cs_out(:) = 0.0D0
      hr = 0.0D0
      nmin = 0.0D0
      pmin = 0.0D0


      frac1 = 0.55D0
      frac2 = 1.0D0 - frac1


!     ! CARBON AND NUTRIENTS COMMING FROM VEGETATION
      leaf_l = leaf_litter
      cwd    = coarse_wd
      root_l = root_litter

      ! Litter Quality  (pool/Nutrient structure):
      ! [ 1(l1n),2(l2n),3(c1dn),4(c2n),5(l1p),6(l2p),7(c1p),8(c2p)]

      ! find nutrient mass/area) : litter fluxes[ML⁻²] * litter nutrient ratios
      ! (lnr) [MM⁻¹]
      leaf_n  = lnc(1) ! g(nutrient) m-2
      froot_n = lnc(2)
      wood_n  = lnc(3)
      leaf_p  = lnc(4)
      froot_p = lnc(5)
      wood_p  = lnc(6)


      ! C:N:P CYCLING NUMERICAL SOLUTION
      ! ORGANIC NUTRIENTS in SOIL

      ! Soil Nutrient ratios and organic nutrients g m-2
      nmass_org(1) = snc_in(1) !                                      ! g(N)m-2
      nmass_org(2) = snc_in(2) !
      nmass_org(3) = snc_in(3) !
      nmass_org(4) = snc_in(4) !
      pmass_org(1) = snc_in(5) !                                      ! g(P)m-2
      pmass_org(2) = snc_in(6) !
      pmass_org(3) = snc_in(7) !
      pmass_org(4) = snc_in(8) !

      snr_in(1) = snc_in(1) / cs(1)
      snr_in(2) = snc_in(2) / cs(2)
      snr_in(3) = snc_in(3) / cs(3)
      snr_in(4) = snc_in(4) / cs(4)
      snr_in(5) = snc_in(5) / cs(1)
      snr_in(6) = snc_in(6) / cs(2)
      snr_in(7) = snc_in(7) / cs(3)
      snr_in(8) = snc_in(8) / cs(4)

      ! CARBON DECAY
      water_modifier = water_effect(water_sat)
      do index = 1,4
            cdec(index) = carbon_decay(q10,tsoil,cs(index),tr_c(index)) * water_modifier
      enddo

      !LITTER I
      cs_out(1) = cs(1) - cdec(1)        ! Update Litter Carbon 1

      ! Mineralization
      ! C
      ! Release as CO2
      het_resp(1) = cdec(1) * clit_atm        ! Heterotrophic respiration ! processed (dacayed) Carbon lost to ATM

      ! Carbon going to LITTER 2
      c_next_pool = cdec(1) - het_resp(1)         ! Carbon going to cl_out(2) (next pool)

      ! N
      ! N mineralized by the release of CO2
      n_min_resp_lit = het_resp(1) * snr_in(1)
      p_min_resp_lit = het_resp(1) * snr_in(5)

      ! N going to the LITTER II
      n_next_pool = c_next_pool * snr_in(1)
      p_next_pool = c_next_pool * snr_in(5)

      ! UPDATE N in Organic MAtter Litter I
      nmass_org(1) = nmass_org(1) - (n_min_resp_lit + n_next_pool)
      pmass_org(1) = pmass_org(1) - (p_min_resp_lit + p_next_pool)

      ! GET MINERAIZED N
      nmin = add_pool(nmin, n_min_resp_lit)
      pmin = add_pool(pmin, p_min_resp_lit)

      n_min_resp_lit = 0.0
      p_min_resp_lit = 0.0

      ! END OF MINERALIZATION PROCESS (LITTER 1)

      ! UPDATE CNP ORGANIC POOLS
      incomming_c_lit = (frac1 * leaf_l) + (frac1 * root_l) + (frac1 * cwd)                  ! INcoming Carbon from vegetation g m-2
      incomming_n_lit = (frac1 * leaf_n) + (frac1 * froot_n) + (frac1 * wood_n)                 ! Iincoming N
      incomming_p_lit = (frac1 * leaf_p) + (frac1 * froot_p) + (frac1 * wood_p)                 ! Incoming P

      cs_out(1) = add_pool(cs_out(1), incomming_c_lit)
      nmass_org(1) = add_pool(nmass_org(1), incomming_n_lit)
      pmass_org(1) = add_pool(pmass_org(1), incomming_p_lit)

      ! UPDATE Soil Nutrient Content g m-2
      snc(1) = nmass_org(1)
      snc(5) = pmass_org(1)
      ! END LITTER 1

      ! CLEAN AUX VARIABLES
      incomming_c_lit = 0.0
      incomming_n_lit = 0.0
      incomming_p_lit = 0.0
      update_c = c_next_pool
      c_next_pool = 0.0
      update_n = n_next_pool
      n_next_pool = 0.0
      update_p = p_next_pool
      p_next_pool = 0.0


      ! LITTER II
      cs_out(2) = cs(2) - cdec(2)

      ! Mineralization
      ! C
      ! Release od CO2
      het_resp(2) = cdec(2) * clit_atm                                       ! Heterotrophic respiration ! processed (dacayed) Carbon lost to ATM

      ! Carbon going to SOIL I
      c_next_pool = cdec(2) - het_resp(2)                                    ! Carbon going to cl_out(2) (next pool)

      ! N
      ! N mineralized by the release of CO2
      n_min_resp_lit = het_resp(2) * snr_in(2)
      p_min_resp_lit = het_resp(2) * snr_in(6)

      ! N going to the SOIL I
      n_next_pool = c_next_pool * snr_in(2) !* 0.5D0
      p_next_pool = c_next_pool * snr_in(6) !* 0.5D0

      ! UPDATE N in Organic MAtter LITTER 2
      nmass_org(2) = nmass_org(2) - (n_min_resp_lit + n_next_pool)
      pmass_org(2) = pmass_org(2) - (p_min_resp_lit + p_next_pool)

      nmin = add_pool(nmin, n_min_resp_lit)
      pmin = add_pool(pmin, p_min_resp_lit)

      n_min_resp_lit = 0.0
      p_min_resp_lit = 0.0

      ! END OF MINERALIZATION PROCESS

      ! UPDATE CNP ORGANIC POOLS

      incomming_c_lit = (frac2 * leaf_l) + (frac2 * root_l) + (frac2 * cwd)                  ! INcoming Carbon from vegetation g m-2
      incomming_n_lit = (frac2 * leaf_n) + (frac2 * froot_n) + (frac2 * wood_n)                 ! Iincoming N
      incomming_p_lit = (frac2 * leaf_p) + (frac2 * froot_p) + (frac2 * wood_p)                 ! Incoming P

      cs_out(2) = add_pool(cs_out(2), incomming_c_lit + update_c)
      nmass_org(2) = add_pool(nmass_org(2), incomming_n_lit + update_n)
      pmass_org(2) = add_pool(pmass_org(2), incomming_p_lit + update_p)

      update_c = 0.0
      update_n = 0.0
      update_p = 0.0

      ! UPDATE SNC
      snc(2) = nmass_org(2)
      snc(6) = pmass_org(2)
      ! END LITTER II

      ! CLEAN AUX VARIABLES
      incomming_c_lit = 0.0
      incomming_n_lit = 0.0
      incomming_p_lit = 0.0
      update_c = c_next_pool
      c_next_pool = 0.0
      update_n = n_next_pool
      n_next_pool = 0.0
      update_p = p_next_pool
      p_next_pool = 0.0

      !SOIL I The same steps commented for the litter pools

      cs_out(3) = cs(3) - cdec(3)

      ! Mineralization
      ! C
      ! Release od CO2
      het_resp(3) = cdec(3) * cwd_atm                                       ! Heterotrophic respiration ! processed (dacayed) Carbon lost to ATM

      ! Carbon going to SOIL 2
      c_next_pool = cdec(3) - het_resp(3)

      ! N
      ! N mineralized by the release of CO2
      n_min_resp_lit = het_resp(3) * snr_in(3)
      p_min_resp_lit = het_resp(3) * snr_in(7)

      ! N going to the SOIL II
      n_next_pool = c_next_pool * snr_in(3) !* 0.01D0
      p_next_pool = c_next_pool * snr_in(7) !* 0.01D0

      ! UPDATE N in Organic MAtter SOIL I
      nmass_org(3) = nmass_org(3) - (n_min_resp_lit + n_next_pool)
      pmass_org(3) = pmass_org(3) - (p_min_resp_lit + p_next_pool)

      nmin = add_pool(nmin, n_min_resp_lit)
      pmin = add_pool(pmin, p_min_resp_lit)

      n_min_resp_lit = 0.0
      p_min_resp_lit = 0.0

      ! UPDATE CNP ORGANIC POOLS

      cs_out(3) = add_pool(cs_out(3), update_c)
      nmass_org(3) = add_pool(nmass_org(3), update_n)
      pmass_org(3) = add_pool(pmass_org(3), update_p)

      update_c = 0.0
      update_n = 0.0
      update_p = 0.0

      ! UPDATE SNC
      snc(3) = nmass_org(3)
      snc(7) = pmass_org(3)
         ! END SOIL 1

      ! CLEAN AUX VARIABLES
      incomming_c_lit = 0.0
      incomming_n_lit = 0.0
      incomming_p_lit = 0.0
      update_c = c_next_pool
      c_next_pool = 0.0
      update_n = n_next_pool
      n_next_pool = 0.0
      update_p = p_next_pool
      p_next_pool = 0.0

      !SOIL II
      ! Mineralization
      ! C
      ! Release od CO2
      het_resp(4) = cdec(4) * cwd_atm                                      ! Heterotrophic respiration ! processed (dacayed) Carbon lost to ATM

      cs_out(4) = cs(4) - het_resp(4)

      ! Carbon going to SOIL 2

      ! N
      ! N mineralized by the release of CO2
      n_min_resp_lit = het_resp(4) * snr_in(4)
      p_min_resp_lit = het_resp(4) * snr_in(8)

      ! UPDATE N in Organic MAtter SOIL II
      nmass_org(4) = nmass_org(4) - n_min_resp_lit
      pmass_org(4) = pmass_org(4) - p_min_resp_lit

      nmin = add_pool(nmin, n_min_resp_lit)
      pmin = add_pool(pmin, p_min_resp_lit)

      n_min_resp_lit = 0.0
      p_min_resp_lit = 0.0

      ! END OF MINERALIZATION PROCESS

      ! UPDATE CNP ORGANIC POOLS
      ! C
      cs_out(4) = cs_out(4) + update_c

      nmass_org(4) = nmass_org(4) + update_n
      pmass_org(4) = pmass_org(4) + update_p

      update_c = 0.0
      update_n = 0.0
      update_p = 0.0

      ! nmin = nmin
      ! pmin = pmin


      ! UPDATE SNC
      snc(4) = nmass_org(4)
      snc(8) = pmass_org(4)

      hr = sum(het_resp)

   end subroutine carbon3


   function carbon_decay(q10_in,tsoil,c,residence_time) result(decay)
   !Based on carbon decay implemented in JeDi and JSBACH - Pavlick et al. 2012
      real(r_4),intent(in) :: q10_in           ! constant ~1.4
      real(r_4),intent(in) :: tsoil            ! Soil temperature °C
      real(r_8),intent(in) :: c                ! Carbon content per area g(C)m-2
      real(r_8),intent(in) :: residence_time   ! Pool turnover rate
      real(r_4) :: decay                       ! ML⁻²
      real(r_4) :: coeff12
      if(c .le. 0.0D0) then
         decay = 0.0
         return
      endif
      coeff12 = real(( c / residence_time), kind=r_4)
      decay = (q10_in ** ((tsoil - 20.0) / 10.0)) * (coeff12)
   end function carbon_decay


   function water_effect(theta) result(retval)
      ! Implement the Moyano function based on soil water content. Moyano et al. 2012;2013
      ! Based on the implementation of Sierra et al. 2012 (SoilR)
      ! This fucntion is ideal and was parametrized for low carbon soils

      real(r_4),intent(in) :: theta  ! Volumetric soil water content (cm³ cm⁻³)
      real(r_4),parameter :: k_a = 3.11, k_b = 2.42
      real(r_4) :: inter, retval, aux

      aux = theta
      if (theta < 0.0) aux = 0.0
      if (theta > 1.0) aux = 1.0

      inter = (k_a * aux) - (k_b * aux**2)
      retval = max(inter, 0.2) ! Residual decay

   end function water_effect


   function sorbed_p_equil(arg) result(retval)
      ! Linear equilibrium between inorganic P and available P pool

      real(r_4), intent(in) :: arg
      real(r_4) :: retval

      retval = arg * ks
   end function sorbed_p_equil


   function solution_p_equil(arg) result(retval)

      real(r_4), intent(in) :: arg
      real(r_4) :: retval

      retval = arg * 0.03
   end function solution_p_equil


   function sorbed_n_equil(arg) result(retval)
      ! Linear equilibrium between inorganic N and available P pool

      real(r_4), intent(in) :: arg
      real(r_4) :: retval

      retval = arg * ks
   end function sorbed_n_equil


   function solution_n_equil(arg) result(retval)

      real(r_4), intent(in) :: arg
      real(r_4) :: retval

      retval = arg * 0.1
   end function solution_n_equil


   function leaching(n_amount, w) result(leached)

      real(r_4), intent(in) :: n_amount, w
      real(r_4) :: leached

      leached = n_amount/w ! DO SOME ALGEBRA * w)

   end function leaching


   function add_pool(a1, a2) result(new_amount)

      real(r_8), intent(in) :: a1, a2
      real(r_8) :: new_amount

      if(a2 .ge. 0.0D0) then
         new_amount = a1 + a2
      else
         new_amount = a1
      endif
   end function add_pool

end module soil_dec
