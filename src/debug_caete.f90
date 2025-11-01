! I use prints and python scripts most of the time to test the fortran library
! Sometimes we need to step in and debug the fortran code directly,
! so this is a simple example of a fortran program
! that calls the carbon3 subroutine from the caete_module.
! You can implement your own test program to debug the fortran code

program test_caete

   use types
   use global_par
   use photo
   use water
   use soil_dec
   use budget

   implicit none

   print *, "Testing/debugging carbon3 subroutine from caete_module"

   call test_c3()
   ! call test_daily_budget()

   contains

   subroutine test_c3()

      integer(i_4) :: index
      real(r_8) :: soilt=25.0, water_s=0.9
      real(r_8) :: ll=5.5, lf=5.5, lw=5.5
      real(r_8), dimension(6) :: lnc = (/2.5461449101567262D-002, 1.2789730913937092D-002, 4.1226762905716891D-002,&
                                        & 3.2206000294536350D-003, 3.1807350460439920D-003, 4.0366222383454442D-003/)
      real(r_8), dimension(4) :: cs = 0.1, cs_out = 0.1
      real(r_8), dimension(8) :: snc = 0.00001, snc_in = 0.0001
      real(r_8) :: hr
      real(r_8) :: nmin, pmin

      do index = 1,200000

         call carbon3(soilt, water_s, ll, lw, lf, lnc, cs, snc_in, cs_out, snc, hr, nmin, pmin)

         cs = cs_out
         snc_in = snc

         print *, snc,"<- snc"
         print *, hr,"<- hr"
         print *, nmin, pmin, "<- N & P"
         print *, cs,"<- cs"
      end do
   end subroutine test_c3

   ! subroutine test_daily_budget()
   !    use types
   !    use global_par
   !    use budget

   !    implicit none

   !    ! Declare variables (sizes are examples, adjust as needed)
   !    integer(i_4), parameter :: ntraits = 20, npls = 5
   !    real(r_8) :: dt(ntraits, npls)
   !    real(r_8) :: w1, w2, mineral_n, labile_p, on, sop, op, catm, wmax_in
   !    real(r_8) :: ts, temp, p0, ipar, rh
   !    real(r_8) :: cl1_in(npls), ca1_in(npls), cf1_in(npls)
   !    real(r_8) :: uptk_costs_in(npls), rnpp(npls)
   !    real(r_8) :: sto_budg_in(3, npls)
   !    ! Outputs
   !    real(r_8) :: epavg
   !    real(r_8) :: evavg, phavg, aravg, nppavg, laiavg, rcavg, f5avg, rmavg, rgavg
   !    real(r_8) :: cleafavg_pft(npls), cawoodavg_pft(npls), cfrootavg_pft(npls)
   !    real(r_8) :: storage_out_bdgt_1(3, npls), ocpavg(npls), wueavg, cueavg, c_defavg
   !    real(r_8) :: vcmax_1, specific_la_1, nupt_1(2), pupt_1(3), litter_l_1, cwd_1, litter_fr_1
   !    real(r_8) :: npp2pay_1(npls), lit_nut_content_1(6)
   !    integer(i_2) :: limitation_status_1(3, npls)
   !    integer(i_4) :: uptk_strat_1(2, npls)
   !    real(r_8) :: cp(4), c_cost_cwm, rnpp_out(npls)

   !    integer :: i, j

   !    ! Initialize inputs with dummy values
   !    dt = 1.0D0
   !    w1 = 100.0D0
   !    w2 = 100.0D0
   !    ts = 25.0
   !    temp = 25.0
   !    p0 = 1013.0
   !    ipar = 1000.0
   !    rh = 0.7
   !    mineral_n = 1.0D0
   !    labile_p = 1.0D0
   !    on = 1.0D0
   !    sop = 1.0D0
   !    op = 1.0D0
   !    catm = 400.0D0
   !    wmax_in = 200.0D0

   !    do i = 1, npls
   !       cl1_in(i) = 5.0D0
   !       ca1_in(i) = 5.0D0
   !       cf1_in(i) = 5.0D0
   !       uptk_costs_in(i) = 0.1D0
   !       rnpp(i) = 1.0D0
   !       ocpavg(i) = 1.0D0 / npls
   !       rnpp_out(i) = 0.0D0
   !       do j = 1, 3
   !          sto_budg_in(j, i) = 0.5D0
   !       end do
   !    end do

   !    ! Call the daily_budget subroutine
   !    call daily_budget(dt, w1, w2, ts, temp, p0, ipar, rh, mineral_n, labile_p, on, sop, op, catm, &
   !         sto_budg_in, cl1_in, ca1_in, cf1_in, uptk_costs_in, wmax_in, rnpp, &
   !         evavg, epavg, phavg, aravg, nppavg, laiavg, rcavg, f5avg, rmavg, rgavg, &
   !         cleafavg_pft, cawoodavg_pft, cfrootavg_pft, storage_out_bdgt_1, ocpavg, wueavg, cueavg, &
   !         c_defavg, vcmax_1, specific_la_1, nupt_1, pupt_1, litter_l_1, cwd_1, litter_fr_1, &
   !         npp2pay_1, lit_nut_content_1, limitation_status_1, uptk_strat_1, cp, c_cost_cwm, rnpp_out)

   !    ! Print some outputs for inspection
   !    print *, "evavg: ", evavg
   !    print *, "epavg: ", epavg
   !    print *, "phavg: ", phavg
   !    print *, "aravg: ", aravg
   !    print *, "nppavg: ", nppavg
   !    print *, "laiavg: ", laiavg
   !    print *, "rcavg: ", rcavg
   !    print *, "f5avg: ", f5avg
   !    print *, "rmavg: ", rmavg
   !    print *, "rgavg: ", rgavg
   !    print *, "wueavg: ", wueavg
   !    print *, "cueavg: ", cueavg
   !    print *, "c_defavg: ", c_defavg
   !    print *, "vcmax_1: ", vcmax_1
   !    print *, "specific_la_1: ", specific_la_1
   !    print *, "litter_l_1: ", litter_l_1
   !    print *, "cwd_1: ", cwd_1
   !    print *, "litter_fr_1: ", litter_fr_1
   !    print *, "c_cost_cwm: ", c_cost_cwm
   !    print *, "cp: ", cp
   !    print *, "nupt_1: ", nupt_1
   !    print *, "pupt_1: ", pupt_1
   !    print *, "lit_nut_content_1: ", lit_nut_content_1
   !    print *, "cleafavg_pft: ", cleafavg_pft
   !    print *, "cawoodavg_pft: ", cawoodavg_pft
   !    print *, "cfrootavg_pft: ", cfrootavg_pft
   !    print *, "rnpp_out: ", rnpp_out

   ! end subroutine test_daily_budget

end program test_caete
