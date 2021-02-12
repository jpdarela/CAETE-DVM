! test_parallel.f90

program test_parallel
    use omp_lib
    implicit none

    real(8) :: smavg, ruavg, evavg, phavg, aravg, nppavg, laiavg, rcavg

    real(8), dimension(5000000) :: smelt, roff, ocp_coeffs, evap, ph, ar, nppa, laia, rc2

    character(len=1) :: test
    integer :: p, nlen, j
    logical :: t = .false.
    call random_number(ocp_coeffs)
    call random_number(roff)
    call random_number(smelt)
    call random_number(evap)
    call random_number(ph)
    call random_number(ar)
    call random_number(nppa)
    call random_number(laia)
    call random_number(rc2)

    read*, test

    if(test .eq. 't') then
        print *, 'serial with sum'
       smavg = sum(smelt * ocp_coeffs)
       ruavg = sum(roff * ocp_coeffs)
       evavg = sum(evap * ocp_coeffs)
       phavg = sum(ph * ocp_coeffs)
       aravg = sum(ar * ocp_coeffs)
       nppavg = sum(nppa * ocp_coeffs)
       laiavg = sum(laia * ocp_coeffs)
       rcavg = sum(rc2 * ocp_coeffs)

    else
        call OMP_SET_NUM_THREADS(2)
        print*, 'parallel'

        !$OMP PARALLEL DO &
        !$OMP SCHEDULE(AUTO) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(p)
        do p = 1, nlen
            smavg = smavg + (smelt(p) * ocp_coeffs(p))
            ruavg = ruavg + (roff(p) * ocp_coeffs(p))
            evavg = evavg + (evap(p) * ocp_coeffs(p))
            phavg = phavg + (ph(p) * ocp_coeffs(p))
            aravg = aravg + (ar(p) * ocp_coeffs(p))
            nppavg = nppavg + (nppa(p) * ocp_coeffs(p))
            laiavg = laiavg + (laia(p) * ocp_coeffs(p))
            rcavg = rcavg + (rc2(p) * ocp_coeffs(p))
        enddo
        !$OMP PARALLEL DO
        ! print*, ruavg, smelt
    endif

end program test_parallel
