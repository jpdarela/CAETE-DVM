! test_iso_kind.f90
program test
    use iso_fortran_env

    print*, logical_kinds
    print*, real32
    print*, real64
    print*, real128
    print*, int16
    print*, int32

end program test
