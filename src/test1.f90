program test1
    real(8), dimension(:,:), allocatable :: x, y, z
    integer(4) :: n = 500
    real(8), dimension(500) :: vec

    call random_number(vec)

    print*, vec
    allocate(x(n, n))
    allocate(z(n, n))
    allocate(y(n, n))

    call random_number(x)
    call random_number(y)
    call random_number(z)

    where(x > 0.2)
        x = 0
    elsewhere(x > 0.8)
        x = 1
    elsewhere
        x = -1
    endwhere

    print*, x
    deallocate(x)
    deallocate(z)
    deallocate(y)

end program test1
