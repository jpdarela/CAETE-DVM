program a

    real(8),dimension(3) :: arr
    integer(4),dimension(1) :: min_index


integer :: j

do j = 1, 3
   arr(j) = 1.44
enddo

arr(1) = 2.0

min_index = minloc(arr)

print*, min_index

end program a
