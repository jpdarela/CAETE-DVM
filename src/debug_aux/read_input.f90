program read_input

    implicit none

    ! Declare variables
    real, allocatable :: time_series(:)
    real :: scalar
    character(len=256) :: file_name
    integer :: i, io_status, num_values

    ! List of files to read
    character(len=256), dimension(12) :: file_list = &
        ["ap.txt", "tp.txt", "tn.txt", "tasmin.txt", "tasmax.txt", "tas.txt", &
         "sfcwind.txt", "rsds.txt", "ps.txt", "pr.txt", "op.txt", "ip.txt"]

    ! Loop through each file
    do i = 1, size(file_list)
        file_name = trim(file_list(i))

        ! Check if the file exists
        inquire(file=file_name, exist=io_status)
        if (io_status == 0) then
            print *, "File not found: ", file_name
            cycle
        end if

        open(unit=10, file=file_name, status='old', action='read')

        ! Count the number of values in the file
        num_values = 0
        do
            read(10, *, iostat=io_status)
            if (io_status /= 0) exit
            num_values = num_values + 1
        end do

        rewind(10)

        if (num_values == 1) then
            ! File contains a scalar value
            read(10, *) scalar
            print *, "Scalar value from ", file_name, ": ", scalar
        else
            ! File contains a time series
            allocate(time_series(num_values))
            read(10, *) time_series
            print *, "Time series from ", file_name, ": ", time_series
            deallocate(time_series)
        end if

        close(10)
    end do

end program read_input