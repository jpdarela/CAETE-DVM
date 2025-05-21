# PowerShell script to transfer folders to sombrero server using scp
# Assumes SSH config for sombrero is already set up

$remote_1 = "sombrero:~/CAETE-DVM/input/20CRv3-ERA5"
$remote_2 = "sombrero:~/CAETE-DVM/input/MPI-ESM1-2-HR"

scp -r 20CRv3-ERA5\counterclim $remote_1
scp -r 20CRv3-ERA5\obsclim $remote_1
scp -r 20CRv3-ERA5\spinclim $remote_1
scp -r 20CRv3-ERA5\transclim $remote_1

scp -r MPI-ESM1-2-HR\historical $remote_2
scp -r MPI-ESM1-2-HR\piControl $remote_2
scp -r MPI-ESM1-2-HR\ssp370 $remote_2
scp -r MPI-ESM1-2-HR\ssp585 $remote_2
