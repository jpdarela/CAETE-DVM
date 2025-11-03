#!/bin/bash
# Bash script to transfer folders to sombrero server using rsync
# Assumes SSH config for sombrero is already set up

REMOTE1="sombrero:~/CAETE-DVM/input/20CRv3-ERA5"
REMOTE2="sombrero:~/CAETE-DVM/input/MPI-ESM1-2-HR"

# Rsync for 20CRv3-ERA5 subfolders
folders1=(obsclim spinclim)
for folder in "${folders1[@]}"; do
    echo "Transferring 20CRv3-ERA5/$folder to $REMOTE1/$folder ..."
    rsync -avz --progress 20CRv3-ERA5/$folder/ "$REMOTE1/$folder/"
    echo "Done with $folder."
done

# Rsync for MPI-ESM1-2-HR subfolders
folders2=(historical piControl ssp370 ssp585)
for folder in "${folders2[@]}"; do
    echo "Transferring MPI-ESM1-2-HR/$folder to $REMOTE2/$folder ..."
    rsync -avz --progress MPI-ESM1-2-HR/$folder/ "$REMOTE2/$folder/"
    echo "Done with $folder."
done

echo "All folders have been transferred using rsync."
