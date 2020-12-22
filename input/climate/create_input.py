import pickle as pkl
import numpy as np
from mpi4py import MPI
from caete_input import data_in


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Location of the input data downloaded from ISIMIP-PIK server via rsync
files = "../ISIMIP-HISTORICAL/"
mask = np.load("../mask/mask_raisg-360-720.npy")


# INCLUDING ONLY 4 gridcells
mask[mask == False] = True
mask[183, 238] = False
mask[184, 238] = False
mask[185, 238] = False
mask[186, 238] = False

# INstantiate our input data object
hurs = data_in("hurs", files)
rsds = data_in("rsds", files)
tas = data_in("tas", files)
pr = data_in("pr", files)
ps = data_in("ps", files)


if rank == 0:
    ru = hurs.data_dict(mask=mask)
    with open("hurs.pkl", 'wb') as fh1:
        pkl.dump(ru, fh1)

elif rank == 1:
    ipar = rsds.data_dict(mask=mask)
    with open("rsds.pkl", 'wb') as fh2:
        pkl.dump(ipar, fh2)

elif rank == 2:
    temp = tas.data_dict(mask=mask)
    with open("tas.pkl", 'wb') as fh3:
        pkl.dump(temp, fh3)

elif rank == 3:
    prec = pr.data_dict(mask=mask)
    with open("pr.pkl", 'wb') as fh4:
        pkl.dump(prec, fh4)

else:
    p0 = ps.data_dict(mask=mask)
    with open("ps.pkl", 'wb') as fh5:
        pkl.dump(p0, fh5)
