import pickle as pkl
import numpy as np
from mpi4py import MPI
from caete_input import data_in


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Location of the input data downloaded from ISIMIP-PIK server via rsync
files = '/home/jdarela/Desktop/caete/avante_caete/caete-dgvm/input/inputs_daily'
mask = np.load(
    "/home/jdarela/Desktop/caete/avante_caete/caete-dgvm/input/mask_raisg-360-720.npy")


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


# ## Use with a mask (creates a bunch og gridcells)
    # #data_in = data_in.data_dict(mask=ct.mask)

    # # OR

# ## Update a existent instance of data_in
# with open("data_in-instance.pkl", 'rb') as fh:
#     data_in = pkl.load(fh)

# # Apply data_dict creation method:

# grid_points = [(239, 183),
#                (219, 184),
#                (404, 186),
#                (440, 61),
#                (572, 55)]


# for x, y in grid_points:
#     for var in data_in.varnames:
#         data_in.data_dict(varname=var, nx=x, ny=y)

# with open("data_in-instance.pkl", 'wb') as fh:
#     pkl.dump(data_in, fh)
