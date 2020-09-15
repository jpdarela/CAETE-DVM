
import numpy as np
import rasterio as rio
from netCDF4 import Dataset

mask = np.load("./mask_raisg-360-720.npy")


def neighbours_index(pos, matrix):
    neighbours = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for i in range(max(0, pos[0] - 1), min(rows, pos[0] + 2)):
        for j in range(max(0, pos[1] - 1), min(cols, pos[1] + 2)):
            if (i, j) != pos:
                neighbours.append((i, j))
    return neighbours


def save_nc(fname, arr, varname, long_name, desc, inunit):

    nc_filename = fname

    rootgrp = Dataset(nc_filename, mode='w', format='NETCDF4')

    la = arr.shape[0]
    lo = arr.shape[1]

    # dimensions
    rootgrp.createDimension("latitude", la)
    rootgrp.createDimension("longitude", lo)

    # variables

    latitude = rootgrp.createVariable(varname="latitude",
                                      datatype=np.float32,
                                      dimensions=("latitude",))

    longitude = rootgrp.createVariable(varname="longitude",
                                       datatype=np.float32,
                                       dimensions=("longitude",))

    var_ = rootgrp.createVariable(varname=varname,
                                  datatype=np.float32,
                                  dimensions=("latitude", "longitude",),
                                  fill_value=-9999.0)

    # attributes
    # rootgrp
    rootgrp.description = desc
    rootgrp.source = "CAETÊ inputs dataset"
    # lat
    latitude.units = u"degrees_north"
    latitude.long_name = u"latitude"
    latitude.standart_name = u"latitude"
    latitude.axis = u'Y'
    # lon
    longitude.units = "degrees_east"
    longitude.long_name = "longitude"
    longitude.standart_name = "longitude"
    longitude.axis = 'X'
    # var
    var_.long_name = long_name
    var_.units = inunit
    var_.standard_name = varname
    var_.missing_value = -9999.0
    # WRITING DATA

    longitude[:] = np.arange(-179.75, 180, 0.5)
    latitude[:] = np.arange(-89.75, 90, 0.5)
    var_[:, :] = np.flipud(arr)
    rootgrp.close()


def open_tiff(file_in):
    data = rio.open(file_in)
    ind = data.indexes
    if len(ind) == 1:
        out = (data.read(ind[0])).copy()
    else:
        out = data.read(range(ind[0], ind[-1] + 1)).copy()
        out = np.mean(out, axis=0,)
    data.close()
    return out


def create_npy(filename, func, tiff=True):
    out = np.zeros((360, 720))
    nei_value = []
    if tiff:
        data = open_tiff(filename)
    else:
        data = filename
    for Y in range(360):
        for X in range(720):
            a = mask[Y, X]
            if not a:
                if func(data[Y, X]):
                    lst = neighbours_index([Y, X], mask)
                    print('NODATA found: PIXEL', Y, X, end=' ')
                    print("CALCULATING NEW VALUE...")
                    for i in lst:
                        if func(data[i[0], i[1]]):
                            pass
                        else:
                            nei_value.append(data[i[0], i[1]])
                    print("FILLING NO_DATA WITH NEIGHBOURS VALUES MEAN")
                    data[Y, X] = sum(nei_value) / len(nei_value)
                # transfer data to the new array
                out[Y, X] = data[Y, X]
            else:
                out[Y, X] = -9999.0
    return out


if __name__ == "__main__":
    pass

    # # Work on the total_n file
    # SAVE TOTAL NITROGEN IN SOIL (g m-2)
    np.save("total_n_PA.npy", create_npy("./total_n.tif", np.isnan))

    # # Work on the P files

    # # BULK DENSITY Kg dm⁻³
    bd = create_npy("./b_ds_type_half_degree.tif",
                    lambda n: n < -3.4e+37)
    np.save("bulk_density.npy", bd)

    # #total P mg(P)kg(Solo) ⁻¹
    total_p = create_npy(open_tiff('./predicted_total_p.nc4'),
                         func=lambda n: n == -9999.0, tiff=False)
    avail_p = create_npy(open_tiff('./predicted_avail_p.nc4'),
                         func=lambda n: n == -9999.0, tiff=False)
    org_p = create_npy(open_tiff('./predicted_org_p.nc4'),
                       func=lambda n: n == -9999.0, tiff=False)
    inorg_p = create_npy(open_tiff('./predicted_inorg_p.nc4'),
                         func=lambda n: n == -9999.0, tiff=False)

    Ddim = np.ma.masked_array(
        np.zeros((360, 720)) + 0.30, mask=mask, fill_value=-9999.0)
    # Transoform mg Kg in g m-2

    # BD = kg dm⁻³ to g m⁻³
    bd_gm = np.ma.masked_array(bd, bd == -9999.0, fill_value=-9999.0) * 1.0e6
    # Total P mg kg⁻¹ to g g⁻¹

    total_p_g = np.ma.masked_array(
        total_p, total_p == -9999.0, fill_value=-9999.0) * 1.0e-6

    avail_p_g = np.ma.masked_array(
        avail_p, avail_p == -9999.0, fill_value=-9999.0) * 1.0e-6

    org_p_g = np.ma.masked_array(
        org_p, org_p == -9999.0, fill_value=-9999.0) * 1.0e-6

    inorg_p_g = np.ma.masked_array(
        inorg_p, inorg_p == -9999.0, fill_value=-9999.0) * 1.0e-6

    total_p_ok = Ddim * bd_gm * total_p_g
    total_p_ok[mask] = -9999.0

    avail_p_ok = Ddim * bd_gm * avail_p_g
    avail_p_ok[mask] = -9999.0

    org_p_ok = Ddim * bd_gm * org_p_g
    org_p_ok[mask] = -9999.0

    inorg_p_ok = Ddim * bd_gm * inorg_p_g
    inorg_p_ok[mask] = -9999.0

    np.save("total_p.npy", total_p_ok.data)
    np.save("avail_p.npy", avail_p_ok.data)
    np.save("org_p.npy", org_p_ok.data)
    np.save("inorg_p.npy", inorg_p_ok.data)


    
