import os
import glob
import cftime as cft
from netCDF4 import num2date, MFDataset


class data_in():

    """Program to extract CAETÊ input data from ISIMIP netCDF4 files"""

    def __init__(self, varname, inputs_folder):

        self.varname = varname
        self.root_dir = os.getcwd()
        self.files = sorted(glob.glob1(inputs_folder, self.varname))
        self.inputs_folder = inputs_folder
        self.celldata = {}
        self.varnames = ['tas', 'ps', 'pr', 'rsds', 'hurs']

#        self.varmask = [True, True, True, True, True, True, True, False, False]
        return None

    def _open_dts(self):
        """Use de MFdataset class for read the array in a netCDF file"""

        # TODO Implement this mechanism without changing directory
        #     e.g. use os.path module
        assert self.varname in self.varnames, 'Unknown Variable: %s' % varname
        # Insert this block in a try/except to catch IO errors and return to the correct dir
        os.chdir(self.inputs_folder)
        fpath = self.varname + '_' + '*' + '.nc4'
        dt = MFDataset(fpath, 'r')
        os.chdir(self.root_dir)
        return dt

    def data_dict(self, mask=None):
        """ An object that stores input data for caete

           Create a data dictionary (self.celldata) with the input data for CAETÊ
           mask: np.array(dtype = bool) mask for creation of several gridcells SHAPE=(360, 720)
        """

        assert len(mask.shape) == 2

        dim1 = mask.shape[0]
        dim2 = mask.shape[1]

        print("Extracting var %s" % self.varname)
        with self._open_dts() as fh:
            init_date = num2date(
                fh.variables['time'][0], fh.variables['time'].units, fh.variables['time'].calendar)
            end_date = num2date(
                fh.variables['time'][-1], fh.variables['time'].units, fh.variables['time'].calendar)
            init_index = cft.date2index(init_date, fh.variables['time'])
            end_index = cft.date2index(end_date, fh.variables['time'])
            cells_done = self.celldata.keys()
            if 'metadata' not in cells_done:
                self.celldata['metadata'] = {
                    'varname':
                    self.varname,
                    'calendar':
                    fh.variables['time'].calendar,
                    'time_unit':
                    fh.variables['time'].units,
                    'var_unit':
                    fh.variables[self.varname].units,
                    'var_nodata': fh.variables[self.varname].missing_value,
                    'init_date': init_date,
                    'end_date': end_date,
                    'start_index': init_index,
                    'end_index': end_index,
                    'time_data':
                    (fh.variables['time'][0],
                        fh.variables['time'][-1]),
                    'lat_unit':
                    fh.variables['lat'].units,
                    'latitude':
                    (fh.variables['lat'][0],
                        fh.variables['lat'][-1]),
                    'lon_unit':
                    fh.variables['lon'].units,
                    'longitude':
                    (fh.variables['lon'][0],
                        fh.variables['lon'][-1]),
                    'ny':
                    fh.dimensions['lat'].size,
                    'nx':
                    fh.dimensions['lon'].size,
                    'len':
                    fh.variables['time'][:].size
                }
            for Y in range(dim1):
                for X in range(dim2):
                    if not mask[Y, X]:
                        print("...gridcell(%d, %d)" % (Y, X))
                        k = self.varname + '_' + str(Y) + '-' + str(X)
                        if k in cells_done:
                            continue
                        # Fill the dict
                        self.celldata[k] = {
                            'var_data':
                            fh.variables[self.varname][:, Y, X],
                        }
        return self.celldata
