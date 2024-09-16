# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

_ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """

# This script contains functions to read binary output
# and create gridded and table outputs.
# Author: Joao Paulo Darela Filho

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Collection, Tuple, Dict, List
from numpy.typing import NDArray

import numpy as np
import pandas as pd

from worker import worker
from region import region
from caete import grd_mt, get_args
from caete_jit import pft_area_frac

from _geos import pan_amazon_region, get_region

#TODO: implement region configuration
if pan_amazon_region is None:
    raise ValueError("pan_amazon_region is not defined or imported correctly")

# Get the region of interest
ymin, ymax, xmin, xmax = get_region(pan_amazon_region)


def get_spins(r:region, gridcell=0):
    """Prints the available spin slices for a gridcell in the region"""
    return r[gridcell].print_available_periods()


def print_variables(r:region):
    """Prints the available variables for each gridcell in the region"""
    r[0]._get_daily_data("DUMMY", 1, pp=True)

def get_var_metadata(var):

    vunits = {'header': ['long_name', 'units', 'standart_name'],
            'rsds': ['Short_wav_rad_down', 'W m-2', 'rsds'],
            'wind': ['Wind_velocity', 'm s-1', 'wind'],
            'ps': ['Sur_pressure', 'Pa', 'ps'],
            'tas': ['Sur_temperature_2m', 'celcius', 'tas'],
            'tsoil': ['Soil_temperature', 'celcius', 'soil_temp'],
            'pr': ['Precipitation', 'Kg m-2 month-1', 'pr'],
            'litter_l': ['Litter C flux - leaf', 'g m-2 day-1', 'll'],
            'cwd': ['Litter C flux - wood', 'g m-2 day-1', 'cwd'],
            'litter_fr': ['Litter C flux fine root', 'g m-2 day-1', 'lr'],
            'litter_n': ['Litter Nitrogen Flux', 'g m-2 day-1', 'ln'],
            'litter_p': ['Litter phosphorus flux', 'g m-2 day-1', 'lp'],
            'sto_c': ['PLant Reserve Carbon', 'g m-2', 'sto_c'],
            'sto_n': ['Pant Reserve Nitrogen', 'g m-2', 'sto_n'],
            'sto_p': ['Plant Reserve Phosphorus', 'g m-2', 'sto_p'],
            'c_cost': ['Carbon costs of Nutrients Uptake', 'g m-2 day-1', 'cc'],
            'wsoil': ['Soil_water_content-wsoil', 'kg m-2', 'mrso'],
            'evapm': ['Evapotranspiration', 'kg m-2 day-1', 'et'],
            'emaxm': ['Potent. evapotrasnpiration', 'kg m-2 day-1', 'etpot'],
            'runom': ['Total_runoff', 'kg m-2 day-1', 'mrro'],
            'aresp': ['Autothrophic respiration', 'kg m-2 year-1', 'ar'],
            'photo': ['Gross primary productivity', 'kg m-2 year-1', 'gpp'],
            'npp': ['Net primary productivity = GPP - AR', 'kg m-2 year-1', 'npp'],
            'rnpp': ['Net primary productivity, C allocation', 'g m-2 day-1', 'npp'],
            'lai': ['Leaf Area Index - LAI', 'm2 m-2', 'lai'],
            'rcm': ['Stomatal resistence', 's m-1', 'rcm'],
            'hresp': ['Soil heterotrophic respiration', 'g m-2 day-1', 'hr'],
            'nupt': ['Nitrogen uptake', 'g m-2 day-1', 'nupt'],
            'pupt': ['Phosphorus uptake', 'g m-2 day-1', 'pupt'],
            'csoil': ['Soil Organic Carbon', 'g m-2', 'csoil'],
            'org_n': ['Soil Organic Nitrogen', 'g m-2', 'org_n'],
            'org_p': ['Soil Organic Phosphorus', 'g m-2', 'org_p'],
            'inorg_n': ['Soil Inorganic Nitrogen', 'g m-2', 'inorg_n'],
            'inorg_p': ['Soil Inorganic Phosphorus', 'g m-2', 'inorg_p'],
            'sorbed_p': ['Soil Sorbed Phosphorus', 'g m-2', 'sorbed_p'],
            'nmin': ['Soil Inorganic Nitrogen (solution)', 'g m-2', 'nmin'],
            'pmin': ['Soil Inorganic Phosphorus (solution)', 'g m-2', 'pmin'],
            'rm': ['Maintenance respiration', 'kg m-2 year-1', 'rm'],
            'rg': ['Growth respiration', 'kg m-2 year-1', 'rg'],
            'wue': ['Water use efficiency', '1', 'wue'],
            'vcmax': ['Maximum RuBisCo activity', 'mol m-2 s-1', 'vcmax'],
            'sla': ['Specfic leaf area', 'm2 g-1', 'sla'],
            'cue': ['Carbon use efficiency', '1', 'cue'],
            'cawood': ['C in woody tissues', 'kg m-2', 'cawood'],
            'cfroot': ['C in fine roots', 'kg m-2', 'cfroot'],
            'cleaf': ['C in leaves', 'kg m-2', 'cleaf'],
            'cmass': ['Total Carbon -Biomass', 'kg m-2', 'cmass'],
            'g1': ['G1 param - Stomatal Resistence model', 'hPA', 'g1'],
            'resopfrac': ['Leaf resorpton fraction N & P', '%', 'resopfrac'],
            'tleaf': ['Leaf C residence time', 'years', 'tleaf'],
            'twood': ['Wood C residence time', 'years', 'twood'],
            'troot': ['Fine root C residence time', 'years', 'troot'],
            'aleaf': ['Allocation coefficients for leaf', '1', 'aleaf'],
            'awood': ['Allocation coefficients for wood', '1', 'awood'],
            'aroot': ['Allocation coefficients for root', '1', 'aroot'],
            'c4': ['C4 photosynthesis pathway', '1', 'c4'],
            'leaf_n2c': ['Leaf N:C', 'g g-1', 'leaf_n2c'],
            'awood_n2c': ['Wood tissues N:C', 'g g-1', 'awood_n2c'],
            'froot_n2c': ['Fine root N:C', 'g g-1', 'froot_n2c'],
            'leaf_p2c': ['Leaf P:C', 'g g-1', 'leaf_p2c'],
            'awood_p2c': ['Wood tissues P:C', 'g g-1', 'awood_p2c'],
            'froot_p2c': ['Fine root P:C', 'g g-1', 'froot_p2c'],
            'amp': ['Percentage of fine root colonized by AM', '%', 'amp'],
            'pdia': ['NPP alocated to N fixers', 'fraction_of_npp', 'pdia'],
            'ls': ['Living Plant Life Strategies', '1', 'ls']
        }
    out = {}
    for v in var:
        out[v] = vunits.get(v, ['unknown', 'unknown', 'unknown'])
    return out


def write_metadata_to_csv(variable_names:Tuple[str,...], output_path:Path):
    metadata = get_var_metadata(("header", ) + variable_names)
    header = metadata.pop("header")
    df = pd.DataFrame(metadata).T
    df.columns = header
    # Add a name for the index col
    df.index.name = "variable_name"
    df.to_csv(output_path / "output_metadata.csv", index_label="variable_name")
    return df


#=========================================
# Functions dealing with gridded outputs
#=========================================
class gridded_data:
    """This class contains methods to read and process gridded data from the model outputs.
    """
     # Daily data --------------------------------
    @staticmethod
    def read_grd(grd:grd_mt,
                 variables: Union[str, Collection[str]],
                 spin_slice: Union[int, Tuple[int, int], None]
                 ) -> Tuple[NDArray, Union[Dict, NDArray, List, Tuple], Union[int, float], Union[int, float]]:
        """helper function to read gridcell output data.

        Args:
            grd (_type_): grd_mt
            variables (Collection[str]): which variables to read from the gridcell
            spin_slice (Union[int, Tuple[int, int], None]): which spin slice to read

        Returns:
            _type_: _description_
        """
        data = grd._get_daily_data(get_args(variables), spin_slice, return_time=True) # type: ignore returns a tuple with data and time Tuple[NDArray, NDArray]
        time = data[-1]
        data = data[0]
        return time, data, grd.y, grd.x


    @staticmethod
    def aggregate_region_data(r: region,
                variables: Union[str, Collection[str]],
                spin_slice: Union[int, Tuple[int, int], None] = None
                )-> Dict[str, NDArray]:
        """_summary_

        Args:
            r (region): a region object

            variables (Union[str, Collection[str]]): variable names to read

            spin_slice (Union[int, Tuple[int, int], None], optional): which spin slice to read.
            Defaults to None, read all available data. Consumes a lot of memory.

        Returns:
            dict: a dict with the following keys: time, coord, data holding data to be transformed
            necessary to create masked arrays and subsequent netCDF files.
        """

        output = []
        nproc = min(len(r), r.nproc//2)
        nproc = max(1, nproc) # Ensure at least one thread is used
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(gridded_data.read_grd, grd, variables, spin_slice) for grd in r]
            for future in futures:
                output.append(future.result())

        # Finalize the data object
        raw_data = np.array(output, dtype=object)
        # Reoeganize resources
        time = raw_data[:,0][0] # We assume all slices have the same time, thus we get the first one
        coord = raw_data[:,2:4][:].astype(np.int64) # 2D matrix of coordinates (y(lat), x(lon))}
        data = raw_data[:,1][:]  # array of dicts, each dict has the variables as keys and the time series as values

        if isinstance(variables, str):
            dim_names = ["time", "coord", variables]
        else:
            dim_names = ["time", "coord", "data"]

        return dict(zip(dim_names, (time, coord, data)))


    @staticmethod
    def create_masked_arrays(data: dict):
        """
        Prototype, used to develop the next def
        Reads a dict generated by aggregate_region_data and reorganize the data
        as masked_arrays with shape=(time, lat, lon) for each variable

        Args:
            data (dict): a dict generated by aggregate_region_data

        Returns:
            _type_: a tuple with a list of masked_arrays (for each variable)
            and the time array.
        """
        time = data["time"]
        coords = data["coord"]
        # If it is only one variable, we put
        variables = list(data["data"][0].keys()) # holds variable names being processed

        dim = data["data"][0][variables[0]].shape

        # TODO: manage 2D and 3D arrays
        assert len(dim) == 1, "Only 1D array allowed"
        arrays_dict = data["data"][:]

        # Read dtypes
        dtypes = []
        for var in variables:
            # We assume all gridcells have the same variables, thus we get the first one
            dtypes.append(arrays_dict[0][var].dtype)

        arrays = []
        for i, var in enumerate(variables):
            arrays.append(np.ma.masked_all(shape=(dim[0], 360, 720), dtype=dtypes[i]))

        for i, var in enumerate(variables):
            for j in range(len(coords)):
                arrays[i][:, coords[j][0], coords[j][1]] = arrays_dict[j][var]
        # Crop the arrays to the region of interest
        arrays = [a[:, ymin:ymax, xmin:xmax] for a in arrays]

        return arrays, time


    @staticmethod
    def create_masked_arrays2D(data: dict):
        """ Reads a dict generated by aggregate_region_data and reorganize the data
        as masked_arrays with shape=(time, lat, lon) for each variable

        Args:
            data (dict): a dict generated by aggregate_region_data

        Returns:
            _type_: a tuple with a list of masked_arrays (for each variable),
            the time array, and the array names.
        """
        time = data["time"]
        coords = data["coord"]

        assert "data" in data.keys(), "The input dict must contain the 'data' keyword"
        assert isinstance(data["data"][0], dict), "Data must be a dict"
        variables = list(data["data"][0].keys())  # holds variable names being processed

        arrays_dict = data["data"][:]

        # Read dtypes
        dtypes = []
        for var in variables:
            dtypes.append(arrays_dict[0][var].dtype)

        # Allocate the arrays
        arrays = []
        array_names = []
        for i, var in enumerate(variables):
            dim = arrays_dict[0][var].shape
            if len(dim) == 1:
                arrays.append(np.ma.masked_all(shape=(dim[0], 360, 720), dtype=dtypes[i]))
                array_names.append(var)
            elif len(dim) == 2:
                ny, nx = dim
                for k in range(ny):
                    arrays.append(np.ma.masked_all(shape=(nx, 360, 720), dtype=dtypes[i]))
                    array_names.append(f"{var}_{k + 1}")
        # Fill the arrays
        array_index = 0
        for i, var in enumerate(variables):
            for j in range(len(coords)):
                if len(arrays_dict[j][var].shape) == 1:
                    arrays[array_index][:, coords[j][0], coords[j][1]] = arrays_dict[j][var]
                elif len(arrays_dict[j][var].shape) == 2:
                    ny, nx = arrays_dict[j][var].shape
                    for k in range(ny):
                        arrays[array_index + k][:, coords[j][0], coords[j][1]] = arrays_dict[j][var][k, :]
            array_index += ny if len(arrays_dict[j][var].shape) == 2 else 1

        # Crop the arrays to the region of interest
        arrays = [a[:, ymin:ymax, xmin:xmax] for a in arrays]

        return arrays, time, array_names


    @staticmethod
    def save_netcdf(data: dict, output_path: Path, file_name: str):
        pass

# ======================================
# Functions dealing with table outputs
# ======================================


class table_data:

    @staticmethod
    def make_daily_dataframe(r:region,
                variables: Union[str, Collection[str]],
                spin_slice: Union[int, Tuple[int, int], None] = None
                ):

        for grd in r:
            d = grd._get_daily_data(get_args(variables), spin_slice, return_time=True) #type: ignore

            time = [t.strftime("%Y-%m-%d") for  t in d[1]] # type: ignore
            data = d[0] # type: ignore

            new_data = {}
            for k, v in data.items(): # type: ignore
                if len(v.shape) == 1:
                    new_data[k] = v
                elif len(v.shape) == 2:
                    ny, _ = v.shape
                    _sum = np.sum(v, axis=0)
                    new_data[f"{k}_sum"] = _sum
                    for i in range(ny):
                        new_data[f"{k}_{i+1}"] = v[i,:] # We assume the first axis is the time axis

            fname = f"grd_{grd.x}_{grd.y}_{time[0]}_{time[-1]}.csv"
            df = pd.DataFrame(new_data, index=time)
            df.rename_axis('day', axis='index')
            df.to_csv(grd.out_dir / fname, index_label='day')


    @staticmethod
    def write_daily_data(r:region, variables:Union[str, Collection[str]]):
        periods = r[0].print_available_periods()
        write_metadata_to_csv(variables, r.output_path) # type: ignore
        for i in range(periods):
            table_data.make_daily_dataframe(r, variables=variables, spin_slice=i+1)


    @staticmethod
    def write_metacomm_output(grd:grd_mt) -> None:
        out = []
        for year in grd._get_years():
            biomass = ["vp_cleaf", "vp_croot", "vp_cwood"]
            data = grd._read_annual_metacomm_biomass(year)
            df = pd.DataFrame(data).astype(np.float32).groupby("pls_id").sum().reset_index()
            df.index = df["pls_id"].astype(np.int32) # type: ignore
            df = df.loc[:, biomass]
            cleaf, croot, cwood = df["vp_cleaf"].to_numpy(), df["vp_croot"].to_numpy(), df["vp_cwood"].to_numpy()
            ocp = pft_area_frac(cleaf, croot, cwood)
            df.loc[:, "cveg"] = cleaf + croot + cwood
            df.loc[:, "ocp"] = ocp
            df.loc[:, "year"] = np.zeros(ocp.size, dtype=np.int32) + year
            out.append(df)
        pd.concat(out).to_csv(grd.out_dir / "metacomunity_biomass.csv", index_label="pls_id")


class output_manager:

    @staticmethod
    def cities_output():
            # # IO
        hist_results: Path = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
        piControl_results: Path = Path("./cities_MPI-ESM1-2-HR-piControl_output.psz")
        ssp370_results: Path = Path("./cities_MPI-ESM1-2-HR-ssp370_output.psz")
        ssp585_results: Path = Path("./cities_MPI-ESM1-2-HR-ssp585_output.psz")

        # # Load the region file
        hist:region = worker.load_state_zstd(hist_results)
        piControl:region = worker.load_state_zstd(piControl_results)
        ssp370:region = worker.load_state_zstd(ssp370_results)
        ssp585:region = worker.load_state_zstd(ssp585_results)

        variables_to_read: Tuple[str,...] = ("cue", "wue", "csoil", "hresp", "aresp", "rnpp", "photo", "npp", "evapm")

        table_data.write_daily_data(r=hist, variables=variables_to_read)
        table_data.write_daily_data(r=piControl, variables=variables_to_read)
        table_data.write_daily_data(r=ssp370, variables=variables_to_read)
        table_data.write_daily_data(r=ssp585, variables=variables_to_read)

        for grd in hist:
            table_data.write_metacomm_output(grd)
        for grd in ssp370:
            table_data.write_metacomm_output(grd)
        for grd in ssp585:
            table_data.write_metacomm_output(grd)
        for grd in piControl:
            table_data.write_metacomm_output(grd)




if __name__ == "__main__":

    output_manager.cities_output()
    # hist_results: Path = Path("./cities_MPI-ESM1-2-HR_hist_output.psz")
    # hist:region = worker.load_state_zstd(hist_results)
    # grd = hist[0]



    # # IO
    # model_results: Path = Path("./pan_amazon_hist_result.psz")
    # # Load the region file
    # reg:region = worker.load_state_zstd(model_results)
    # # Gridded outputs
    # # variables_to_read = ("cue", "rnpp", "aresp", "photo", "csoil")
    # # data = data = gridded_data.aggregate_region_data(reg, variables_to_read, (24,25))
    # # a = gridded_data.create_masked_arrays2D(data)
    # # TODO: Save netcdfs

