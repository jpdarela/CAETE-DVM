# -*-coding:utf-8-*-
# "CAETÊ"
# Author João Paulo Darela Filho

"""
Copyright 2017- LabTerra

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import _pickle as cPickle
import bz2
import concurrent.futures

import cftime
import numpy as np

from caete_module import global_par as gp
from caete_module import budget as model
from caete_module import water as st
from caete_module import photo as m
from caete_module import soil_dec
from caete_module import utils as utl

# GLOBAL
npls = gp.npls
# Mask for model execution
mask = np.load('../input/mask/mask_raisg-360-720.npy')
# Create the semi-random table// of Plant Life Strategies
# AUX FUNCS


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=30):
    """FROM Stack Overflow/GIST, THANKS
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    bar_utf = b'\xe2\x96\x88'  # bar -> unicode symbol = u'\u2588'
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def wm(weight, data):
    shp = data.shape
    if len(shp) < 2:
        return np.sum(weight * data)
    else:
        out = np.zeros(shape=(shp[0],))
        for i in range(shp[0]):
            out[i] = np.sum(weight * data[i, :])
        return out


def neighbours_index(pos, matrix):
    neighbours = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for i in range(max(0, pos[0] - 1), min(rows, pos[0] + 2)):
        for j in range(max(0, pos[1] - 1), min(cols, pos[1] + 2)):
            if (i, j) != pos:
                neighbours.append((i, j))
    return neighbours


def catch_out_budget(out):
    lst = ["w2", "g2", "s2", "smavg", "ruavg", "evavg", "epavg", "phavg", "aravg", "nppavg",
           "laiavg", "rcavg", "f5avg", "rmavg", "rgavg", "cleafavg_pft", "cawoodavg_pft",
           "cfrootavg_pft", "stodbg", "ocpavg", "wueavg", "cueavg", "c_defavg", "vcmax",
           "specific_la", "nupt", "pupt", "litter_l", "cwd", "litter_fr", "lnc", "delta_cveg",
           "mineral_n_pls", "labile_p_pls", "limitation_status", "sto_min"]

    return dict(zip(lst, out))


def catch_out_carbon3(out):
    lst = ['cs', 'snc', 'hr', 'nmin', 'pmin']

    return dict(zip(lst, out))

# MODEL


class grd:

    """
    Defines the gridcell object - This object stores all the input data,
    the data comming from model runs for each grid point, all the state varables and all the metadata
    describing the life cycle of the gridcell
    """

    def __init__(self, x, y):
        """Construct the gridcell object"""

        # CELL Identifiers
        self.x = x                            # Grid point x coordinate
        self.y = y                            # Grid point y coordinate
        self.xyname = str(y) + '-' + str(x)   # IDENTIFIES GRIDCELLS
        self.name = None                      # identifies gridcells started together
        self.pos = (int(self.x), int(self.y))
        self.pls_table = None   # will receive the np.array with functional traits data
        self.outputs = {}       # dict, store filepaths of output data
        # counts the execution of a time slice (a call of self.run_spinup)
        self.run_counter = 0
        self.neighbours = None
        self.ls = None          # Number of surviving plss//

        self.out_dir = "./outputs/gridcell{}/".format(self.xyname)
        self.flush_data = None

        # Time attributes
        self.time_index = None  # Array with the time stamps
        self.calendar = None    # Calendar name
        self.time_unit = None   # Time unit
        self.start_date = None
        self.end_date = None
        self.ssize = None
        self.sind = None
        self.eind = None

        # Input data
        self.filled = False     # Indicates when the gridcell is filled with input data
        self.pr = None
        self.ps = None
        self.rsds = None
        self.tas = None
        self.rhs = None

        # Spinup data
        self.clin = None
        self.cfin = None
        self.cwin = None

        # OUTPUTS
        self.soil_temp = None
        self.emaxm = None
        self.tsoil = None
        self.photo = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.lim_status = None

        # WATER POOLS
        self.wfim = None
        self.gfim = None
        self.sfim = None

        # SOIL POOLS
        self.input_nut = None
        self.sp_available_p = None
        self.sp_available_n = None
        self.sp_so_n = None
        self.sp_in_n = None
        self.sp_so_p = None
        self.sp_in_p = None
        self.sp_csoil = None
        self.sp_snr = None
        self.soil_spinup = None

        # CVEG POOLS
        self.vp_cleaf = None
        self.vp_croot = None
        self.vp_cwood = None
        self.vp_dcl = None
        self.vp_dca = None
        self.vp_dcf = None
        self.vp_ocp = None
        self.vp_wdl = None
        self.vp_sto = None

    def _allocate_output(self, n):
        """allocate space for the outputs
        n: int NUmber of days being simulated"""
        self.emaxm = []
        self.tsoil = []
        self.photo = np.zeros(shape=(n,), order='F')
        self.aresp = np.zeros(shape=(n,), order='F')
        self.npp = np.zeros(shape=(n,), order='F')
        self.lai = np.zeros(shape=(n,), order='F')
        self.csoil = np.zeros(shape=(4, n), order='F')
        self.inorg_n = np.zeros(shape=(n,), order='F')
        self.inorg_p = np.zeros(shape=(n,), order='F')
        self.sorbed_n = np.zeros(shape=(n,), order='F')
        self.sorbed_p = np.zeros(shape=(n,), order='F')
        self.snc = np.zeros(shape=(8, n), order='F')
        self.hresp = np.zeros(shape=(n,), order='F')
        self.rcm = np.zeros(shape=(n,), order='F')
        self.f5 = np.zeros(shape=(n,), order='F')
        self.runom = np.zeros(shape=(n,), order='F')
        self.evapm = np.zeros(shape=(n,), order='F')
        self.wsoil = np.zeros(shape=(n,), order='F')
        self.rm = np.zeros(shape=(n,), order='F')
        self.rg = np.zeros(shape=(n,), order='F')
        self.cleaf = np.zeros(shape=(n,), order='F')
        self.cawood = np.zeros(shape=(n,), order='F')
        self.cfroot = np.zeros(shape=(n,), order='F')
        self.area = np.zeros(shape=(npls, n))
        self.wue = np.zeros(shape=(n,), order='F')
        self.cue = np.zeros(shape=(n,), order='F')
        self.cdef = np.zeros(shape=(n,), order='F')
        self.nmin = np.zeros(shape=(n,), order='F')
        self.pmin = np.zeros(shape=(n,), order='F')
        self.vcmax = np.zeros(shape=(n,), order='F')
        self.specific_la = np.zeros(shape=(n,), order='F')
        self.nupt = np.zeros(shape=(n,), order='F')
        self.pupt = np.zeros(shape=(n,), order='F')
        self.litter_l = np.zeros(shape=(n,), order='F')
        self.cwd = np.zeros(shape=(n,), order='F')
        self.litter_fr = np.zeros(shape=(n,), order='F')
        self.lnc = np.zeros(shape=(6, n), order='F')
        self.storage_pool = np.zeros(shape=(3, n), order='F')
        self.ls = np.zeros(shape=(n,), order='F')
        self.lim_status = np.zeros(
            shape=(3, npls, n), dtype=np.dtype('int16'), order='F')

    def _flush_output(self, run_descr, index):
        """1 - Clean variables that receive outputs from the fortran subroutines
           2 - Fill self.outputs dict with filepats of output data
           3 - Returns the output data

           runs_descr: str a name for the files
           index = tuple or list with the first and last values of the index time variable"""
        to_pickle = {}
        self.run_counter += 1
        if self.run_counter < 10:
            spiname = run_descr + "0" + str(self.run_counter) + ".pbz2"
        else:
            spiname = run_descr + str(self.run_counter) + ".pbz2"

        self.outputs[spiname] = self.out_dir + spiname
        to_pickle = {'emaxm': np.array(self.emaxm),
                     "tsoil": np.array(self.tsoil),
                     "photo": self.photo,
                     "aresp": self.aresp,
                     'npp': self.npp,
                     'lai': self.lai,
                     'csoil': self.csoil,
                     'inorg_n': self.inorg_n,
                     'inorg_p': self.inorg_p,
                     'sorbed_n': self.sorbed_n,
                     'sorbed_p': self.sorbed_p,
                     'snc': self.snc,
                     'hresp': self.hresp,
                     'rcm': self.rcm,
                     'f5': self.f5,
                     'runom': self.runom,
                     'evapm': self.evapm,
                     'wsoil': self.wsoil,
                     'rm': self.rm,
                     'rg': self.rg,
                     'cleaf': self.cleaf,
                     'cawood': self.cawood,
                     'cfroot': self.cfroot,
                     'area': self.area,
                     'wue': self.wue,
                     'cue': self.cue,
                     'cdef': self.cdef,
                     'nmin': self.nmin,
                     'pmin': self.pmin,
                     'vcmax': self.vcmax,
                     'specific_la': self.specific_la,
                     'nupt': self.nupt,
                     'pupt': self.pupt,
                     'litter_l': self.litter_l,
                     'cwd': self.cwd,
                     'litter_fr': self.litter_fr,
                     'lnc': self.lnc,
                     'ls': self.ls,
                     'lim_status': self.lim_status,
                     # 'n_pls': self.mineral_n_pls,
                     # 'p_pls': self.labile_p_pls,
                     'storage_pool': self.storage_pool,
                     'calendar': self.calendar,    # Calendar name
                     'time_unit': self.time_unit,   # Time unit
                     'sind': index[0],
                     'eind': index[1]}
        # Flush attrs
        self.emaxm = []
        self.tsoil = []
        self.photo = None
        self.aresp = None
        self.npp = None
        self.lai = None
        self.csoil = None
        self.inorg_n = None
        self.inorg_p = None
        self.sorbed_n = None
        self.sorbed_p = None
        self.snc = None
        self.hresp = None
        self.rcm = None
        self.f5 = None
        self.runom = None
        self.evapm = None
        self.wsoil = None
        self.rm = None
        self.rg = None
        self.cleaf = None
        self.cawood = None
        self.cfroot = None
        self.area = None
        self.wue = None
        self.cue = None
        self.cdef = None
        self.nmin = None
        self.pmin = None
        self.vcmax = None
        self.specific_la = None
        self.nupt = None
        self.pupt = None
        self.litter_l = None
        self.cwd = None
        self.litter_fr = None
        self.lnc = None
        self.storage_pool = None
        self.ls = None
        self.lim_status = None

        return to_pickle

    def init_caete_dyn(self, dt1, soil_nu, co2, pls_table, name):
        """
            dt1: tuple of np.ndarrays with climatic input data look line 440
            soil_nu: tuple of np.ndarrays with soil nutrient contents
            co2: (list) a alist (association list) with yearly cCO2 ATM data
            pls_table: np.ndarray with functional traits of a set of PLant life strategies
            name: str a name for the gridcell group"""

        assert self.filled == False, "already done"

        os.mkdir(self.out_dir)
        self.flush_data = 0

        pr, ps, rsds, tas, rhs = dt1
        nx = str(self.x)
        ny = str(self.y)
        self.name = name

        self.pr = pr['pr_' + ny + '-' + nx]['var_data'][:]
        self.ps = ps['ps_' + ny + '-' + nx]['var_data'][:]
        self.rsds = rsds['rsds_' + ny + '-' + nx]['var_data'][:]
        self.tas = tas['tas_' + ny + '-' + nx]['var_data'][:]
        self.rhs = rhs['hurs_' + ny + '-' + nx]['var_data'][:]

        assert self.pr.size == self.ps.size, 'ps is different from pr'
        assert self.pr.size == self.rsds.size, 'rsds is different from pr'
        assert self.pr.size == self.tas.size, 'tas is different from pr'
        assert self.pr.size == self.rhs.size, 'rhs is different from pr'

        # SOIL AND NUTRIENTS
        self.input_nut = []
        self.nutlist = ['tn', 'tp', 'ap', 'ip', 'op']
        for nut in soil_nu:
            self.input_nut.append(nut[self.y, self.x])
        self.soil_dict = dict(zip(self.nutlist, self.input_nut))

        # TIME
        self.calendar = pr['metadata']['calendar']
        self.time_index = pr['metadata']['time_data']
        self.time_unit = pr['metadata']['time_unit']
        self.ssize = pr['metadata']['len']
        self.sind = int(self.time_index[0])
        self.eind = int(self.time_index[-1])
        self.start_date = cftime.num2date(
            self.time_index[0], self.time_unit, calendar=self.calendar)
        self.end_date = cftime.num2date(
            self.time_index[-1], self.time_unit, calendar=self.calendar)

        # OTHER INPUTS
        self.pls_table = pls_table
        self.neighbours = neighbours_index(self.pos, mask)
        self.soil_temp = st.soil_temp_sub(self.tas[:1095] - 273.15)

        # Prepare co2 inputs (we have annually means)
        self.co2_data = co2

        # STATE
        self.wfim = np.zeros(shape=(npls,), order='F') + 0.01
        self.gfim = np.zeros(shape=(npls,), order='F')
        self.sfim = np.zeros(shape=(npls,), order='F')
        self.tsoil = []
        self.emaxm = []

        self.vp_cleaf, self.vp_croot, self.vp_cwood = m.spinup2(
            0.365242, self.pls_table)

        self.vp_dcl = np.zeros(shape=(npls,), order='F')
        self.vp_dca = np.zeros(shape=(npls,), order='F')
        self.vp_dcf = np.zeros(shape=(npls,), order='F')
        self.vp_ocp = np.zeros(shape=(npls,), order='F')
        self.vp_sto = np.zeros(shape=(3, npls), order='F')

        # # # SOIL SPINUP
        # TODO  Prepare soil nutrient data
        self.soil_spinup = False
        self.sp_csoil = np.zeros(shape=(4,), order='F') + 1.0
        self.sp_snc = np.zeros(shape=(8,), order='F')
        self.sp_available_p = self.soil_dict['ap']
        self.sp_available_n = 0.2 * self.soil_dict['tn']
        self.sp_in_n = 0.5 * self.soil_dict['tn']
        self.sp_so_n = 0.3 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']

        self.outputs = dict()
        self.filled = True

        return None

    def _save_output(self, data_obj):
        """Compress and save output data
        data_object: dict; the dict returned from _flush_output"""
        if self.run_counter < 10:
            fpath = "spin{}{}.pbz2".format(0, self.run_counter)
        else:
            fpath = "spin{}.pbz2".format(self.run_counter)
        with bz2.BZ2File(self.outputs[fpath], 'w') as fh:
            cPickle.dump(data_obj, fh)
        self.flush_data = 0

    def run_spinup(self, start_date, end_date, spinup, coupled=False):
        """ start_date [str] "yyyymmdd" Start model execution
            end_date   [str] "yyyymmdd" End model execution
            spinup     [int] Number of repetitions in spinup. 0 for no spinu
            coupled    [bool] engage the nutrients cycle

            this function run the fortran subroutines and manage data flux
            Is the proper CAETÊ-DGVM execution in the start_date - end_date period
        """

        assert self.filled, "The gridcell has no input data"

        def find_co2(year):
            for i in self.co2_data:
                if int(i.split('\t')[0]) == year:
                    return float(i.split('\t')[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)
        spin = spinup

        # Catch climatic input and make conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # transforamando de Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.5 / 2.18e5
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0

        for s in range(spin):
            self._allocate_output(steps.size)
            for step in range(steps.size):
                loop += 1
                count_days += 1
                # CAST CO2 ATM CONCENTRATION
                days = 366 if utl.leap(year0) == 1 else 365
                if count_days == days:
                    count_days = 0
                    year0 = cftime.num2date(day_indexes[step],
                                            self.time_unit, self.calendar).year
                    co2 = find_co2(year0)
                    next_year = (find_co2(year0 + 1) - co2) / days

                elif loop == 1 and count_days < days:
                    year0 = start.year
                    next_year = (find_co2(year0 + 1) - co2) / \
                        (days - count_days)

                co2 += next_year

                # Update soil temperature
                self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

                out = model.daily_budget(self.pls_table, self.wfim, self.gfim, self.sfim,
                                         self.soil_temp, temp[step], prec[step], p_atm[step],
                                         ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                         co2, self.vp_sto, self.vp_cleaf, self.vp_cwood, self.vp_croot,
                                         self.vp_dcl, self.vp_dca, self.vp_dcf)

                # Create a dict with the function output
                daily_output = catch_out_budget(out)

                # UPDATE STATE VARIABLES
                ocp = daily_output['ocpavg']
                self.ls[step] = np.sum((ocp > 0.0))

                # WATER
                water = wm(ocp, daily_output['w2'])
                ice = wm(ocp, daily_output['g2'])
                snow = wm(ocp, daily_output['s2'])

                for pls in range(npls):
                    self.wfim[pls] = water
                    self.gfim[pls] = ice
                    self.sfim[pls] = snow

                # UPDATE vegetation pools
                self.vp_cleaf = daily_output['cleafavg_pft']
                self.vp_cwood = daily_output['cawoodavg_pft']
                self.vp_croot = daily_output['cfrootavg_pft']
                self.vp_dcl = daily_output['delta_cveg'][0]
                self.vp_dca = daily_output['delta_cveg'][1]
                self.vp_dcf = daily_output['delta_cveg'][2]
                self.vp_sto = daily_output['stodbg']
                self.vp_ocp = ocp

                self.nupt[step] = wm(ocp, daily_output['nupt'])
                self.pupt[step] = wm(ocp, daily_output['pupt'])

                self.litter_l[step] = wm(ocp, daily_output['litter_l'])
                self.cwd[step] = wm(ocp, daily_output['cwd'])
                self.litter_fr[step] = wm(ocp, daily_output['litter_fr'])
                self.lnc[:, step] = wm(ocp, daily_output['lnc'])

                self.storage_pool[:, step] = wm(ocp, daily_output['stodbg'])

                s_out = soil_dec.carbon3(self.soil_temp, water / gp.wmax, self.litter_l[step],
                                         self.cwd[step], self.litter_fr[step], self.lnc[:, step],
                                         self.sp_csoil, self.sp_snc)

                soil_out = catch_out_carbon3(s_out)
                self.sp_csoil = soil_out['cs']
                self.sp_snc = soil_out['snc']

                if coupled:
                    # CALCULATE THE EQUILIBTIUM IN SOIL POOLS

                    self.sp_available_p -= self.pupt[step]
                    self.sp_available_n -= self.nupt[step]
                    self.sp_available_p += soil_out['pmin']
                    self.sp_available_n += soil_out['nmin'] + soil_dec.bnf(2.7)

                    # NUTRIENT DINAMICS
                    self.sp_in_n += self.sp_available_n + self.sp_so_n

                    self.sp_so_n = soil_dec.sorbed_n_equil(self.sp_in_n)
                    self.sp_available_n = soil_dec.solution_n_equil(
                        self.sp_in_n)
                    self.sp_in_n -= self.sp_so_n + self.sp_available_n

                    self.sp_in_p += self.sp_available_p + self.sp_so_p

                    self.sp_so_p = soil_dec.sorbed_p_equil(self.sp_in_p)
                    self.sp_available_p = soil_dec.solution_p_equil(
                        self.sp_in_p)
                    self.sp_in_p -= self.sp_so_p + self.sp_available_p

                # # # Process (cwm) & store (np.array) outputs
                self.emaxm.append(daily_output['epavg'])
                self.tsoil.append(self.soil_temp)
                self.photo[step] = wm(ocp, daily_output['phavg'])
                self.aresp[step] = wm(ocp, daily_output['aravg'])
                self.npp[step] = wm(ocp, daily_output['nppavg'])
                self.lai[step] = wm(ocp, daily_output['laiavg'])
                self.rcm[step] = wm(ocp, daily_output['rcavg'])
                self.f5[step] = wm(ocp, daily_output['f5avg'])
                self.runom[step] = wm(ocp, daily_output['ruavg'])
                self.evapm[step] = wm(ocp, daily_output['evavg'])
                self.wsoil[step] = wm(ocp, daily_output['w2'])
                self.rm[step] = wm(ocp, daily_output['rmavg'])
                self.rg[step] = wm(ocp, daily_output['rgavg'])
                self.cleaf[step] = wm(ocp, daily_output['cleafavg_pft'])
                self.cawood[step] = wm(ocp, daily_output['cawoodavg_pft'])
                self.cfroot[step] = wm(ocp, daily_output['cfrootavg_pft'])
                self.area[:, step] = ocp
                self.wue[step] = wm(ocp, daily_output['wueavg'])
                self.cue[step] = wm(ocp, daily_output['cueavg'])
                self.cdef[step] = wm(ocp, daily_output['c_defavg'])
                self.vcmax[step] = wm(ocp, daily_output['vcmax'])
                self.specific_la[step] = wm(ocp, daily_output['specific_la'])
                self.lim_status[:, :, step] = daily_output['limitation_status']
                self.hresp[step] = soil_out['hr']
                self.csoil[:, step] = soil_out['cs']
                self.inorg_n[step] = self.sp_in_n
                self.inorg_p[step] = self.sp_in_p
                self.sorbed_n[step] = self.sp_so_n
                self.sorbed_p[step] = self.sp_so_p
                self.snc[:, step] = soil_out['snc']
                self.nmin[step] = self.sp_available_n
                self.pmin[step] = self.sp_available_p

            self.flush_data = self._flush_output(
                'spin', (start_index, end_index))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                f = executor.submit(self._save_output, self.flush_data)
        return None

    def bdg_spinup(self, start_date='19810101', end_date='19821231'):
        """SPINUP VEGETATION"""

        assert self.filled, "The gridcell has no input data"
        self.budget_spinup = True

        def find_co2(year):
            for i in self.co2_data:
                if int(i.split('\t')[0]) == year:
                    return float(i.split('\t')[1].strip())

        def find_index(start, end):
            result = []
            num = np.arange(self.ssize)
            ind = np.arange(self.sind, self.eind + 1)
            for r, i in zip(num, ind):
                if i == start:
                    result.append(r)
            for r, i in zip(num, ind):
                if i == end:
                    result.append(r)
            return result

        # Define start and end dates
        start = cftime.real_datetime(int(start_date[:4]), int(
            start_date[4:6]), int(start_date[6:]))
        end = cftime.real_datetime(int(end_date[:4]), int(
            end_date[4:6]), int(end_date[6:]))
        # Check dates sanity
        assert start < end, "start > end"
        assert start >= self.start_date
        assert end <= self.end_date

        # Define time index
        start_index = int(cftime.date2num(
            start, self.time_unit, self.calendar))
        end_index = int(cftime.date2num(end, self.time_unit, self.calendar))

        lb, hb = find_index(start_index, end_index)
        steps = np.arange(lb, hb + 1)
        day_indexes = np.arange(start_index, end_index + 1)

        # Catch climatic input and make conversions
        temp = self.tas[lb: hb + 1] - 273.15  # ! K to °C
        prec = self.pr[lb: hb + 1] * 86400  # kg m-2 s-1 to  mm/day
        # transforamando de Pascal pra mbar (hPa)
        p_atm = self.ps[lb: hb + 1] * 0.01
        # W m-2 to mol m-2 s-1 ! 0.5 converts RSDS to PAR
        ipar = self.rsds[lb: hb + 1] * 0.5 / 2.18e5
        ru = self.rhs[lb: hb + 1] / 100.0

        year0 = start.year
        co2 = find_co2(year0)
        count_days = start.dayofyr - 2
        loop = 0

        wo = []
        llo = []
        cwdo = []
        rlo = []
        lnco = []
        for step in range(steps.size):
            loop += 1
            count_days += 1
            # CAST CO2 ATM CONCENTRATION
            days = 366 if utl.leap(year0) == 1 else 365
            if count_days == days:
                count_days = 0
                year0 = cftime.num2date(day_indexes[step],
                                        self.time_unit, self.calendar).year
                co2 = find_co2(year0)
                next_year = (find_co2(year0 + 1) - co2) / days

            elif loop == 1 and count_days < days:
                year0 = start.year
                next_year = (find_co2(year0 + 1) - co2) / \
                    (days - count_days)

            co2 += next_year
            self.soil_temp = st.soil_temp(self.soil_temp, temp[step])

            out = model.daily_budget(self.pls_table, self.wfim, self.gfim, self.sfim,
                                     self.soil_temp, temp[step], prec[step], p_atm[step],
                                     ipar[step], ru[step], self.sp_available_n, self.sp_available_p,
                                     co2, self.vp_sto, self.vp_cleaf, self.vp_cwood, self.vp_croot,
                                     self.vp_dcl, self.vp_dca, self.vp_dcf)

            # Create a dict with the function output
            daily_output = catch_out_budget(out)

            # UPDATE STATE VARIABLESl
            ocp = daily_output['ocpavg']

            # WATER IS KILLING EVERyONE
            water = wm(ocp, daily_output['w2'])
            ice = wm(ocp, daily_output['g2'])
            snow = wm(ocp, daily_output['s2'])

            for pls in range(npls):
                self.wfim[pls] = water
                self.gfim[pls] = ice
                self.sfim[pls] = snow

            # UPDATE vegetation pools
            self.vp_cleaf = daily_output['cleafavg_pft']
            self.vp_cwood = daily_output['cawoodavg_pft']
            self.vp_croot = daily_output['cfrootavg_pft']
            self.vp_dcl = daily_output['delta_cveg'][0]
            self.vp_dca = daily_output['delta_cveg'][1]
            self.vp_dcf = daily_output['delta_cveg'][2]
            self.vp_sto = daily_output['stodbg']
            self.vp_ocp = ocp

            wo.append(wm(ocp, daily_output['w2']))
            llo.append(wm(ocp, daily_output['litter_l']))
            cwdo.append(wm(ocp, daily_output['cwd']))
            rlo.append(wm(ocp, daily_output['litter_fr']))
            lnco.append(wm(ocp, daily_output['lnc']))

        return (np.array(wo[365:])).mean(), (np.array(llo[365:])).mean(),\
               (np.array(cwdo[365:])).mean(), (np.array(rlo[365:])).mean(),\
               (np.array(lnco)[:365, :]).mean(axis=0,)

    def sdc_spinup(self, water, ll, cwd, rl, lnc):
        """SOIL POOLS SPINUP"""

        for x in range(100000):

            s_out = soil_dec.carbon3(self.soil_temp, water / gp.wmax, ll, cwd, rl, lnc,
                                     self.sp_csoil, self.sp_snc)

            soil_out = catch_out_carbon3(s_out)
            self.sp_csoil = soil_out['cs']
            self.sp_snc = soil_out['snc']

        self.sp_in_n = 0.5 * self.soil_dict['tn']
        self.sp_so_n = 0.3 * self.soil_dict['tn']
        self.sp_so_p = self.soil_dict['tp'] - sum(self.input_nut[2:])
        self.sp_in_p = self.soil_dict['ip']
