#   Copyright 2017- LabTerra

#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.)

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.


# template_tables.py
# Templates for tables of CAETÊ
from caete_module import global_par as gp
import tables as tb


__author__ = "JPDarela"

# Group 1
G1_1d = ['emaxm',
         'tsoil',
         'photo',
         'aresp',
         'npp',
         'lai',
         'f5',
         'rm',
         'rg',
         'wue',
         'cue',
         'cdef',
         'vcmax',
         'specific_la',
         'ls']

G1_2d = ['nupt', ]

G1_3d = ['pupt', ]

G1 = G1_1d + G1_2d + G1_3d

# Group 2
G2_1d = ['inorg_n',
         'inorg_p',
         'sorbed_n',
         'sorbed_p',
         'hresp',
         'nmin',
         'pmin']

G2_4d = ['csoil', ]

G2_8d = ['snc', ]

G2 = G2_1d + G2_4d + G2_8d

# Group 3
G3_1d = ["rcm",
         "runom",
         "evapm",
         "wsoil",
         "swsoil",
         "cleaf",
         "cawood",
         "cfroot",
         "litter_l",
         "cwd",
         "litter_fr",
         "c_cost"]

G3_3d = ['storage_pool', ]

G3_6d = ["lnc", ]

G3 = G3_1d + G3_3d + G3_6d

PLS_head = ['PLS_id', 'g1', 'resopfrac', 'tleaf', 'twood', 'troot', 'aleaf', 'awood', 'aroot', 'c4',
            'leaf_n2c', 'awood_n2c', 'froot_n2c', 'leaf_p2c', 'awood_p2c', 'froot_p2c',
            'amp', 'pdia']


class run_g1(tb.IsDescription):
    """ Row template for CAETÊ output data"""
    # ID
    row_id = tb.Int64Col(dflt=0, pos=0)
    date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=1)
    grid_y = tb.Int16Col(dflt=0, pos=2)
    grid_x = tb.Int16Col(dflt=0, pos=3)
    # Fluxees
    emaxm = tb.Float32Col(dflt=0.0, pos=4)
    tsoil = tb.Float32Col(dflt=0.0, pos=5)
    photo = tb.Float32Col(dflt=0.0, pos=6)
    aresp = tb.Float32Col(dflt=0.0, pos=7)
    npp = tb.Float32Col(dflt=0.0, pos=8)
    lai = tb.Float32Col(dflt=0.0, pos=9)
    f5 = tb.Float32Col(dflt=0.0, pos=10)
    rm = tb.Float32Col(dflt=0.0, pos=11)
    rg = tb.Float32Col(dflt=0.0, pos=12)
    wue = tb.Float32Col(dflt=0.0, pos=13)
    cue = tb.Float32Col(dflt=0.0, pos=14)
    cdef = tb.Float32Col(dflt=0.0, pos=15)
    vcmax = tb.Float32Col(dflt=0.0, pos=16)
    specific_la = tb.Float32Col(dflt=0.0, pos=17)
    ls = tb.Float32Col(dflt=0.0, pos=18)
    nupt1 = tb.Float32Col(dflt=0.0, pos=19)
    nupt2 = tb.Float32Col(dflt=0.0, pos=20)
    pupt1 = tb.Float32Col(dflt=0.0, pos=21)
    pupt2 = tb.Float32Col(dflt=0.0, pos=22)
    pupt3 = tb.Float32Col(dflt=0.0, pos=23)


class run_g2(tb.IsDescription):
    """ Row template for CAETÊ output data"""
    # ID
    row_id = tb.Int64Col(dflt=0, pos=0)
    date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=1)
    grid_y = tb.Int16Col(dflt=0, pos=2)
    grid_x = tb.Int16Col(dflt=0, pos=3)

    csoil1 = tb.Float32Col(dflt=0.0, pos=4)
    csoil2 = tb.Float32Col(dflt=0.0, pos=5)
    csoil3 = tb.Float32Col(dflt=0.0, pos=6)
    csoil4 = tb.Float32Col(dflt=0.0, pos=7)
    sncN1 = tb.Float32Col(dflt=0.0, pos=8)
    sncN2 = tb.Float32Col(dflt=0.0, pos=8)
    sncN3 = tb.Float32Col(dflt=0.0, pos=10)
    sncN4 = tb.Float32Col(dflt=0.0, pos=11)
    sncP1 = tb.Float32Col(dflt=0.0, pos=12)
    sncP2 = tb.Float32Col(dflt=0.0, pos=13)
    sncP3 = tb.Float32Col(dflt=0.0, pos=14)
    sncP4 = tb.Float32Col(dflt=0.0, pos=15)
    inorg_n = tb.Float32Col(dflt=0.0, pos=16)
    inorg_p = tb.Float32Col(dflt=0.0, pos=17)
    sorbed_n = tb.Float32Col(dflt=0.0, pos=18)
    sorbed_p = tb.Float32Col(dflt=0.0, pos=19)
    hresp = tb.Float32Col(dflt=0.0, pos=20)
    nmin = tb.Float32Col(dflt=0.0, pos=21)
    pmin = tb.Float32Col(dflt=0.0, pos=22)


class run_g3(tb.IsDescription):
    """ Row template for CAETÊ output data"""
    # ID
    row_id = tb.Int64Col(dflt=0, pos=0)
    date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=1)
    grid_y = tb.Int16Col(dflt=0, pos=2)
    grid_x = tb.Int16Col(dflt=0, pos=3)

    rcm = tb.Float32Col(dflt=0.0, pos=4)
    runom = tb.Float32Col(dflt=0.0, pos=5)
    evapm = tb.Float32Col(dflt=0.0, pos=6)
    wsoil = tb.Float32Col(dflt=0.0, pos=7)
    swsoil = tb.Float32Col(dflt=0.0, pos=8)
    cleaf = tb.Float32Col(dflt=0.0, pos=9)
    cawood = tb.Float32Col(dflt=0.0, pos=10)
    cfroot = tb.Float32Col(dflt=0.0, pos=11)
    litter_l = tb.Float32Col(dflt=0.0, pos=12)
    cwd = tb.Float32Col(dflt=0.0, pos=13)
    litter_fr = tb.Float32Col(dflt=0.0, pos=14)
    lnc1 = tb.Float32Col(dflt=0.0, pos=15)
    lnc2 = tb.Float32Col(dflt=0.0, pos=16)
    lnc3 = tb.Float32Col(dflt=0.0, pos=17)
    lnc4 = tb.Float32Col(dflt=0.0, pos=18)
    lnc5 = tb.Float32Col(dflt=0.0, pos=19)
    lnc6 = tb.Float32Col(dflt=0.0, pos=20)
    sto1 = tb.Float32Col(dflt=0.0, pos=21)
    sto2 = tb.Float32Col(dflt=0.0, pos=22)
    sto3 = tb.Float32Col(dflt=0.0, pos=23)
    c_cost = tb.Float32Col(dflt=0.0, pos=24)


class PLS_temp(tb.IsDescription):
    """PLS table row template"""
    PLS_id = tb.Int16Col(dflt=0, pos=0)
    g1 = tb.Float64Col(dflt=0.0, pos=1)
    resopfrac = tb.Float64Col(dflt=0.0, pos=2)
    tleaf = tb.Float32Col(dflt=0.0, pos=3)
    twood = tb.Float32Col(dflt=0.0, pos=4)
    troot = tb.Float32Col(dflt=0.0, pos=5)
    aleaf = tb.Float32Col(dflt=0.0, pos=6)
    awood = tb.Float32Col(dflt=0.0, pos=7)
    aroot = tb.Float32Col(dflt=0.0, pos=8)
    c4 = tb.UInt8Col(dflt=0, pos=9)
    leaf_n2c = tb.Float32Col(dflt=0.0, pos=10)
    awood_n2c = tb.Float32Col(dflt=0.0, pos=11)
    froot_n2c = tb.Float32Col(dflt=0.0, pos=12)
    leaf_p2c = tb.Float32Col(dflt=0.0, pos=13)
    awood_p2c = tb.Float32Col(dflt=0.0, pos=14)
    froot_p2c = tb.Float32Col(dflt=0.0, pos=15)
    amp = tb.Float32Col(dflt=0.0, pos=16)
    pdia = tb.Float32Col(dflt=0.0, pos=17)


class spin_snapshots(tb.IsDescription):

    row_id = tb.Int64Col(dflt=0, pos=0)
    start_date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=2)
    end_date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=3)
    grid_y = tb.Int16Col(dflt=0, pos=4)
    grid_x = tb.Int16Col(dflt=0, pos=5)

    leaf_nolim = tb.Float32Col(dflt=0.0, pos=6)
    leaf_lim_n = tb.Float32Col(dflt=0.0, pos=7)
    leaf_lim_p = tb.Float32Col(dflt=0.0, pos=8)
    leaf_colim_n = tb.Float32Col(dflt=0.0, pos=9)
    leaf_colim_p = tb.Float32Col(dflt=0.0, pos=10)
    leaf_colim_np = tb.Float32Col(dflt=0.0, pos=11)
    wood_nolim = tb.Float32Col(dflt=0.0, pos=12)
    wood_lim_n = tb.Float32Col(dflt=0.0, pos=13)
    wood_lim_p = tb.Float32Col(dflt=0.0, pos=14)
    wood_colim_n = tb.Float32Col(dflt=0.0, pos=15)
    wood_colim_p = tb.Float32Col(dflt=0.0, pos=16)
    wood_colim_np = tb.Float32Col(dflt=0.0, pos=17)
    root_nolim = tb.Float32Col(dflt=0.0, pos=18)
    root_lim_n = tb.Float32Col(dflt=0.0, pos=19)
    root_lim_p = tb.Float32Col(dflt=0.0, pos=20)
    root_colim_n = tb.Float32Col(dflt=0.0, pos=21)
    root_colim_p = tb.Float32Col(dflt=0.0, pos=22)
    root_colim_np = tb.Float32Col(dflt=0.0, pos=23)

    upt_stratN0 = tb.Float32Col(dflt=0.0, pos=24)
    upt_stratN1 = tb.Float32Col(dflt=0.0, pos=25)
    upt_stratN2 = tb.Float32Col(dflt=0.0, pos=26)
    upt_stratN3 = tb.Float32Col(dflt=0.0, pos=27)
    upt_stratN4 = tb.Float32Col(dflt=0.0, pos=28)
    upt_stratN6 = tb.Float32Col(dflt=0.0, pos=29)
    upt_stratP0 = tb.Float32Col(dflt=0.0, pos=30)
    upt_stratP1 = tb.Float32Col(dflt=0.0, pos=31)
    upt_stratP2 = tb.Float32Col(dflt=0.0, pos=32)
    upt_stratP3 = tb.Float32Col(dflt=0.0, pos=33)
    upt_stratP4 = tb.Float32Col(dflt=0.0, pos=34)
    upt_stratP5 = tb.Float32Col(dflt=0.0, pos=35)
    upt_stratP6 = tb.Float32Col(dflt=0.0, pos=36)
    upt_stratP7 = tb.Float32Col(dflt=0.0, pos=37)
    upt_stratP8 = tb.Float32Col(dflt=0.0, pos=38)

    area_0 = tb.Float32Col(shape=(gp.npls,), pos=39)
    area_f = tb.Float32Col(shape=(gp.npls,), pos=40)
