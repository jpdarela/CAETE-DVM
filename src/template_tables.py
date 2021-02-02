# template_tables.py
# Templates for tables
import tables as tb


def write_output(dt):
    pass


def write_PLSTable(fh):
    pass


class run(tb.IsDescription):
    """ Row template for CAETÊ output data"""
    # ID
    row_id = tb.Int64Col(dflt=0, pos=0)
    date = tb.StringCol(itemsize=8, dflt="yyyymmdd", pos=1)
    grid_y = tb.Int16Col(dflt=0, pos=2)
    grid_x = tb.Int16Col(dflt=0, pos=3)
    # Data
    emaxm = tb.Float64Col(dflt=0.0, pos=4)
    tsoil = tb.Float64Col(dflt=0.0, pos=5)
    photo = tb.Float64Col(dflt=0.0, pos=6)
    aresp = tb.Float64Col(dflt=0.0, pos=7)
    npp = tb.Float64Col(dflt=0.0, pos=8)
    lai = tb.Float64Col(dflt=0.0, pos=9)
    csoil1 = tb.Float64Col(dflt=0.0, pos=10)
    csoil2 = tb.Float64Col(dflt=0.0, pos=11)
    csoil3 = tb.Float64Col(dflt=0.0, pos=12)
    csoil4 = tb.Float64Col(dflt=0.0, pos=13)
    inorg_n = tb.Float64Col(dflt=0.0, pos=14)
    inorg_p = tb.Float64Col(dflt=0.0, pos=15)
    sorbed_n = tb.Float64Col(dflt=0.0, pos=16)
    sorbed_p = tb.Float64Col(dflt=0.0, pos=17)
    sncN1 = tb.Float64Col(dflt=0.0, pos=18)
    sncN2 = tb.Float64Col(dflt=0.0, pos=19)
    sncN3 = tb.Float64Col(dflt=0.0, pos=20)
    sncN4 = tb.Float64Col(dflt=0.0, pos=21)
    sncp1 = tb.Float64Col(dflt=0.0, pos=22)
    sncp2 = tb.Float64Col(dflt=0.0, pos=23)
    sncp3 = tb.Float64Col(dflt=0.0, pos=24)
    sncp4 = tb.Float64Col(dflt=0.0, pos=25)
    hresp = tb.Float64Col(dflt=0.0, pos=26)
    rcm = tb.Float64Col(dflt=0.0, pos=27)
    f5 = tb.Float64Col(dflt=0.0, pos=28)
    runom = tb.Float64Col(dflt=0.0, pos=29)
    evapm = tb.Float64Col(dflt=0.0, pos=30)
    wsoil = tb.Float64Col(dflt=0.0, pos=31)
    swsoil = tb.Float64Col(dflt=0.0, pos=32)
    rm = tb.Float64Col(dflt=0.0, pos=33)
    rg = tb.Float64Col(dflt=0.0, pos=34)
    cleaf = tb.Float64Col(dflt=0.0, pos=35)
    cawood = tb.Float64Col(dflt=0.0, pos=36)
    cfroot = tb.Float64Col(dflt=0.0, pos=37)
    wue = tb.Float64Col(dflt=0.0, pos=38)
    cue = tb.Float64Col(dflt=0.0, pos=39)
    cdef = tb.Float64Col(dflt=0.0, pos=40)
    nmin = tb.Float64Col(dflt=0.0, pos=41)
    pmin = tb.Float64Col(dflt=0.0, pos=42)
    vcmax = tb.Float64Col(dflt=0.0, pos=43)
    sla = tb.Float64Col(dflt=0.0, pos=44)
    nupt1 = tb.Float64Col(dflt=0.0, pos=45)
    nupt2 = tb.Float64Col(dflt=0.0, pos=46)
    pupt1 = tb.Float64Col(dflt=0.0, pos=47)
    pupt2 = tb.Float64Col(dflt=0.0, pos=48)
    pupt3 = tb.Float64Col(dflt=0.0, pos=49)
    litter_l = tb.Float64Col(dflt=0.0, pos=50)
    cwd = tb.Float64Col(dflt=0.0, pos=51)
    litter_fr = tb.Float64Col(dflt=0.0, pos=52)
    lnc1 = tb.Float64Col(dflt=0.0, pos=53)
    lnc2 = tb.Float64Col(dflt=0.0, pos=54)
    lnc3 = tb.Float64Col(dflt=0.0, pos=55)
    lnc4 = tb.Float64Col(dflt=0.0, pos=56)
    lnc5 = tb.Float64Col(dflt=0.0, pos=57)
    lnc6 = tb.Float64Col(dflt=0.0, pos=58)
    sto1 = tb.Float64Col(dflt=0.0, pos=59)
    sto2 = tb.Float64Col(dflt=0.0, pos=60)
    sto3 = tb.Float64Col(dflt=0.0, pos=61)
    c_cost = tb.Float64Col(dflt=0.0, pos=62)


class PLS_table(tb.IsDescription):
    """PLS table row template"""
    g1 = tb.Float64Col(dflt=0.0, pos=1)
    resopfrac = tb.Float64Col(dflt=0.0, pos=2)
    tleaf = tb.Float64Col(dflt=0.0, pos=3)
    twood = tb.Float64Col(dflt=0.0, pos=4)
    troot = tb.Float64Col(dflt=0.0, pos=5)
    aleaf = tb.Float64Col(dflt=0.0, pos=6)
    awood = tb.Float64Col(dflt=0.0, pos=7)
    aroot = tb.Float64Col(dflt=0.0, pos=8)
    c4 = tb.UInt8Col(dflt=0, pos=9)
    leaf_n2c = tb.Float64Col(dflt=0.0, pos=10)
    awood_n2c = tb.Float64Col(dflt=0.0, pos=11)
    froot_n2c = tb.Float64Col(dflt=0.0, pos=12)
    leaf_p2c = tb.Float64Col(dflt=0.0, pos=13)
    awood_p2c = tb.Float64Col(dflt=0.0, pos=14)
    froot_p2c = tb.Float64Col(dflt=0.0, pos=15)
    amp = tb.Float64Col(dflt=0.0, pos=16)
    pdia = tb.Float64Col(dflt=0.0, pos=17)


class ocp_snapshots:
    pass
# vai salvar a area/ocp para cada pls entre o inicio e o fim de uma chamada de run_caete
# tb limitação e uptake

# init_date & end_date = str"yyyymmdd"

# nolim  (% of the simulated time)
    # LEAF/WOOD/ROOT
# lim_n  (% of the simulated time)
    # LEAF/WOOD/ROOT
# lim_p  (% of the simulated time)
    # LEAF/WOOD/ROOT
# colim_n  (% of the simulated time)
    # LEAF/WOOD/ROOT
# colim_p  (% of the simulated time)
    # LEAF/WOOD/ROOT
# colim_np (% of the simulated time)
    # LEAF/WOOD/ROOT

# upt_stratN1 (% of the simulated time)
# upt_stratN2 (% of the simulated time)
# upt_stratP1 (% of the simulated time)
# upt_stratP2 (% of the simulated time)
# upt_stratP3 (% of the simulated time)
  # 3
  # subclass_area: init_area[npls]
  #                endarea[npls]
