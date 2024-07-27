import os
import sys
import tomllib as tl


class Config:
    def __init__(self, d=None) -> None:
        if d is not None:
            self.__dict__ = d
            for k, v in d.items():
                setattr(self, k, v)

def get_parameters(config):
    """ Get parameters from the pls_gen.toml file """

    with open(config, 'rb') as f:
        data = tl.load(f)
    return data


# path to the fortran compiler dlls, used in windows systems.
fortran_compiler_dlls = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.1\bin"

if sys.platform == "win32":
      try:
         os.add_dll_directory(fortran_compiler_dlls)
      except:
         raise ImportError("Could not add the DLL directory to the PATH")

NO_DATA = [-9999.0, -9999.0]


run_breaks_hist = [('19790101', '19801231'),
                   ('19810101', '19821231'),
                   ('19830101', '19841231'),
                   ('19850101', '19861231'),
                   ('19870101', '19881231'),
                   ('19890101', '19901231'),
                   ('19910101', '19921231'),
                   ('19930101', '19941231'),
                   ('19950101', '19961231'),
                   ('19970101', '19981231'),
                   ('19990101', '20001231'),
                   ('20010101', '20021231'),
                   ('20030101', '20041231'),
                   ('20050101', '20061231'),
                   ('20070101', '20081231'),
                   ('20090101', '20101231'),
                   ('20110101', '20121231'),
                   ('20130101', '20141231'),
                   ('20150101', '20161231')]

run_breaks_CMIP5_hist = [('19300101', '19391231'),
                        ('19400101', '19491231'),
                        ('19500101', '19591231'),
                        ('19600101', '19691231'),
                        ('19700101', '19791231'),
                        ('19800101', '19891231'),
                        ('19900101', '19991231'),
                        ('20000101', '20051231')]

run_breaks_CMIP5_proj = [('20060101', '20091231'),
                         ('20100101', '20191231'),
                         ('20200101', '20291231'),
                         ('20300101', '20391231'),
                         ('20400101', '20491231'),
                         ('20500101', '20591231'),
                         ('20600101', '20691231'),
                         ('20700101', '20791231'),
                         ('20800101', '20891231'),
                         ('20900101', '20991231')]

# historical and projection periods respectively
rbrk = [run_breaks_hist, run_breaks_CMIP5_hist, run_breaks_CMIP5_proj]

