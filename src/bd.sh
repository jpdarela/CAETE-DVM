f2py -h cc.pyf -m cc global.f90 utils.f90 cc.f90 funcs.f90 allocation.f90 --overwrite-signature
f2py -c cc.pyf global.f90 utils.f90 cc.f90 funcs.f90 allocation.f90
