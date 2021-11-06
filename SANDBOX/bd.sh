f2py3.8 -h cc.pyf -m cc global.f90 utils.f90 cc.f90 funcs.f90 allocation.f90 --overwrite-signature
f2py3.8 -c cc.pyf global.f90 utils.f90 cc.f90 funcs.f90 allocation.f90
