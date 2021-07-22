# create_plsbin.py
import caete_module
import plsgen as pls
import numpy as np

txt = pls.table_gen(caete_module.global_par.npls)
np.savetxt("pls_ex.txt", txt.T)
