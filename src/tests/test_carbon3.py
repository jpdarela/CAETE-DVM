
import sys
# sys.path.insert(0, '../')
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caete_import import *
carbon3 = soil_dec.carbon3

def getout(out):

    lst = ["cs_out", "snc", "hr", "nmin", "pmin"]
    return dict(zip(lst, out))

ts = 23.0
ws = 0.7

ll = 1.0
cwd = 0.50
rl = 1.0


lnc = np.array([4.0e-02, 1.0e-02, 2.0e-03,
                5.0e-03, 4.0e-03, 1.0e-04])


snc = np.array([0.0001, 0.0001, 0.0001, 0.0001,
                0.0001, 0.0001, 0.0001, 0.0001])
cs = np.array([1.0, 1.0, 1.0, 1.0])


header = ['hr', 'nmin', 'pmin', 'cs_out 1', 'cs_out 2', 'cs_out 3', 'cs_out 4',
          'snc 1', 'snc 2', 'snc 3', 'snc 4', 'snc 5', 'snc 6', 'snc 7', 'snc 8']

a = int(input("loops: "))
with open('carbon3_test.csv', 'w') as fh:
    CSV_WRITER = csv.writer(fh, delimiter=',', lineterminator='\n')
    CSV_WRITER.writerow(header)

    for x in range(a):
        data = getout(carbon3(ts, ws, ll, cwd, rl, lnc, cs, snc))

        snc = data['snc']
        cs = data['cs_out']

        line = [data['hr'],
                data['nmin'],
                data['pmin'],
                data['cs_out'][0],
                data['cs_out'][1],
                data['cs_out'][2],
                data['cs_out'][3],
                data['snc'][0],
                data['snc'][1],
                data['snc'][2],
                data['snc'][3],
                data['snc'][4],
                data['snc'][5],
                data['snc'][6],
                data['snc'][7]]
        CSV_WRITER.writerow(line)

data = pd.read_csv('carbon3_test.csv')


print("NMIN: {} | LNC: {}".format(data.nmin.__array__()[-1], lnc[:3].sum()))
print("PMIN: {} | LNC: {}".format(data.pmin.__array__()[-1], lnc[3:].sum()))

# # def reset_soil():

# #     funcs.sp_snr = zeroes(8) + 0.001
# #     funcs.sp_csoil[0:2] = zeroes(2)
# #     funcs.sp_csoil[2:] = zeroes(2)
# #     funcs.sp_available_n = 3
# #     funcs.sp_available_p = 2
# #     funcs.sp_in_n = 0.0
# #     funcs.sp_so_n = 0.0
# #     funcs.sp_in_p = 0.0
# #     funcs.sp_so_p = 0.0


# # reset_soil()


# def est_uptake(lnr, ll, cwd, rl, nt='n'):
#     upt = 0

#     if nt == 'n':
#         data = lnr[0:3]
#     else:
#         data = lnr[3:]

#     upt = (np.array([ll, rl, cwd]) * data).sum() * \
#         (np.random.randint(0, 10) / 10)

#     return upt


# def catch_out_carbon3(out):
#     lst = ['avail_p', 'avail_n', 'inorg_n', 'sorbed_n', 'inorg_p',
#            'sorbed_p', 'cl', 'cs', 'snr', 'hr', 'nmin', 'pmin']

#     return dict(zip(lst, out))


# with open('carbon3_test.csv', 'w') as fh:
#     CSV_WRITER = csv.writer(fh, delimiter=',')
#     CSV_WRITER.writerow(header)
#     # start inputs
#     # lnr = zeroes(6) + 0.01 * np.random.randint(0, 2)

#   # * 0.37

#     for x in range(50000):
#         lnr_std = zeroes(6) + 0.01  # * (np.random.randint(0, 10) / 10)
#         lnr_std[3:] = lnr_std[3:] * 0.8

#         leaf_litter = 0.1 * (np.random.randint(0, 10) / 10)
#         root_litter = 0.1 * (np.random.randint(0, 10) / 10)
#         cwd = 0.1 * (np.random.randint(0, 10) / 10)

#         nupt = est_uptake(lnr_std, leaf_litter,
#                           root_litter, cwd, 'n')  # * 0.30
#         pupt = est_uptake(lnr_std, leaf_litter, root_litter, cwd, 'p')
#         snr_in = funcs.sp_snr
#         avail_p = funcs.sp_available_p
#         avail_n = funcs.sp_available_n
#         inorg_n = funcs.sp_in_n
#         sorbed_n = funcs.sp_so_n
#         inorg_p = funcs.sp_in_p
#         sorbed_p = funcs.sp_so_p
#         cl = funcs.sp_csoil[0:2]
#         cs = funcs.sp_csoil[2:]

# #  subroutine carbon3(tsoil, water_sat, leaf_litter, coarse_wd,&
# #                     &        root_litter, lnr, cl, cs, snr_in,&
# #                     &        avail_p, avail_n, inorg_n, sorbed_n ,inorg_p, sorbed_p,&


# #                     &        avail_p_out, avail_n_out, inorg_n_out,sorbed_n_out,&
# #                     &        inorg_p_out, sorbed_p_out, cl_out, cs_out, snr, hr, nmin, pmin)

#         df = catch_out_carbon3(carbon3.carbon3(20, 0.7, leaf_litter, cwd, root_litter, lnr_std, cl, cs,
#                                                snr_in, avail_p, avail_n, inorg_n, sorbed_n, inorg_p, sorbed_p))

#         funcs.sp_snr = df['snr'][:]
#         funcs.sp_csoil[0:2] = df['cl'][:]
#         funcs.sp_csoil[2:] = df['cs'][:]

#         funcs.sp_available_n = df['avail_n'] - nupt
#         funcs.sp_available_p = df['avail_p'] - pupt

#         funcs.sp_in_n = df['inorg_n']
#         funcs.sp_so_n = df['sorbed_n']
#         funcs.sp_in_p = df['inorg_p']
#         funcs.sp_so_p = df['sorbed_p']

#         #print("py ", df['nmin'], nupt, df['hr'])
#         # print(df)
#         line = [df['hr'],
#                 df['inorg_n'],
#                 df['avail_p'],
#                 df['inorg_p'],
#                 df['sorbed_p'],
#                 df['cl'][0],
#                 df['cl'][1],
#                 df['cs'][0],
#                 df['cs'][1],
#                 df['snr'][0],
#                 df['snr'][1],
#                 df['snr'][2],
#                 df['snr'][3],
#                 df['snr'][4],
#                 df['snr'][5],
#                 df['snr'][6],
#                 df['snr'][7]]
#         CSV_WRITER.writerow(line)
#         # # update globl pools


data = pd.read_csv('carbon3_test.csv')
os.system('rm -rf carbon3_test.csv')

dt = data.iloc(1)

n1 = 'n2c_inc_litter_Q.png'
n2 = 'p2c_inc_litter_Q.png'
n3 = 'in_nut_inc_litter_Q.png'
n4 = 'cpools_inc_litter_Q.png'
n5 = 'in_nut_comp_inc_litter_Q.png'

dt[9:13].plot(cmap=plt.get_cmap('cool'))
plt.xlabel('Iterations')
plt.ylabel('g(N) g(C)⁻¹')
# plt.savefig(n1, dpi=700)
plt.show()

dt[13:].plot(cmap=plt.get_cmap('rainbow'))
plt.xlabel('Iterations')
plt.ylabel('g(P) g(C)⁻¹')
# plt.savefig(n2, dpi=700)
plt.show()

# .rolling(1000).mean()
dt[1:5].plot(cmap=plt.get_cmap('cividis'))
plt.xlabel('Iterations')
plt.ylabel('g(Nutrient) m⁻²')
# plt.savefig(n3, dpi=700)
plt.show()

dt[5:9:].plot(cmap=plt.get_cmap('cividis'))
plt.xlabel('Iterations')
plt.ylabel('g(C) m⁻²')
# plt.savefig(n4, dpi=700)
plt.show()

dt[1:5].rolling(1000).mean().plot(
    subplots=True, figsize=(6, 6), cmap=plt.get_cmap('viridis'))
# plt.savefig(n5, dpi=700)
plt.show()
