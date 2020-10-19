# ravenPmodel.py


# adapt fun logic to the different P pools

import numpy as np
import matplotlib.pyplot as plt
from cc import carbon_costs as cc

ccp = cc.active_costp
ccn = cc.active_costn


# For P
out1 = np.zeros(1000,)
out2 = np.zeros(1000,)
out3 = np.zeros(1000,)
out4 = np.zeros(1000,)
out5 = np.zeros(1000,)
out6 = np.zeros(1000,)
out7 = np.zeros(1000,)
out8 = np.zeros(1000,)

# P
start = 0.001
plab = np.linspace(start, 100, 1000) * 0.003
sop = 15  # np.linspace(start, 25, 1000) * 0.003
op = 57  # np.linspace(start, 6, 1000) * 0.003


nmin = np.linspace(start, 900, 1000) * 0.0045
on = 70  # np.linspace(start, 200, 1000) * 0.0045


amp = 0.16
ecm = 1 - amp
croot = 800.0

string_to = (amp * 100, croot / 1000, on, sop, op)


# PLant only reach a fraction of nutrients
sop *= 0.003
op *= 0.003
on *= 0.0045
# compare P uptake costs
for x in range(1000):
    out0 = ccp(amp, plab[x], sop, op, croot)
    out1[x] = out0[0]
    out2[x] = out0[1]
    # Active uptake via Mycorrhiza
    out3[x] = out0[2]
    out4[x] = out0[3]

    # AM/ AP ACTIVITY on organic P
    out5[x] = out0[4]  # ROOT AM AP
    out6[x] = out0[5]  # ROOT EM AP

    out7[x] = out0[6]   # AM AP activity
    out8[x] = out0[7]  # EM exudate Oxalate

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(1000), out1, color='navy')
ax.plot(range(1000), out2, color='darkorange')
ax.plot(range(1000), out3, color='royalblue')
ax.plot(range(1000), out4, color='r')
ax.plot(range(1000), out5, color='blueviolet')
ax.plot(range(1000), out6, color='maroon')
ax.plot(range(1000), out7, color='green')
ax.plot(range(1000), out8, color='yellow')
ax.set_xticks([20, 900])
ax.set_xticklabels(['Low', 'High'])
ax.set_xlabel('Soluble P')
ax.set_ylabel("g(C)g(P)⁻¹")
plt.legend(['NM-AM', 'NM-ECM', 'AM', 'EM',
            'R-AM-AP', 'R-ECM-AP', 'AM-AP', 'EM-EX'])
plt.yscale('log')
plt.show()


nmin = np.linspace(start, 200, 1000) * 0.0045
on = np.linspace(start, 200, 1000) * 0.0045

out1 = np.zeros(1000,)
out2 = np.zeros(1000,)
out3 = np.zeros(1000,)
out4 = np.zeros(1000,)
out5 = np.zeros(1000,)
out6 = np.zeros(1000,)


# compare N uptake costs
for x in range(1000):
    out0 = ccn(amp, nmin[x], on, croot)
    out1[x] = out0[0]
    out2[x] = out0[1]
    # Active uptake via Mycorrhiza
    out3[x] = out0[2]
    out4[x] = out0[3]

    # AM/ AP ACTIVITY on organic P
    out5[x] = out0[4]  # ROOT AM AP
    out6[x] = out0[5]  # ROOT EM AP

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(range(1000), out1, color='navy')
ax.plot(range(1000), out2, color='darkorange')
ax.plot(range(1000), out3, color='royalblue')
ax.plot(range(1000), out4, color='r')
ax.plot(range(1000), out5, color='blueviolet')
ax.plot(range(1000), out6, color='maroon')
ax.set_xticks([20, 900])
ax.set_xticklabels(['Low', 'High'])
ax.set_xlabel('Soluble N')
ax.set_ylabel("g(C)g(N)⁻¹")
text = 'AMP = {0}%\nRoot C = {1} kg(C)m⁻²\noN = {2} gm⁻²\nSoP = {3} gm⁻²\noP = {4} gm⁻²'.format(
    *string_to)
#ax.text(-25, 1 / 3000, text)
print(text)
plt.legend(['NM-AM', 'NM-ECM', 'AM', 'ECM', 'R-AM-NA', 'R-ECM-NA'])
plt.yscale('log')
plt.show()
