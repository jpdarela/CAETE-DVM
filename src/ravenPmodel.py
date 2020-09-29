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

start = 0.1
plab = np.linspace(start, 8, 1000) * 0.003
sop = np.linspace(start, 25, 1000) * 0.003
op = np.linspace(start, 6, 1000) * 0.003

amp = 0.94
ecm = 1 - amp
croot = 1200.0

# compare P uptake costs
for x in range(1000):
    out0 = ccp(amp, plab[x], sop[x], op[x], croot)
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
ax.plot(range(1000), out1, color='tab:blue')
ax.plot(range(1000), out2, color='tab:green')
ax.plot(range(1000), out3, color='tab:gray')
ax.plot(range(1000), out4, color='tab:olive')
ax.plot(range(1000), out5, color='tab:red')
ax.plot(range(1000), out6, color='tab:cyan')
ax.plot(range(1000), out7, color='tab:orange')
ax.plot(range(1000), out8, color='yellow')
plt.legend(['nmam', 'nmem', 'am', 'em', 'ramAP', 'remAP', 'AMAP', 'EM0'])
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
    out0 = ccn(amp, nmin[x], on[x], croot)
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
ax.plot(range(1000), out1, color='tab:blue')
ax.plot(range(1000), out2, color='tab:green')
ax.plot(range(1000), out3, color='tab:gray')
ax.plot(range(1000), out4, color='tab:olive')
ax.plot(range(1000), out5, color='tab:red')
ax.plot(range(1000), out6, color='tab:cyan')
plt.legend(['nmam', 'nmem', 'am', 'em', 'ramNA', 'remNA'])
plt.show()
