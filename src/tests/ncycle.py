"""
A more detailed N cycle module for CAETE, inspired by CABLE2.0.
Implements explicit NH4, NO3, DON, and microbial pools, and process-based rates.
"""
import numpy as np
import matplotlib.pyplot as plt

def arr_clip(x):
    return np.maximum(x, 0.0)

class NCycleCABLE:
    def __init__(self, nh4=1.0, no3=1.0, don=0.1, mbn=0.5, soil_temp=20.0, soil_moist=0.3, dt=1.0):
        self.nh4 = nh4  # ammonium pool (gN/m2)
        self.no3 = no3  # nitrate pool (gN/m2)
        self.don = don  # dissolved organic N (gN/m2)
        self.mbn = mbn  # microbial biomass N (gN/m2)
        self.soil_temp = soil_temp
        self.soil_moist = soil_moist
        self.dt = dt  # timestep (days)

    def temperature_scalar(self, t):
        return np.exp(0.1 * (t - 20.0))

    def moisture_scalar(self, w):
        return min(1.0, w / 0.3)

    def mineralization(self, orgN, k_min=0.02):
        # Organic N to NH4+ (mineralization)
        rate = k_min * orgN * self.temperature_scalar(self.soil_temp) * self.moisture_scalar(self.soil_moist)
        return max(rate, 0.0)

    def immobilization(self, nh4, k_imm=0.01):
        # NH4+ to microbial N (immobilization)
        rate = k_imm * nh4 * self.temperature_scalar(self.soil_temp) * self.moisture_scalar(self.soil_moist)
        return max(rate, 0.0)

    def nitrification(self, nh4, k_nit=0.03):
        # NH4+ to NO3- (nitrification)
        rate = k_nit * nh4 * self.temperature_scalar(self.soil_temp) * self.moisture_scalar(self.soil_moist)
        return max(rate, 0.0)

    def denitrification(self, no3, k_deni=0.01):
        # NO3- to N2/N2O (denitrification, increases with wetness)
        moist_factor = min(1.0, (self.soil_moist - 0.6) / 0.4) if self.soil_moist > 0.6 else 0.0
        rate = k_deni * no3 * moist_factor * self.temperature_scalar(self.soil_temp)
        return max(rate, 0.0)

    def leaching(self, no3, drainage, k_leach=0.01):
        # NO3- leaching loss (proportional to drainage)
        return min(no3, k_leach * no3 * drainage)

    def plant_uptake(self, nh4, no3, demand_n):
        # Plants take up N from NH4+ and NO3- pools, preference for NH4+
        nh4_uptake = min(nh4, demand_n * 0.5)
        no3_uptake = min(no3, demand_n - nh4_uptake)
        return nh4_uptake, no3_uptake

    def deposition(self, n_dep):
        return n_dep

    def step(self, orgN, demand_n, drainage, n_dep=0.0):
        # 1. Mineralization
        min_n = self.mineralization(orgN)
        self.nh4 += min_n * self.dt
        # 2. Immobilization
        imm_n = self.immobilization(self.nh4)
        self.nh4 -= imm_n * self.dt
        self.mbn += imm_n * self.dt
        # 3. Nitrification
        nit_n = self.nitrification(self.nh4)
        self.nh4 -= nit_n * self.dt
        self.no3 += nit_n * self.dt
        # 4. Denitrification
        deni_n = self.denitrification(self.no3)
        self.no3 -= deni_n * self.dt
        # 5. Leaching
        leach_n = self.leaching(self.no3, drainage)
        self.no3 -= leach_n
        # 6. Plant uptake
        nh4_uptake, no3_uptake = self.plant_uptake(self.nh4, self.no3, demand_n)
        self.nh4 -= nh4_uptake
        self.no3 -= no3_uptake
        # 7. Deposition
        self.nh4 += self.deposition(n_dep)
        # 8. Prevent negative pools
        self.nh4, self.no3, self.mbn = arr_clip(self.nh4), arr_clip(self.no3), arr_clip(self.mbn)
        return {
            'nh4': self.nh4,
            'no3': self.no3,
            'mbn': self.mbn,
            'mineralization': min_n,
            'immobilization': imm_n,
            'nitrification': nit_n,
            'denitrification': deni_n,
            'leaching': leach_n,
            'nh4_uptake': nh4_uptake,
            'no3_uptake': no3_uptake,
            'deposition': n_dep
        }

# Example usage:
ncycle = NCycleCABLE()
days = 200000
results = {k: [] for k in [
    'nh4', 'no3', 'mbn', 'mineralization', 'immobilization', 'nitrification',
    'denitrification', 'leaching', 'nh4_uptake', 'no3_uptake', 'deposition']}
for day in range(days):
    out = ncycle.step(orgN=12.0, demand_n=0.1, drainage=0.1, n_dep=0.0005)
    for k in results:
        results[k].append(out[k])
    if day % 30 == 0 or day == 0:
        print(f"Day {day}: {out}")

t = np.arange(1, days + 1)
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

# NH4, NO3, Microbial N pools
axs[0].plot(t, results['nh4'], label='NH4+')
axs[0].plot(t, results['no3'], label='NO3-')
axs[0].plot(t, results['mbn'], label='Microbial N')
axs[0].set_title('N Pools')
axs[0].legend()

# Mineralization, Immobilization
axs[1].plot(t, results['mineralization'], label='Mineralization')
axs[1].plot(t, results['immobilization'], label='Immobilization')
axs[1].set_title('Mineralization & Immobilization')
axs[1].legend()

# Nitrification, Denitrification
axs[2].plot(t, results['nitrification'], label='Nitrification')
axs[2].plot(t, results['denitrification'], label='Denitrification')
axs[2].set_title('Nitrification & Denitrification')
axs[2].legend()

# Leaching
axs[3].plot(t, results['leaching'], label='Leaching')
axs[3].set_title('Leaching Loss')
axs[3].legend()

# Plant Uptake
axs[4].plot(t, results['nh4_uptake'], label='NH4+ Uptake')
axs[4].plot(t, results['no3_uptake'], label='NO3- Uptake')
axs[4].set_title('Plant Uptake')
axs[4].legend()

# Deposition
axs[5].plot(t, results['deposition'], label='Deposition')
axs[5].set_title('N Deposition')
axs[5].legend()

for ax in axs:
    ax.set_xlabel('Day of Year')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
