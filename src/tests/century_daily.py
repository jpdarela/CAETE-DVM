import numpy as np
import matplotlib.pyplot as plt

class CenturyDailyModel:
    def __init__(
        self,
        soil_c_fast=1000.0, soil_c_slow=2000.0,
        soil_n_fast=100.0, soil_n_slow=200.0,
        soil_p_fast=10.0, soil_p_slow=20.0,
        litter_c_fast=50.0, litter_c_slow=50.0,
        litter_n_fast=2.5, litter_n_slow=2.5,
        litter_p_fast=0.25, litter_p_slow=0.25,
        avail_n=5.0, avail_p=1.0,
        temp=15.0, precip=800.0
    ):
        # Pools (g/m2)
        self.soil_c_fast = soil_c_fast
        self.soil_c_slow = soil_c_slow
        self.soil_n_fast = soil_n_fast
        self.soil_n_slow = soil_n_slow
        self.soil_p_fast = soil_p_fast
        self.soil_p_slow = soil_p_slow

        self.litter_c_fast = litter_c_fast
        self.litter_c_slow = litter_c_slow
        self.litter_n_fast = litter_n_fast
        self.litter_n_slow = litter_n_slow
        self.litter_p_fast = litter_p_fast
        self.litter_p_slow = litter_p_slow

        self.avail_n = avail_n
        self.avail_p = avail_p

        self.temp = temp
        self.precip = precip

        # Parameters
        self.k_litter_fast = 0.3 / 365  # fast litter decay (per day)
        self.k_litter_slow = 0.05 / 365 # slow litter decay (per day)
        self.k_soil_fast = 0.05 / 365   # fast soil decay (per day)
        self.k_soil_slow = 0.01 / 365   # slow soil decay (per day)

        self.cn_litter_fast = 20.0
        self.cn_litter_slow = 40.0
        self.cn_soil_fast = 10.0
        self.cn_soil_slow = 20.0

        self.cp_litter_fast = 200.0
        self.cp_litter_slow = 400.0
        self.cp_soil_fast = 100.0
        self.cp_soil_slow = 200.0

        self.leach_frac_n = 0.001  # daily N leaching
        self.leach_frac_p = 0.0002 # daily P leaching
        self.deni_frac = 0.0005    # daily denitrification

    def temperature_scalar(self):
        return 2.0 ** ((self.temp - 15.0) / 10.0)

    def moisture_scalar(self):
        return min(1.0, self.precip / 800.0)

    def step(self, litter_input_fast=(0.137, 0.0068, 0.00068), litter_input_slow=(0.137, 0.0068, 0.00068)):
        """
        Advance the model by one day.
        litter_input_fast/slow: tuple (C, N, P) input to each litter pool (g/m2/day)
        """
        temp_eff = self.temperature_scalar()
        moist_eff = self.moisture_scalar()
        env_eff = temp_eff * moist_eff

        # Litter decomposition
        litf_decomp_c = self.k_litter_fast * self.litter_c_fast * env_eff
        lits_decomp_c = self.k_litter_slow * self.litter_c_slow * env_eff
        litf_decomp_n = litf_decomp_c / self.cn_litter_fast
        lits_decomp_n = lits_decomp_c / self.cn_litter_slow
        litf_decomp_p = litf_decomp_c / self.cp_litter_fast
        lits_decomp_p = lits_decomp_c / self.cp_litter_slow

        # Soil decomposition
        soilf_decomp_c = self.k_soil_fast * self.soil_c_fast * env_eff
        soils_decomp_c = self.k_soil_slow * self.soil_c_slow * env_eff
        soilf_decomp_n = soilf_decomp_c / self.cn_soil_fast
        soils_decomp_n = soils_decomp_c / self.cn_soil_slow
        soilf_decomp_p = soilf_decomp_c / self.cp_soil_fast
        soils_decomp_p = soils_decomp_c / self.cp_soil_slow

        # Mineralization (only soil N/P goes to available pool)
        mineralized_n = soilf_decomp_n + soils_decomp_n
        mineralized_p = soilf_decomp_p + soils_decomp_p

        # Leaching and denitrification
        n_leach = self.leach_frac_n * self.avail_n
        p_leach = self.leach_frac_p * self.avail_p
        n_deni = self.deni_frac * self.avail_n

        # Update pools
        # Litter
        self.litter_c_fast += litter_input_fast[0] - litf_decomp_c
        self.litter_n_fast += litter_input_fast[1] - litf_decomp_n
        self.litter_p_fast += litter_input_fast[2] - litf_decomp_p

        self.litter_c_slow += litter_input_slow[0] - lits_decomp_c
        self.litter_n_slow += litter_input_slow[1] - lits_decomp_n
        self.litter_p_slow += litter_input_slow[2] - lits_decomp_p

        # Soil: add N/P from litter decomposition, subtract N/P from soil decomposition
        self.soil_c_fast += litf_decomp_c - soilf_decomp_c
        self.soil_n_fast += litf_decomp_n - soilf_decomp_n
        self.soil_p_fast += litf_decomp_p - soilf_decomp_p

        self.soil_c_slow += lits_decomp_c - soils_decomp_c
        self.soil_n_slow += lits_decomp_n - soils_decomp_n
        self.soil_p_slow += lits_decomp_p - soils_decomp_p

        # Available pools: only add N/P mineralized from soil, not from litter
        self.avail_n += (soilf_decomp_n + soils_decomp_n) - n_leach - n_deni
        self.avail_p += (soilf_decomp_p + soils_decomp_p) - p_leach

        # Prevent negative pools
        for attr in [
            'litter_c_fast', 'litter_n_fast', 'litter_p_fast',
            'litter_c_slow', 'litter_n_slow', 'litter_p_slow',
            'soil_c_fast', 'soil_n_fast', 'soil_p_fast',
            'soil_c_slow', 'soil_n_slow', 'soil_p_slow',
            'avail_n', 'avail_p'
        ]:
            setattr(self, attr, max(getattr(self, attr), 0.0))

        # Mass balance correction: if soil N/P goes negative, reduce available N/P accordingly
        # This prevents available pools from accumulating N/P that never existed
        if self.soil_n_fast == 0.0 and self.soil_n_slow == 0.0:
            self.avail_n = min(self.avail_n, self.soil_n_fast + self.soil_n_slow)
        if self.soil_p_fast == 0.0 and self.soil_p_slow == 0.0:
            self.avail_p = min(self.avail_p, self.soil_p_fast + self.soil_p_slow)

        return {
            'litter_c_fast': self.litter_c_fast,
            'litter_n_fast': self.litter_n_fast,
            'litter_p_fast': self.litter_p_fast,
            'litter_c_slow': self.litter_c_slow,
            'litter_n_slow': self.litter_n_slow,
            'litter_p_slow': self.litter_p_slow,
            'soil_c_fast': self.soil_c_fast,
            'soil_n_fast': self.soil_n_fast,
            'soil_p_fast': self.soil_p_fast,
            'soil_c_slow': self.soil_c_slow,
            'soil_n_slow': self.soil_n_slow,
            'soil_p_slow': self.soil_p_slow,
            'avail_n': self.avail_n,
            'avail_p': self.avail_p,
            'n_leach': n_leach,
            'p_leach': p_leach,
            'n_deni': n_deni
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = CenturyDailyModel()
    days = 200000
    # Collect daily values
    results = {k: [] for k in [
        'litter_c_fast', 'litter_n_fast', 'litter_p_fast',
        'litter_c_slow', 'litter_n_slow', 'litter_p_slow',
        'soil_c_fast', 'soil_n_fast', 'soil_p_fast',
        'soil_c_slow', 'soil_n_slow', 'soil_p_slow',
        'avail_n', 'avail_p', 'n_leach', 'p_leach', 'n_deni']}
    for day in range(1, days + 1):
        pools = model.step()
        for k in results:
            results[k].append(pools[k])
        if day % 30 == 0 or day == 1:
            print(f"Day {day}: {pools}")

    t = np.arange(1, days + 1)
    fig, axs = plt.subplots(4, 2, figsize=(14, 12))
    axs = axs.flatten()
    # C pools
    axs[0].plot(t, results['litter_c_fast'], label='Litter C Fast')
    axs[0].plot(t, results['litter_c_slow'], label='Litter C Slow')
    axs[0].set_title('Litter C Pools')
    axs[0].legend()
    axs[1].plot(t, results['soil_c_fast'], label='Soil C Fast')
    axs[1].plot(t, results['soil_c_slow'], label='Soil C Slow')
    axs[1].set_title('Soil C Pools')
    axs[1].legend()
    # N pools
    axs[2].plot(t, results['litter_n_fast'], label='Litter N Fast')
    axs[2].plot(t, results['litter_n_slow'], label='Litter N Slow')
    axs[2].set_title('Litter N Pools')
    axs[2].legend()
    axs[3].plot(t, results['soil_n_fast'], label='Soil N Fast')
    axs[3].plot(t, results['soil_n_slow'], label='Soil N Slow')
    axs[3].set_title('Soil N Pools')
    axs[3].legend()
    # P pools
    axs[4].plot(t, results['litter_p_fast'], label='Litter P Fast')
    axs[4].plot(t, results['litter_p_slow'], label='Litter P Slow')
    axs[4].set_title('Litter P Pools')
    axs[4].legend()
    axs[5].plot(t, results['soil_p_fast'], label='Soil P Fast')
    axs[5].plot(t, results['soil_p_slow'], label='Soil P Slow')
    axs[5].set_title('Soil P Pools')
    axs[5].legend()
    # Available N/P
    axs[6].plot(t, results['avail_n'], label='Available N')
    axs[6].plot(t, results['avail_p'], label='Available P')
    axs[6].set_title('Available N & P')
    axs[6].legend()
    # Losses
    axs[7].plot(t, results['n_leach'], label='N Leaching')
    axs[7].plot(t, results['p_leach'], label='P Leaching')
    axs[7].plot(t, results['n_deni'], label='N Denitrification')
    axs[7].set_title('N/P Losses')
    axs[7].legend()
    for ax in axs:
        ax.set_xlabel('Day of Year')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()