# ! Copyright 2017- LabTerra

# !     This program is free software: you can redistribute it and/or modify
# !     it under the terms of the GNU General Public License as published by
# !     the Free Software Foundation, either version 3 of the License, or
# !     (at your option) any later version.)

# !     This program is distributed in the hope that it will be useful,
# !     but WITHOUT ANY WARRANTY; without even the implied warranty of
# !     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# !     GNU General Public License for more details.

# !     You should have received a copy of the GNU General Public License
# !     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ! AUTHORS: Gabriel Marandola & JP Darela

import warnings
from math import log as ln
# from numba import jit
# import caete_module as caete_mod


def rwarn(txt='RuntimeWarning'):
    """Raise a runtime warning with the given text.

    Args:
        txt (str): The warning message to display. Defaults to 'RuntimeWarning'.
    """
    warnings.warn(f"{txt}", RuntimeWarning)


def B_func(Th33, Th1500):
    """calculates the coefficient of moisture-tension"""

    D = ln(Th33) - ln(Th1500)
    B = (ln(1500) - ln(33)) / D

    def lbd_func(C):
        """return the slope of logarithmic tension-moisture curve"""
        if C == 0:
            return 0.0
        lbd = 1 / C
        return lbd

    return lbd_func(B)

# @jit(nopython=True)
def ksat_func(ThS, Th33, lbd):
    """soil conductivity in saturated condition. Output in mm/h"""

    # assert ThS > Th33, "sat <= fc IN ksat_func"
    ksat = 1930 * (ThS - Th33) ** (3 - lbd)
    return ksat

# @jit(nopython=True)
def kth_func(Th, ThS, lbd, ksat):
    """soil conductivity in unsaturated condition. Output in mm/h"""
    if Th < 0.0:
        # rwarn("water content < 0 IN kth_func")
        Th = 0.0
    kth = ksat * (Th / ThS) ** (3 + (2 / lbd))

    return kth


class soil_water:

    def __init__(self, ws1, ws2, fc1, fc2, wp1, wp2):
        """
        ws: float: water saturation (Fraction  of soil volume - all)
        fc: float: field capacity
        wp: float: wilting point
        """
        assert ws1 > fc1 and fc1 > wp1
        assert ws1 > 0.0 and ws1 <= 1.0
        assert fc1 > 0.0 and fc1 <= 1.0
        assert wp1 > 0.0 and wp1 <= 1.0

        assert ws2 > fc2 and fc2 > wp2
        assert ws2 > 0.0 and ws2 <= 1.0
        assert fc2 > 0.0 and fc2 <= 1.0
        assert wp2 > 0.0 and wp2 <= 1.0

        self.ws1 = ws1
        self.fc1 = fc1
        self.wp1 = wp1

        self.ws2 = ws2
        self.fc2 = fc2
        self.wp2 = wp2

        # Soil Water POols mm (Kg m-2). We subtract the water content at wilt point
        self.w1 = 100.0  # Initial water content
        self.w2 = 140.0  # Initial water content
        self.awc1 = self.w1 - (self.wp1 * self.w1)
        self.awc2 = self.w2 - (self.wp2 * self.w2)


        # Set pool sizes
        # Kilograms of water that would fit the  soil layers in the ausence of soil
        self.p1_vol = 300.0  # Kg(H2O) - Layer 1
        self.p2_vol = 700.0  # Kg(H2O) - Layer 2

        # MAX soil water depth kg m-2
        self.w1_max = self.p1_vol * self.ws1
        self.w2_max = self.p2_vol * self.ws2

        self.wmax = self.w1_max + self.w2_max

        self.lbd_1 = B_func(self.fc1, self.wp1)

        self.ksat_1 = ksat_func(self.ws1, self.fc1, self.lbd_1)

        self.lbd_2 = B_func(self.fc2, self.wp2)

        self.ksat_2 = ksat_func(self.ws2, self.fc2, self.lbd_2)

    def calc_awc(self):
        """Calculates the available water capacity for the grid cell"""
        # self.awc1 = max(0.0, self.w1 - (self.wp1 * self.w1))
        # self.awc2 = max(0.0, self.w2 - (self.wp2 * self.w2))
        # Layer 1
        fc1_amt = self.fc1 * self.p1_vol
        wp1_amt = self.wp1 * self.p1_vol
        self.awc1 = max(0.0, min(self.w1, fc1_amt) - wp1_amt)
        # Layer 2
        fc2_amt = self.fc2 * self.p2_vol
        wp2_amt = self.wp2 * self.p2_vol
        self.awc2 = max(0.0, min(self.w2, fc2_amt) - wp2_amt)

    def calc_total_water(self):
        """Returns the total water content in the soil (kg m-2)."""
        return self.w1 + self.w2

    def _update_pool(self, prain, evapo):
        """Calculates upper and lower soil water pools for the grid cell,
        as well as the grid runoff and the water fluxes between layers"""

        # evapo adaptation to use in both layers
        ev1 = evapo * 0.3  # Kg m-2
        ev2 = evapo - ev1  # Kg m-2

        # runoff initial values
        runoff1 = 0.0  # Kg m-2
        runoff2 = 0.0  # Kg m-2
        ret1 = 0.0  # Kg m-2

        # POOL 1 - 0-30 cm
        # rainfall addition to upper water content
        self.w1 += prain  # kg m-2

        if self.w1 > self.w1_max:
            # saturated condition (runoff and flux)
            runoff1 += (self.w1 - self.w1_max)
            self.w1 = self.w1_max
            flux1_mm = self.ksat_1 * 24.0  # Kg m-2 day-1
        else:
            # unsaturated condition (no runoff)
            w1_vol = self.w1 / self.p1_vol
            flux1_mm = kth_func(w1_vol, self.ws1, self.lbd_1,
                                self.ksat_1) * 24.0  # Kg m-2 day-1

        # Update pool 1

        self.w1 -= (ev1 + flux1_mm)

        # POOL 2 30-100 cm
        # Flux comming from w1
        self.w2 += flux1_mm  # Kg m-2 day-1
        if self.w2 < 0.0:
            self.w2 = 0.0

        if self.w2 > self.w2_max:
            # saturated condition
            ret1 = self.w2 - self.w2_max
            # The surplus remains in w1
            self.w1 += ret1
            # Check w1 runoff by return
            if self.w1 > self.w1_max:
                runoff1 += (self.w1 - self.w1_max)
                self.w1 = self.w1_max
            self.w2 = self.w2_max
            runoff2 += self.ksat_2 * 24.0
        else:
            w2_vol = self.w2 / self.p2_vol
            # unsaturated condition (runoff)
            runoff2 += kth_func(w2_vol, self.ws2,
                                self.lbd_2, self.ksat_2) * 24.0

        self.w2 -= (runoff2 + ev2)
        if runoff1 < 1e-17:
            runoff1 = 0.0
        if runoff2 < 1e-17:
            runoff2 = 0.0

        # Update available water capacity
        self.calc_awc()

        return runoff1 + runoff2


def test_water_balance():
    import numpy as np

    ws1, ws2 = 0.45, 0.48
    fc1, fc2 = 0.39, 0.43
    wp1, wp2 = 0.24, 0.27

    wp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)

    nsteps = 100000
    total_rain = 0.0
    total_evap = 0.0
    total_runoff = 0.0

    initial_storage = wp.calc_total_water()

    for x in range(nsteps):
        evapo = abs(np.random.normal()) if np.random.normal() > 0 else 0
        rain = evapo + abs(np.random.normal())
        runoff = wp._update_pool(rain, evapo)
        wp.w1 = 0.0 if wp.w1 < 0.0 else wp.w1
        wp.w2 = 0.0 if wp.w2 < 0.0 else wp.w2

        total_rain += rain
        total_evap += evapo
        total_runoff += runoff

    final_storage = wp.calc_total_water()
    storage_change = final_storage - initial_storage
    balance_error = total_rain - (total_evap + total_runoff + storage_change)

    print(f"Initial storage: {initial_storage:.4f} kg/m2")
    print(f"Final storage:   {final_storage:.4f} kg/m2")
    print(f"Total rain:      {total_rain:.4f} kg/m2")
    print(f"Total evap:      {total_evap:.4f} kg/m2")
    print(f"Total runoff:    {total_runoff:.4f} kg/m2")
    print(f"Storage change:  {storage_change:.4f} kg/m2")
    print(f"Water balance error (should be ~0): {balance_error:.6e} kg/m2")
    if abs(balance_error) > 1e-6:
        print("Warning: Water balance error exceeds tolerance!")
    else:
        print("Water balance is within acceptable limits: < 1e-6 kg/m2")


def main():
    import numpy as np
    import matplotlib.pyplot as plt

    ws1, ws2 = 0.45, 0.48
    fc1, fc2 = 0.39, 0.43
    wp1, wp2 = 0.24, 0.27

    wp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)

    nsteps = 1000
    w1_list = []
    w2_list = []
    awc1_list = []
    awc2_list = []
    runoff_list = []

    for x in range(nsteps):
        evapo = 5 if np.random.normal() > 0 else 0
        roff = wp._update_pool(evapo + 1, evapo)
        wp.w1 = 0.0 if wp.w1 < 0.0 else wp.w1
        wp.w2 = 0.0 if wp.w2 < 0.0 else wp.w2
        w1_list.append(wp.w1)
        w2_list.append(wp.w2)
        awc1_list.append(wp.awc1)
        awc2_list.append(wp.awc2)
        runoff_list.append(roff)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(w1_list, label='w1 (upper layer)')
    axs[0].plot(w2_list, label='w2 (lower layer)')
    axs[0].set_ylabel('Soil Water (kg/m²)')
    axs[0].legend()
    axs[0].set_title('Soil Water Content')

    axs[1].plot(awc1_list, label='awc1 (upper layer)')
    axs[1].plot(awc2_list, label='awc2 (lower layer)')
    axs[1].set_ylabel('AWC (kg/m²)')
    axs[1].legend()
    axs[1].set_title('Available Water Capacity')

    axs[2].plot(runoff_list, label='Runoff')
    axs[2].set_ylabel('Runoff (kg/m²)')
    axs[2].set_xlabel('Timestep')
    axs[2].legend()
    axs[2].set_title('Runoff')

    plt.tight_layout()
    plt.show()


def test_water_balance_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    ws1, ws2 = 0.45, 0.48
    fc1, fc2 = 0.39, 0.43
    wp1, wp2 = 0.24, 0.27

    wp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)

    nsteps = 1000
    total_rain = 0.0
    total_evap = 0.0
    total_runoff = 0.0

    initial_storage = wp.calc_total_water()

    w1_list = []
    w2_list = []
    awc1_list = []
    awc2_list = []
    runoff_list = []
    rain_list = []
    evap_list = []
    storage_list = []
    balance_error_list = []

    for x in range(nsteps):
        evapo = 5 if np.random.normal() > 0 else 0
        rain = evapo + 1
        runoff = wp._update_pool(rain, evapo)
        wp.w1 = 0.0 if wp.w1 < 0.0 else wp.w1
        wp.w2 = 0.0 if wp.w2 < 0.0 else wp.w2

        total_rain += rain
        total_evap += evapo
        total_runoff += runoff

        w1_list.append(wp.w1)
        w2_list.append(wp.w2)
        awc1_list.append(wp.awc1)
        awc2_list.append(wp.awc2)
        runoff_list.append(runoff)
        rain_list.append(rain)
        evap_list.append(evapo)
        storage_list.append(wp.calc_total_water())
        # Water balance error up to this step
        storage_change = wp.calc_total_water() - initial_storage
        balance_error = total_rain - (total_evap + total_runoff + storage_change)
        balance_error_list.append(balance_error)

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    axs[0].plot(w1_list, label='w1 (upper layer)')
    axs[0].plot(w2_list, label='w2 (lower layer)')
    axs[0].set_ylabel('Soil Water (kg/m²)')
    axs[0].legend()
    axs[0].set_title('Soil Water Content')

    axs[1].plot(awc1_list, label='awc1 (upper layer)')
    axs[1].plot(awc2_list, label='awc2 (lower layer)')
    axs[1].set_ylabel('AWC (kg/m²)')
    axs[1].legend()
    axs[1].set_title('Available Water Capacity')

    axs[2].plot(rain_list, label='Rain')
    axs[2].plot(evap_list, label='Evaporation')
    axs[2].plot(runoff_list, label='Runoff')
    axs[2].set_ylabel('Fluxes (kg/m²)')
    axs[2].legend()
    axs[2].set_title('Water Fluxes')

    axs[3].plot(storage_list, label='Total Storage')
    axs[3].plot(balance_error_list, label='Cumulative Water Balance Error')
    axs[3].set_ylabel('kg/m²')
    axs[3].set_xlabel('Timestep')
    axs[3].legend()
    axs[3].set_title('Storage and Water Balance Error')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    test_water_balance()
    test_water_balance_plot()

