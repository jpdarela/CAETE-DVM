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
import numpy as np
from math import log as ln


def rwarn(txt='RuntimeWarning'):
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


def ksat_func(ThS, Th33, lbd):
    """soil conductivity in saturated condition. Output in mm/h"""

    # assert ThS > Th33, "sat <= fc IN ksat_func"
    ksat = 1930 * (ThS - Th33) ** (3 - lbd)
    return ksat


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

        self.ws1 = np.float64(ws1)
        self.fc1 = np.float64(fc1)
        self.wp1 = np.float64(wp1)

        self.ws2 = np.float64(ws2)
        self.fc2 = np.float64(fc2)
        self.wp2 = np.float64(wp2)

        # Soil Water POols mm (Kg m-2)
        self.w1 = np.float64(20.0)
        self.w2 = np.float64(60.0)

        # Set pool sizes
        # Kilograms of water that would fit the  soil layers in the ausence of soil
        self.p1_vol = np.float64(300.0)  # Kg(H2O) - Layer 1
        self.p2_vol = np.float64(700.0)  # Kg(H2O) - Layer 2

        # MAX soil water depth kg m-2
        self.w1_max = self.p1_vol * self.ws1
        self.w2_max = self.p2_vol * self.ws2

        self.wmax = np.float64(self.w1_max + self.w2_max)

        self.lbd_1 = B_func(self.fc1, self.wp1)

        self.ksat_1 = ksat_func(self.ws1, self.fc1, self.lbd_1)

        self.lbd_2 = B_func(self.fc2, self.wp2)

        self.ksat_2 = ksat_func(self.ws2, self.fc2, self.lbd_2)

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

        return runoff1 + runoff2


if __name__ == "__main__":

    ws1, ws2 = 0.45, 0.48
    fc1, fc2 = 0.39, 0.43
    wp1, wp2 = 0.24, 0.27

    wp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)

    for x in range(5000):
        evapo = 5 if np.random.normal() > 0 else 0
        roff = wp._update_pool(evapo, evapo)
        wp.w1 = np.float(0.0) if wp.w1 < 0.0 else wp.w1
        wp.w2 = np.float(0.0) if wp.w2 < 0.0 else wp.w2
        print(wp.w1, ':w1')
        print(wp.w2, ':w2')
        print(roff, ' :runoff')
