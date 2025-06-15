"""
Test for atm_canopy_coupling function from caete_jit.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from caete_jit import atm_canopy_coupling

# Example input values
e_max = 5.0      # maximum evaporation rate (mm/day)
evap = 2.0       # evaporation rate (mm/day)
air_temp = 25.0  # air temperature (Celsius)
vpd = np.linspace(0.01, 6)        # vapor pressure deficit (kPa)

# Calculate the atm_canopy_coupling for each VPD value
results = [atm_canopy_coupling(e_max, evap, air_temp, v) for v in vpd]
# PLot the results
import matplotlib.pyplot as plt
plt.plot(vpd, results, marker='o')
plt.title('Atmosphere-Canopy Coupling')
plt.xlabel('Vapor Pressure Deficit (kPa)')
plt.ylabel('Evapotranspiration Rate (mm/day)')
plt.grid()
plt.show()