import sys
import numpy as np
import os

sys.path.append("../")
from parameters import fortran_runtime
current_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    try:
        os.add_dll_directory(fortran_runtime)
    except:
        raise ImportError("Could not add the DLL directory to the PATH")

from caete_module import photo as model

def test_water_stress_modifier():

    # Example: test for a range of soil moistures and other parameters
    # You may need to adjust the argument names and number
    # Common arguments: soil moisture, wilting point, field capacity, etc.

    # Example input values (adjust as needed)
    soil_moisture = np.linspace(0, 1, 11)  # 0.0 to 1.0 in steps of 0.1
    wilting_point = 0.1
    field_capacity = 0.4

    print("Soil Moisture | Water Stress Modifier")
    for sm in soil_moisture:
        # Try common function names; adjust as needed
        try:
            # If your function is named differently, change here
            wsm = model.water_stress_modifier(sm, wilting_point, field_capacity)
        except AttributeError:
            try:
                wsm = model.wstress_mod(sm, wilting_point, field_capacity)
            except AttributeError:
                raise RuntimeError("Could not find water stress modifier function in caete_module.")
        print(f"{sm:.2f}          | {wsm:.4f}")

if __name__ == "__main__":
    test_water_stress_modifier()