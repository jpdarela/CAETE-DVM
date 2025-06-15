from caete_import import model
import numpy as np

# print(model.photo.canopy_resistence.__doc__)

def test_stomatal_conductance():
    print ("Testing stomatal conductance...")
    print ("Function signature: ")
    print(model.photo.stomatal_conductance.__doc__)
    print("ENd of function signature---")

    vpd = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example vapor pressure deficit values
    A = np.array([5.0, 10.0, 15.0, 20.0, 25.0])  # Example photosynthesis rates
    g1 = 7  # Example g1 value
    ca = 400  # Example ambient CO2 concentration

    # Call the stomatal conductance function
    for i in range(len(vpd)):
        gsw = model.photo.stomatal_conductance(vpd[i], A[i], g1, ca)
        print(f"VPD: {vpd[i]}, A: {A[i]}, g1: {g1}, ca: {ca} => gsw: {gsw}")

if __name__ == "__main__":
    test_stomatal_conductance()
