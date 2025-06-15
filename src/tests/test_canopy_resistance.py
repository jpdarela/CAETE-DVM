from caete_import import model
import numpy as np
import matplotlib.pyplot as plt


vpd = 0.2079972
f1_in = 8.468732432758826e-6
g1 = 0.689989626407623
# ca = 2.832006835937500E-004
ca = 390

test_func = model.photo.stomatal_resistance

catm = np.linspace(190, 900, 100)  # Example range for atmospheric CO2 concentration

def co2pp(ca, p0 = 1013.25):
     return (ca * (p0 * 1.0e2) / 1.0e6) /  (p0 * 1.0e2)

def co2_atm():
    """Test ca range form 200 to 600 ppm"""
    co2_atm = [co2pp(ca) for ca in catm]

    out = np.zeros(100,)
    for i, ca in enumerate(co2_atm):
        out[i] = test_func(vpd, f1_in, g1, ca)

    plt.plot(catm, out)
    plt.xlabel('Atmospheric CO2 Concentration (ca)')
    plt.ylabel('Stomatal Resistance (rs)')
    plt.title('Stomatal Resistance vs Atmospheric CO2 Concentration')
    plt.grid()
    plt.show()

def co2_vpd():
    """Test vpd range from 0.1 to 2.0 kPa"""
    vpd_range = np.linspace(0.01, 5.0, 100)  # Example range for VPD

    ca = co2pp(390)  # Use a fixed CO2 concentration for this test
    f1_in = 8.468732432758826e-6
    g1 = 0.689989626407623

    out = np.zeros(100,)
    for i, vpd in enumerate(vpd_range):
        out[i] = test_func(vpd, f1_in, g1, ca)

    plt.plot(vpd_range, out)
    plt.xlabel('Vapor Pressure Deficit (kPa)')
    plt.ylabel('Stomatal Resistance (rs)')
    plt.title('Stomatal Resistance vs Vapor Pressure Deficit')
    plt.grid()
    plt.show()

def co2_f1():
    """Test f1_in range from 0.0 to 0.01"""
    f1_range = np.linspace( 8.0e-7,10e-5, 1000)  # Example range for f1_in

    ca = co2pp(390)  # Use a fixed CO2 concentration for this test
    vpd = 3
    g1 = 0.689989626407623

    out = np.zeros(1000,)
    for i, f1 in enumerate(f1_range):
        out[i] = test_func(vpd, f1, g1, ca)

    plt.plot(f1_range, out)
    plt.xlabel('f1_in')
    plt.ylabel('Stomatal Resistance (rs)')
    plt.title('Stomatal Resistance vs f1_in')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    co2_atm()
    co2_vpd()
    co2_f1()