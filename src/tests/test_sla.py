import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from caete_import import *
import config

conf = config.fetch_config(config="../plsgen.toml")
tau_leaf_max = conf.residence_time.woody.leaf_max
tau_leaf_min = conf.residence_time.woody.leaf_min


n = 100  # Number of points for the residence time
m = n  # Number of points for the leaf biomass
residence_time = np.linspace(tau_leaf_min, tau_leaf_max, n)
colormap = cm.get_cmap("viridis", m)

def test_lai():
    """Test the SLA values"""
    leaf_biomass = np.linspace(0.001, 3, m)  # Leaf biomass in kg/m^2
    combo = np.zeros((n, m))
    fig, ax = plt.subplots()
    for i, lb in enumerate(leaf_biomass):
        sla = np.zeros(n)
        lai = np.zeros(n)
        for j in range(n):
            sla[j] = model.spec_leaf_area(residence_time[j])
            lai[j] = model.leaf_area_index(lb, sla[j])
        ax.plot(residence_time, lai, alpha=0.5, color=colormap(i))
        combo[:, i] = lai
    ax.set_xlabel("Residence Time (years)")
    ax.set_ylabel("Leaf Area Index (m²/m²)")
    ax.set_title("Leaf Area Index vs. Residence Time")
    norm = plt.Normalize(vmin=leaf_biomass.min(), vmax=leaf_biomass.max())
    plt.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax, label="Leaf Biomass (kg/m²)")
    # ax.plot(residence_time, mean_lai, alpha=0.8, color="black")
    plt.show()
    plt.close(fig)

    mean_lai = np.mean(combo, axis=0)
    fig, ax = plt.subplots()
    ax.plot(leaf_biomass, mean_lai, alpha=0.8, color="black")
    ax.set_xlabel("Leaf Biomass (kg/m²)")
    ax.set_ylabel("Leaf Area Index (m²/m²)")
    ax.set_title("Leaf Area Index vs. Leaf Biomass")
    plt.show()
    plt.close(fig)
    return mean_lai


if __name__ == "__main__":
    a = test_lai()
