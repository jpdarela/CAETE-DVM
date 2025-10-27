# import matplotlib.pyplot as plt
# import pandas as pd


# # Edite o caminho para o arquivo .csv con as PLSs, na lista de colunas, coloque as colunas que deseja plotar
# # Pra saber quais as colunas disponíveis, abra o arquivo .csv no Excel
# # table_woody = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[1600:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]
# # table_grass = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[:16, ["aleaf","aroot","tleaf","troot"]]

import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import numpy as np
from matplotlib.patches import Patch

GRASS_FRAC = 0.07  # Fraction of grass PLSs in the dataset

def assert_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = GRASS_FRAC
    diffg = ceil(dsize * g2w_ratio)
    diffw = int(dsize - diffg)
    assert diffg + diffw == dsize
    return diffg, diffw

def create_colored_scatter_matrix(data, columns, plant_type_col, title, filename):
    """Create a scatter matrix with colored plant types"""
    
    # Color mapping
    color_map = {'Grass': "#C709B7", 'Woody': "#05F826"}
    colors = [color_map[pt] for pt in data[plant_type_col]]
    
    # Create scatter matrix
    pd.plotting.scatter_matrix(data[columns], 
                              c=colors, 
                              figsize=(8, 6), 
                              alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor=color_map[pt], label=pt) 
                      for pt in color_map.keys() if pt in data[plant_type_col].values]
    plt.figlegend(handles=legend_elements, 
                  loc='upper right', 
                  bbox_to_anchor=(0.02, 0.98))
    plt.tight_layout()
    plt.savefig(filename, dpi=400)


# def create_colored_scatter_matrix(data, columns, plant_type_col, title, filename):
#     """Create a scatter matrix with colored plant types and density isolines"""
#     from scipy.stats import gaussian_kde
#     import numpy as np
    
#     # Set up the figure and axes grid
#     n_vars = len(columns)
#     fig, axes = plt.subplots(nrows=n_vars, ncols=n_vars, figsize=(10, 10))
#     plt.subplots_adjust(hspace=0.2, wspace=0.2)
    
#     # Color mapping for scatter points
#     color_map = {'Grass': "#C709B7", 'Woody': "#05F826"}
#     color_map_isolines = {'Grass': "#44033F", 'Woody': "#03460C"}
    
#     # Create density plots
#     for i in range(n_vars):
#         for j in range(n_vars):
#             ax = axes[i, j]
            
#             if i != j:  # Off-diagonal: scatter plots with density contours
#                 # Separate data by plant type
#                 for plant_type in ['Grass', 'Woody']:
#                     mask = data[plant_type_col] == plant_type
#                     x = data[columns[j]][mask]
#                     y = data[columns[i]][mask]
                    
#                     # Scatter plot
#                     ax.scatter(x, y, c=color_map[plant_type], alpha=0.3, s=10)
                    
#                     # Calculate density
#                     if len(x) > 1:  # Need at least 2 points for KDE
#                         xy = np.vstack([x, y])
#                         try:
#                             kde = gaussian_kde(xy)
                            
#                             # Create a regular grid to evaluate the density
#                             xmin, xmax = x.min(), x.max()
#                             ymin, ymax = y.min(), y.max()
#                             xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#                             zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
#                             zi = zi.reshape(xi.shape)
                            
#                             # Plot density contours
#                             levels = np.percentile(zi, [25, 50, 75, 90])
#                             ax.contour(xi, yi, zi, levels=levels, colors=color_map_isolines[plant_type],
#                                      alpha=0.8, linewidths=1)
#                         except (ValueError, np.linalg.LinAlgError):
#                             # Skip density plot if KDE fails
#                             pass
                
#                 # Set labels only on the bottom and left edges
#                 if i == n_vars-1:
#                     ax.set_xlabel(columns[j])
#                 if j == 0:
#                     ax.set_ylabel(columns[i])
            
#             else:  # Diagonal: histogram
#                 for plant_type in ['Grass', 'Woody']:
#                     mask = data[plant_type_col] == plant_type
#                     ax.hist(data[columns[i]][mask], bins=30, 
#                            color=color_map[plant_type], alpha=0.5,
#                            density=True)
    
#     # Add legend
#     legend_elements = [Patch(facecolor=color_map[pt], label=pt) 
#                       for pt in color_map.keys() if pt in data[plant_type_col].values]
#     plt.figlegend(handles=legend_elements, 
#                   loc='center right',
#                   bbox_to_anchor=(0.98, 1.2))
    
#     # plt.suptitle(title, y=0.95)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=400, bbox_inches='tight')
#     plt.close()

# Read and process data
table_cturn = pd.read_csv("../src/PLS_MAIN/pls_attrs-500.csv")
table_nconc = table_cturn.copy()

# Add plant type classification (adjust ranges as needed)
def classify_plant_type(df):
    plant_types = []
    gf, _ = assert_data_size(len(df))
    for i in range(len(df)):
        if i < gf:
            plant_types.append('Grass')
        else:
            plant_types.append('Woody')
    return plant_types

table_cturn['plant_type'] = classify_plant_type(table_cturn)
table_nconc['plant_type'] = classify_plant_type(table_nconc)

# Create all three plots
create_colored_scatter_matrix(
    table_cturn, 
    ["aleaf","awood","aroot","tleaf", 'twood', "troot"],
    'plant_type',
    'Carbon Allocation and Turnover Traits',
    'scatter_matrix_Cturn_colored.png'
)

create_colored_scatter_matrix(
    table_nconc,
    ["leaf_n2c","awood_n2c","froot_n2c","leaf_p2c","awood_p2c","froot_p2c"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_nconc_colored.png'
)

create_colored_scatter_matrix(
    table_nconc,
    ["leaf_n2c","leaf_p2c", "aleaf", "tleaf"],
    'plant_type', 
    'Leaf CNP and Allocation Traits',
    'scatter_matrix_leafCNP_colored.png'
)

create_colored_scatter_matrix(
    table_nconc,
    ["awood_n2c","awood_p2c", "awood", "twood"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_wood_CNP.png'
)

create_colored_scatter_matrix(
    table_nconc,
    ["froot_n2c","froot_p2c", "aroot", "troot"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_root_CNP.png'
)
