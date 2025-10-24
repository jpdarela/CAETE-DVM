# import matplotlib.pyplot as plt
# import pandas as pd


# # Edite o caminho para o arquivo .csv con as PLSs, na lista de colunas, coloque as colunas que deseja plotar
# # Pra saber quais as colunas disponíveis, abra o arquivo .csv no Excel
# # table_woody = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[1600:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]
# # table_grass = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[:16, ["aleaf","aroot","tleaf","troot"]]

# table_cturn = pd.read_csv("../src/PLS_MAIN/pls_attrs-5000.csv").loc[:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]

# pd.plotting.scatter_matrix(pd.DataFrame(table_cturn), figsize=(10, 10))
# plt.tight_layout()
# plt.savefig("scatter_matrix_Cturn.png", dpi=300)

# table_nconc = pd.read_csv("../src/PLS_MAIN/pls_attrs-9999.csv").loc[:, ["leaf_n2c","awood_n2c","froot_n2c","leaf_p2c","awood_p2c","froot_p2c"]]

# pd.plotting.scatter_matrix(pd.DataFrame(table_nconc), figsize=(10, 10))
# plt.tight_layout()
# plt.savefig("scatter_matrix_nconc.png", dpi=300)


# table_nconc = pd.read_csv("../src/PLS_MAIN/pls_attrs-9999.csv").loc[:, ["leaf_n2c","leaf_p2c", "aleaf", "tleaf"]]

# pd.plotting.scatter_matrix(pd.DataFrame(table_nconc), figsize=(10, 10))
# plt.tight_layout()
# plt.savefig("scatter_matrix_leafCNP.png", dpi=300)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

def create_colored_scatter_matrix(data, columns, plant_type_col, title, filename):
    """Create a scatter matrix with colored plant types"""
    
    # Color mapping
    color_map = {'Grass': '#228B22', 'Woody': '#8B4513'}
    colors = [color_map[pt] for pt in data[plant_type_col]]
    
    # Create scatter matrix
    fig = plt.figure(figsize=(12, 10))
    pd.plotting.scatter_matrix(data[columns], 
                              c=colors, 
                              figsize=(12, 10), 
                              alpha=0.6,
                              s=20)
    
    # Add legend
    legend_elements = [Patch(facecolor=color_map[pt], label=pt) 
                      for pt in color_map.keys() if pt in data[plant_type_col].values]
    plt.figlegend(handles=legend_elements, 
                  loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Read and process data
table_cturn = pd.read_csv("../src/PLS_MAIN/pls_attrs-5000.csv")
table_nconc = table_cturn.copy()

# Add plant type classification (adjust ranges as needed)
def classify_plant_type(df):
    plant_types = []
    for i in range(len(df)):
        if i <= 351:
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