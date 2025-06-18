import matplotlib.pyplot as plt
import pandas as pd


# Edite o caminho para o arquivo .csv con as PLSs, na lista de colunas, coloque as colunas que deseja plotar
# Pra saber quais as colunas disponíveis, abra o arquivo .csv no Excel
# table_woody = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[1600:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]
# table_grass = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[:16, ["aleaf","aroot","tleaf","troot"]]

table_cturn = pd.read_csv("../src/PLS_MAIN/pls_attrs-9999.csv").loc[:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]

pd.plotting.scatter_matrix(pd.DataFrame(table_cturn), figsize=(10, 10))
plt.tight_layout()
plt.savefig("scatter_matrix_Cturn.png", dpi=300)

table_nconc = pd.read_csv("../src/PLS_MAIN/pls_attrs-9999.csv").loc[:, ["leaf_n2c","awood_n2c","froot_n2c","leaf_p2c","awood_p2c","froot_p2c"]]

pd.plotting.scatter_matrix(pd.DataFrame(table_nconc), figsize=(10, 10))
plt.tight_layout()
plt.savefig("scatter_matrix_nconc.png", dpi=300)


table_nconc = pd.read_csv("../src/PLS_MAIN/pls_attrs-9999.csv").loc[:, ["leaf_n2c","leaf_p2c", "aleaf", "tleaf"]]

pd.plotting.scatter_matrix(pd.DataFrame(table_nconc), figsize=(10, 10))
plt.tight_layout()
plt.savefig("scatter_matrix_leafCNP.png", dpi=300)


