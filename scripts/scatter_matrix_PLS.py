import matplotlib.pyplot as plt
import pandas as pd


# Edite o caminho para o arquivo .csv con as PLSs, na lista de colunas, coloque as colunas que deseja plotar
# Pra saber quais as colunas disponíveis, abra o arquivo .csv no Excel
# table_woody = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[1600:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]
# table_grass = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[:16, ["aleaf","aroot","tleaf","troot"]]

table_all = pd.read_csv("../src/PLS_MAIN/pls_attrs-99999.csv").loc[:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]

pd.plotting.scatter_matrix(pd.DataFrame(table_all), figsize=(10, 10))
plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=300)