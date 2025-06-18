import pandas as pd
import re

FRED = r"C:\Users\darel\OneDrive\Documentos\FRED\FRED3_Entire_Database_2021.csv"

fred_database = pd.read_csv(FRED, low_memory=False)

indexer = fred_database.loc(0)
traits = indexer[0][10:]

fc = 0.47 # conversion factor for converting from g(dry weight) to g(C)

# use this function to search for traits in the FRED database
def search_traits(trait_str, traits=traits):
    return traits[traits.str.contains(trait_str, case=True)]

# Once you know the trait ID (e.g. "F00261"), you can use this function to get the trait data
def get_trait(trait_id:str, dataframe=fred_database):
    return dataframe[trait_id][13:].reset_index(drop=True).astype(float) / fc / 1000  # convert from g(dry weight) to g(C) and then to g(growth weight)

def get_trait_name(trait_id:str, dataframe=fred_database):
    return dataframe[trait_id][0]

def get_trait_description(trait_id:str, dataframe=fred_database):
    return dataframe[trait_id][1:3]

def get_trait_units(trait_id:str, dataframe=fred_database):
    return dataframe[trait_id][3]

def get_trait_data(trait_id:str, dataframe=fred_database):
    return {
        "id": trait_id,
        "name": get_trait_name(trait_id, dataframe),
        "description": get_trait_description(trait_id, dataframe),
        "units": get_trait_units(trait_id, dataframe),
        "data": get_trait(trait_id, dataframe)
    }

root_n_content_dw = "F00261"
root_p_content_dw = "F00277"

max_n_content_dw = "F00266"
max_p_content_dw = "F00278"

min_n_content_dw = "F00265"
min_p_content_dw = "F00282"

root_c_content_dw = "F00253"
root_cn_ratio = "F00413"


mc = get_trait_data(root_c_content_dw)["data"]
mcn = get_trait_data(root_cn_ratio)["data"]
mn = get_trait_data(root_n_content_dw)["data"]
mp = get_trait_data(root_p_content_dw)["data"]

minn = get_trait_data(min_n_content_dw)["data"]
minp = get_trait_data(min_p_content_dw)["data"]
maxn = get_trait_data(max_n_content_dw)["data"]
maxp = get_trait_data(max_p_content_dw)["data"]

# header = ["rootN","rootP","maxRootN","maxRootP","minRootN","minRootP"]

data = pd.DataFrame({
    "rootC": mc,
    "rootCN": mcn,
    "rootN": mn,
    "rootP": mp,
    "maxRootN": maxn,
    "maxRootP": maxp,
    "minRootN": minn,
    "minRootP": minp
})

# Save the data to a CSV file
data.to_csv("fred_database.csv", index=False)

# Calculate Maximum and minimum Root N:C ratios (g/g)
rootN = data["rootN"] / data["rootC"]

# Calculate Maximum and minimum Root P:C ratios (g/g)
rootP = data["rootP"] / data["rootC"]

rootNP = data["rootN"] / data["rootP"]

print(f"Root N:C ratios (g/g): {rootN.min()} - {rootN.max()}")
print(f"Root P:C ratios (g/g): {rootP.min()} - {rootP.max()}")
print(f"Root N:P ratios (g/g): {rootNP.min()} - {rootNP.max()}")
