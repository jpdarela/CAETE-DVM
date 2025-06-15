from caete_import import *


f = model.photo.leaf_nitrogen_concetration

n = 1000
table = table_gen(n)
tleaf = table[2, :]  # tau_leaf in years
leaf_n = table[9, :]

nmean = np.mean(leaf_n)

def test_n_conc(tleaf, i):
    """
    Test the leaf_nitrogen_concetration function from the model.photo module.
    """
    nconc = f(tleaf[i])
    print(f"tau_leaf: {tleaf[i]}, nconc: {nconc:.3f} g/g")

out = np.zeros(n)
for i in range(n):
    out[i] = f(tleaf[i])
    test_n_conc(tleaf, i)

mmean = np.mean(out)