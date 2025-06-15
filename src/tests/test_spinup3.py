from caete_import import *

spinup = model.photo.spinup3
n = 1000

npp = 0.01

table = table_gen(n)

c_turn = table[2:8, :]  # Carbon turnover in days

header = "leaf,root,wood"
with open("spinup_test.csv", "w") as f:
    f.write(header + "\n")
with open("spinup_test.csv", "a") as f:
    for x in range(n):
        leaf, root, wood = spinup(npp, c_turn[:, x])
        line = f"{leaf:.3f},{root:.3f},{wood:.3f}"
        f.write(line + "\n")
        # print(f"Leaf: {leaf:.8f}, Root: {root:.8f}, Wood: {wood:.8f}")
