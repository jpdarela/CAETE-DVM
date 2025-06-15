# import matplotlib.pyplot as plt
# from numba import vectorize
# import numpy as np

# @vectorize(["float64(float64, float64)"], target="parallel")
# def r1(M, Y):
#     return Y * M ** (-0.25)

# def r2(M, Y):
#     return Y * M ** (0.75)

# a = np.arange(1, 100, dtype=np.float64)
# b = r1(a, 1.0)
# c = r2(a, 1.0)

# plt.plot(a, c, label="r1")
# # plt.plot(a, c, label="r2")
# plt.xlabel("M")
# plt.ylabel("r")
# plt.legend()
# plt.show()

from worker import worker

if __name__ == "__main__":
    region = worker.load_state_zstd(r"C:\Users\darel\OneDrive\Desktop\CAETE-DVM\src\cities_MPI-ESM1-2-HR_hist_output.psz")

