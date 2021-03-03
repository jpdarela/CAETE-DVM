import time
from post_processing import write_h5
from h52nc import h52nc


print(time.ctime())
# print("Salvando db")
# write_h5()
print("Processamento Final")
h52nc("../h5/CAETE.h5")
print(time.ctime())
