# test_cc.py
import os
import time

try:
    from cc import carbon_costs as cc
except:
    os.system("./bd.sh")
    time.sleep(4)
finally:
    from cc import carbon_costs as cc

print(cc.__doc__)

os.system("rm -rf cc.cpython*")
