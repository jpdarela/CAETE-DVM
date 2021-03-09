# experiments.py
from pathlib import Path
import os
import joblib
import tables as tb
from post_processing import write_h5
import timeit as tit
# from h52nc import h52nc

# h52nc(Path("../outputs/r5/CAETE.h5").resolve(), Path("../h5"))

h5f = tb.open_file("../outputs/r5/CAETE.h5", mode='a')

t1d = h5f.root.RUN0.indexedT1date


def a():
    coords = t1d.get_where_list("(date == b'19790101')")
    data = t1d.read_coordinates(coords)
    return data


def b():
    data = t1d.read_where("(date == b'19790101')")
    return data


print("Timing strateg a")
print(tit.timeit(stmt=a, number=600))

print("Timing strateg b")
print(tit.timeit(stmt=b, number=600))
