import caete_module as m

flai = m.photo.leaf_area_index
f4 = m.photo.f_four

def clai(cl, sl):
    lai = flai(cl,sl)
    fc = f4(90, cl, sl)
    return lai, fc
