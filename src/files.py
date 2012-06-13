import matplotlib.pyplot as plt
import os

def open_imfile(fname):
    "Opens image file"
    afn = os.path.abspath(fname)
    return plt.imread(afn)


