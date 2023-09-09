import numpy as np
import pickle

from netCDF4 import Dataset
from enso.network import build_network2


fname = "./data/t2m/2m_temp_corrected.nc"
dataset = Dataset(fname, "r")

# t2m = dataset["t2m"][:, 60:120:5, 180:260:5]
t2m = dataset["t2m"][:, 60:121:5, 180:301:5]
print(t2m.shape)

# Remove seasonal mean
folded = np.array(np.vsplit(t2m, 43))
mean = np.mean(folded, axis=0)
mean = np.tile(mean, (43, 1, 1))

t2m -= mean


s = build_network2(t2m, 200)


with open("sample_larger.pkl", "wb") as f:
    pickle.dump(s, f)
