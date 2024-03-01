import numpy as np
import pickle

from netCDF4 import Dataset
from enso.network import build_network, build_network_pnas


# fname = "./2m_temp_corrected.nc"
fname = "./data/t2m/2m_temp_corrected.nc"
dataset = Dataset(fname, "r")

t2m = dataset["t2m"][:, 60:120:5, 180:260:5]
print(t2m.shape)

# Remove seasonal mean
folded = np.array(np.vsplit(t2m, 43))
mean = np.mean(folded, axis=0)
mean = np.tile(mean, (43, 1, 1))

t2m -= mean


# Build correlation network
# s = build_network(t2m, 20, 0.5)

# Build network as defined by Ludescher et al., PNAS 110, 29, 11743 (2013)
s = build_network_pnas(t2m, 5)


with open("network.pkl", "wb") as f:
    pickle.dump(s, f)
