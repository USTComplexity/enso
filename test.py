import numpy as np

from netCDF4 import Dataset
from enso.network import build_network


fname = "./data/t2m/2m_temp_corrected.nc"
dataset = Dataset(fname, "r")

t2m = dataset["t2m"][:,:20,:20]

# Remove seasonal trend
folded = np.array(np.vsplit(t2m, 43))
mean = np.mean(folded, axis=0)
mean = np.tile(mean, (43, 1, 1))

t2m -= mean


edges = build_network(t2m, 20, 0.5)
