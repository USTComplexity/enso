from netCDF4 import Dataset
from enso.network import build_network


fname = "./data/t2m/2m_temp_merged.nc"
dataset = Dataset(fname, "r")

t2m = dataset["t2m"][:]

edges = build_network(t2m, 20, 0.1)

