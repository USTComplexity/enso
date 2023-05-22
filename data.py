import numpy as np
from netCDF4 import Dataset


def lat_to_idx(lat):
    """
    The ERA5 dataset we are using have 1 degree lat, lon grid,
    so matching lat to the array index can be done by a simple shifting
    """

    idx = 90 - lat
    return idx


def read_t2m(path):
    
    for yr in range(1979, 2022):
        data = Dataset(f"{path}/2m_temp_{yr}.nc", "r")
        t2m = data["t2m"][:]
        data.close()

        yield t2m


def build_tseries_1d(dataset, lat0, lat1, lon0, lon1):
    
    lat0 = lat_to_idx(lat0)
    lat1 = lat_to_idx(lat1)

    tseries = np.zeros(365 * (2022-1979))

    for i, data in enumerate(dataset):
        tseries[i*365:(i+1)*365] = np.mean(
            data[:365, lat1:lat0, lon0:lon1], axis=(1, 2)
        )

    return tseries
