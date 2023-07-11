import numpy as np
import polars as pl

from netCDF4 import Dataset
from io import StringIO
from glob import glob


def lat_to_idx(lat):
    """
    The ERA5 dataset we are using have 1 degree lat, lon grid,
    so matching lat to the array index can be done by a simple shifting
    """

    idx = 90 - lat
    return idx


def read_nc(path, var):

    for fname in sorted(glob(f"{path}/*.nc")):
        data = Dataset(fname, "r")
        field = data[var][:]
        data.close()

        yield field


def read_nc_concat(path, var):
    
    data = read_nc(path, var)
    field = np.concatenate(list(data), axis=0)

    return field


def read_t2m_mean(path, lat0, lat1, lon0, lon1):

    dataset = read_nc(path, "t2m")
    tseries = build_tseries_1d(dataset, lat0, lat1, lon0, lon1)

    return tseries


def build_tseries_1d(dataset, lat0, lat1, lon0, lon1):
    
    lat0 = lat_to_idx(lat0)
    lat1 = lat_to_idx(lat1)

    tseries = np.zeros(365 * (2022-1979))

    for i, data in enumerate(dataset):
        tseries[i*365:(i+1)*365] = np.mean(
            data[:365, lat1:lat0, lon0:lon1], axis=(1, 2)
        )

    return tseries


def read_noaa(path, view="table"):
    
    with open(path, "r") as f:
        rows = [",".join(r.split()) for r in f.readlines()]

    csv_str = "\n".join(rows)

    # Add header row
    hedaer = "year,01,02,03,04,05,06,07,08,09,10,11,12\n"
    csv_str = hedaer + csv_str

    df = pl.read_csv(StringIO(csv_str), null_values="-99.99")


    # Flatten the dataframe to a simple time series
    if view == "series":
        q = (
            df.lazy()
            .melt(id_vars="year")
            .select(
                pl.date(pl.col("year"), pl.col("variable"), 1).alias("date"),
                pl.col("value").alias("nino34")        
            )
            .sort(by="date")
            .drop_nulls()
        )

        return q.collect()

    return df
