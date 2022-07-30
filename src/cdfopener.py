"""
This file is a testing playground for analyzing PV data structures.
"""
import netCDF4 as nc
import xarray as xr
import pandas as pd

"""
NOTES:
    ncf is the raw nc file dataset with the structure of:

    <class 'netCDF4._netCDF4.Dataset'>
    root group (NETCDF4 data model, file format HDF5):
    dimensions(sizes): datetime(387254)
    variables(dimensions): float32 10003(datetime), float32 10004(dtetime), ...
    
    unpacking a single variable (e.g. 10003):
    
    elem = ncf.variabels['10003']
    elem.shape = (387254,)
    elem.dimensions = ('datetime',)
"""

metadata_df = pd.read_csv('data/metadata.csv')
fn = 'data/pv.netcdf'
#ncf = nc.Dataset(fn)   # Using xarray to open files instead of netcdf for pipeline simplicity
ncf = xr.open_dataset(fn, engine="h5netcdf")
#item = ncf.variables['10003']
print(ncf.keys())



on_pv_system = ncf['10003'].to_dataframe()

on_pv_system = on_pv_system[on_pv_system.index < '2021-06-02']
on_pv_system = on_pv_system[on_pv_system.index > '2021-06-01 10:50:00']
print(on_pv_system)
