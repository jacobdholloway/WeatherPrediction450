import os
os.chdir(os.pardir)
from DLWP.data import ERA5Reanalysis
import xarray as xr


# variables = ['relative_humidity', 'fraction_of_cloud_cover']
# levels = [500]
# years = list(range(2000, 2018))




# data_directory = '/Users/jacobholloway/Developer/'
# os.makedirs(data_directory, exist_ok=True)
# era = ERA5Reanalysis(root_directory=data_directory, file_id='weather_reanalysis')
# era.set_variables(variables)
# era.set_levels(levels)

# era.retrieve(variables, levels, years=years, hourly=3,
#              request_kwargs={'grid': [2., 2.]}, verbose=True, delete_temporary=True)


# Load the data set that is downloaded and print the available variables
ds = xr.open_dataset('tutorial_z500_t2m.nc_nocoord.nc')
print(ds)
print(ds.variables)
