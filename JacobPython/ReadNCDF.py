import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap


output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load your dataset
ds = xr.open_dataset('forecast_dlwp-cs_tutorial.nc.cs')

# Select specific slices for visualization
# For example, select the first time point, first forecast hour, first face, and first variable level
selected_data = ds['forecast'].isel(time=0, f_hour=0, face=0, varlev=0)

# Create a meshgrid for height and width dimensions for 3D plotting
height, width = selected_data['height'], selected_data['width']
H, W = np.meshgrid(height, width)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
surf = ax.plot_surface(H, W, selected_data, cmap='viridis')

# Adding labels and title
ax.set_xlabel('Height')
ax.set_ylabel('Width')
ax.set_zlabel('Forecast Value')
ax.set_title('3D Surface Plot of Forecast')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()




def plot_movie(m, lat, lon, val, pred, dates, model_title='', plot_kwargs=None, out_directory=None):
    if (len(dates) != val.shape[0]) and (len(dates) != pred.shape[0]):
        raise ValueError("'val' and 'pred' must have the same first (time) dimension as 'dates'")
    plot_kwargs = plot_kwargs or {}
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    x, y = m(lon, lat)
    dt = dates[1] - dates[0]
    for d, date in enumerate(dates):
        hours = (d + 1) * dt.total_seconds() / 60 / 60
        ax = plt.subplot(211)
        m.pcolormesh(x, y, val[d], **plot_kwargs)
        m.drawcoastlines()
        m.drawparallels(np.arange(0., 91., 45.))
        m.drawmeridians(np.arange(0., 361., 90.))
        ax.set_title('Verification (%s)' % date)
        ax = plt.subplot(212)
        m.pcolormesh(x, y, pred[d], **plot_kwargs)
        m.drawcoastlines()
        m.drawparallels(np.arange(0., 91., 45.))
        m.drawmeridians(np.arange(0., 361., 90.))
        ax.set_title('%s at $t=%d$ (%s)' % (model_title, hours, date))
        if out_directory:
            plt.savefig('%s/%05d.png' % (out_directory, d), bbox_inches='tight', dpi=150)
        fig.clear()

# Sample usage of the plot_movie function
lat = np.linspace(-90, 90, 50)  # Adjust as per your data
lon = np.linspace(-180, 180, 50)  # Adjust as per your data
val = np.random.rand(10, 50, 50)  # Simulated data
pred = np.random.rand(10, 50, 50)  # Simulated data
dates = pd.date_range(start='2020-01-01', periods=10, freq='D')

# Create Basemap instance
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

# Call the function
plot_movie(m, lat, lon, val, pred, dates, model_title='Neural Net Prediction', out_directory='output')

