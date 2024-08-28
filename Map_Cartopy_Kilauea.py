import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from functions import eq_kilauea
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()

dates = ['2018-07-13T00:42:27.110Z', '2018-07-14T04:13:33.600Z', '2018-07-14T05:08:03.680Z',
         '2018-07-15T13:26:05.130Z', '2018-07-12T05:12:41.420Z']

file = eq_kilauea(min_mag=3., paper = True)


# Extract latitude and longitude coordinates
coordinates = file[['time', 'latitude', 'longitude', 'mag', 'magType', 'dist']].values.tolist()
station = [19.420908, -155.292023]

# Create a new figure with a GeoAxes object
fig = plt.figure(figsize=(4, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add ESRI Satellite Imagery
esri_img_url = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml'
ax.add_wmts(esri_img_url, 'World_Imagery')

# Add borders, coastlines, and other features
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)

# Plot the coordinates
for time, lat, lon, mag, magtype, dist in coordinates:
    if '07-15' in time:
        ax.plot(lon, lat, marker='*', color='darkred', markersize=9, markeredgecolor='white', markeredgewidth=0.2, transform=ccrs.PlateCarree())
    else:
        ax.plot(lon, lat, marker='*', color='indianred', markersize=6, markeredgecolor='white', markeredgewidth=0.2, transform=ccrs.PlateCarree())
    if lat < 19.3995:
        if mag < 5:
            ax.text(lon, lat, f' Ml {mag}', color = 'white', fontsize=9, ha='left', va='top', transform=ccrs.PlateCarree())
        else:
            ax.text(lon, lat, f' Mw {mag}', color = 'white', fontsize=9, ha='left', va='top', transform=ccrs.PlateCarree())
    else:
        if mag < 5:
            ax.text(lon, lat, f' Ml {mag}', color='white', fontsize=9, ha='left', va='bottom', transform=ccrs.PlateCarree())
        else:
            ax.text(lon, lat, f' Mw {mag}', color='white', fontsize=9, ha='left', va='bottom', transform=ccrs.PlateCarree())

# Plot station
ax.plot(station[1], station[0], marker='^', color='red', markersize=7, markeredgecolor='white', markeredgewidth=0.2, transform=ccrs.PlateCarree())
ax.text(station[1], station[0], ' UWE', color = 'white', fontsize=9, ha='left', transform=ccrs.PlateCarree())

# Set extent around Kilauea volcano (latitude, longitude)
ax.set_extent([-155.305, -155.245, 19.38, 19.44], crs=ccrs.PlateCarree())

# Add a grid
gridlines = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth = 0.2)

# Customize gridline labels
#gridlines.xlines = False
#gridlines.ylines = False
gridlines.top_labels = False
gridlines.right_labels = False

# Display the plot
plt.title('(a)', loc='left')
plt.savefig('%s/Map_Kilauea.png' % root_savefig, dpi=300, bbox_inches='tight')
plt.show()
