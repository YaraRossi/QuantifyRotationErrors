import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()

# Coordinates in the upper part of Taiwan (latitude, longitude)
coordinates = [
    (24.02305, 121.63013, ' MDSA0'),
    (24.46760, 121.79333, ' NA01'),
]
earthquake = [23.836, 121.598, ' 7.4 mw']

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
for lat, lon, name in coordinates:
    ax.plot(lon, lat, marker='^', color='red', markersize=7, markeredgecolor='white', markeredgewidth=0.2, transform=ccrs.PlateCarree())
    ax.text(lon, lat, f'{name}', color='white', fontsize=9, ha='left', transform=ccrs.PlateCarree())

# Plot Earthquake
ax.plot(earthquake[1], earthquake[0], marker='*', color='pink', markersize=7, markeredgecolor='white', markeredgewidth=0.2, transform=ccrs.PlateCarree())
ax.text(earthquake[1], earthquake[0], earthquake[2], color='white', fontsize=9, ha='left', transform=ccrs.PlateCarree())


# Set extent to focus on the upper part of Taiwan (longitude, latitude)
ax.set_extent([121.0, 122.5, 23.5, 25], crs=ccrs.PlateCarree())


# Add a grid
gridlines = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth = 0.2)
# Customize gridline labels
gridlines.top_labels = False
gridlines.right_labels = False

# Display the plot
plt.title('(b)', loc='left')
plt.savefig('%s/Map_Hualien.png' % root_savefig, dpi=300, bbox_inches='tight')
plt.show()
