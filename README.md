# AttitudeEquations


# Define roots
In the file roots.py you need to define the roots to the different folders:

1. Original Data
2. Processed Data: this folders needs to include 3 folders: 'Latitudes', 'Scaling', 'All_EQ'
3. Figures

Folder structure should be like this:
Data
- Original
- Processed
  - All_EQ
  - Latitudes
  - Scaling  



# Import data
## To get data: run find_load_EQ.py

> find_load_EQ.py


# Processing data
## To get data ready: run Kilauea4Paper.py

> Kilauea4Paper.py

It will save timeseries in mseed in a new folder, as defined by user. This is done by running the function
makeAnglesKilauea_lat_v3 which does the attitude correction that includes either or and both of
misorientation correction and earth spin correction.

The timeseries are:
1. amplitude scaling for 3 earthquakes going from *0.0001 to *1000
2. change of latitude from 0° to 90° for 3 earthquakes
3. original amplitude and original latitude for 5 different earthquakes and volcanic eruptions.


# Plotting data
## Plotting the scaling data: run Kilauea4Paper_scalingfiltering.py
## Plotting the latitude data: run Kilauea4Paper_latitudesfiltering.py

> Kilauea4Paper_scalingfiltering.py

> Kilauea4Paper_latitudesfiltering.py

