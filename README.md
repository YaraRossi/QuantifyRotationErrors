# AttitudeEquations

## Create the virtual environment
Requirements are in: requirements_env.txt or environment.yml

> pip install -r requirements_env.txt
OR
> conda env create -f environment.yml

## Define roots
In the file roots.py you need to define the roots to the different folders:

1. Original Data
2. Processed Data: this folders needs to include 3 folders: 'Latitudes', 'Scaling', 'All_EQ'
3. Figures

My folder structure was like this:
Data
- Original
- Processed
  - All_EQ
  - All_EQbig
  - Latitudes
  - Scaling
- ProcessedHualien
  - All_EQ

Figures
- 4SSA
- 4Correction
  - 3
  - 4
  - 5  


## Import data
To get data: run find_load_EQ.py

> find_load_EQ.py


## Processing rotation data
- To get Kilauea data ready: run Kilauea4Paper_proc_corrections.py

> Kilauea4Paper_proc_rot.py

It will save timeseries in mseed in a new folder, as defined by user. This is done by running the function
makeAnglesKilauea_lat_v3 which does the attitude correction that includes either and/or both of
misorientation correction and earth spin correction.

- To get Hualien data ready: run Hualien4Paper_proc_corrections.py

> Hualien4Paper_proc_rot.py

It will save timeseries in mseed in a new folder, as defined by user. This is done by running the function
makeAnglesHualien_lat_v3 which does the attitude correction that includes either and/or both of
misorientation correction and earth spin correction.


The timeseries are:
1. amplitude scaling for a couple of earthquakes going from *0.0001 to *1000
2. change of latitude from 0° to 90° for a couple of earthquakes
3. original amplitude and original latitude for 7 different earthquakes and volcanic eruptions.


## Plotting data
- Plotting the scaling data: run KilHua4Paper_plot_scalingfiltering.py

> KilHua4Paper_plot_scalingfiltering.py #(Fig. 5)

- Plotting the latitude data: run Kilauea4Paper_latitudesfiltering.py

> Kilauea4Paper_plot_latitudesfiltering.py #(Fig. 6)


- Plotting the timeseries of the Supplementary figures and the Mw 5.3 for lowpass and highpass in main text. 
Run the file KilHua4Paper_proc_corrections.py to get both the timeseries, the timeseries differences and the errors in displacement 
and acceleration.

> KilHua4Paper_proc_corrections.py #(Fig. 4, 7, 8, 9, 10, A1-A12)


- If you want to plot the timeseries of a specific earthquake you can run Kilauea4Paper_plotEQ.py or 
Hualien4Paper_plotEQ.py

> Kilauea4Paper_plotEQ.py  #(Fig. 3a)
> Hualien4Paper_plotEQ #(Fig. 3c)

## Plotting additional figures
- Plotting the two Maps run Map_Cartopy_Hualien.py or Map_Cartopy_Hualien.py.

> Map_Cartopy_Kilauea.py #(Fig. 1a)
> Map_Cartopy_Hualien.py #(Fig. 1b)

- Plotting the Amplitudes for various earthquake magnitudes run Thesis_MagAmp_plot.py

> Thesis_MagAmp_plot.py (Fig. 3b)


