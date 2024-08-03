import numpy
import obspy
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy import read, read_inventory, UTCDateTime
import matplotlib.pyplot as plt
from numpy import mean
from attitudeequation import earth_rotationrate, attitude_equation_simple
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()
from functions import makeAnglesKilauea_lat_v3


def eq_kilauea(min_mag=4):
    root = root_originaldata
    file_ = pd.read_csv('Kilauea_EQ_201807_3MW.txt', sep=',')

    file = file_[file_.mag > min_mag]
    for i in range(len(file['depth'])):
        if file.loc[i, 'depth'] < 0:
            file.loc[i, 'depth'] = 0

    T_lat, T_lon = 19.420908, -155.292023
    model = TauPyModel(model="iasp91")

    file['dist'] = locations2degrees(T_lat, T_lon, file['latitude'], file['longitude'])  # (lat1, long1, lat2, long2)
    arrivaltime = []
    for index, row in file.iterrows():
        arrivaltime.append(model.get_travel_times(source_depth_in_km=abs(row['depth']),
                                                  distance_in_degree=row['dist'])[0].time)
    file['arrivaltime'] = arrivaltime

    return file

########## Being able to chaneg the latitude!!!!
############################
#### Start Calculations ####

## 1. amplitude ascaling and latitude changing
# mw 5.3, ml 3.18m mw 5.3

for date_name, starttime, endtime in zip(['201807140400','201807120400', '20180713'],
                                         [obspy.UTCDateTime('2018-07-14T05:07:49.173848'),obspy.UTCDateTime('2018-07-12T05:12:25'),obspy.UTCDateTime('2018-07-13T00:42:12.61')],
                                         [obspy.UTCDateTime('2018-07-14T05:08:45'),obspy.UTCDateTime('2018-07-12T05:13:25'),obspy.UTCDateTime('2018-07-13T00:43:12')]):
    #if date_name != '20180713':
    #    continue
    for ampscale in [0.0001,0.001,0.01,0.1,1,10,100,1000]:
        makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=19.420908, ampscale=ampscale,
                                 plot=False, savedate=True, folder='Scaling')

    for latitude in [0,15,20,30,45,60,75,90]:
        makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=latitude, ampscale=1, plot=False, savedate=True,folder='Latitudes')

print ('done with scaling and latitude changing!')

'''## 2. Other earthquakes
# get time of various EQ's:
info_eq = eq_kilauea(min_mag=4.0)
ampscale=1

# get start and end times of Earthquakes:
date, arrival, starttime, endtime = [], [], [], []

for date_time,arrival_time in zip(info_eq['time'],info_eq['arrivaltime']):
    date.append(date_time[:10])
    arrival.append(arrival_time)
    starttime.append(UTCDateTime(date_time)+arrival_time-15)
    endtime.append(UTCDateTime(date_time)+arrival_time+30)

# performed on 19. March 2024
for NN in range(len(info_eq['time'])):
    try:
        date_ = date[NN]
        date_name = date_[0:4]+date_[5:7]+date_[8:10]
    except Exception as e:
        print(starttime[NN], e)'''

# here I calculate a couple of eq that are larger and I just want original amplitude scale and original latitude.
# Ml 3.18m, Mw5.3, Mw5.3, Mw5.3, Ml4.36
for date_name, starttime, endtime in zip(['201807120400','20180713', '20180714', '20180715', '20180714'],
         [obspy.UTCDateTime('2018-07-12T05:12:25'),obspy.UTCDateTime('2018-07-13T00:42:12'),obspy.UTCDateTime('2018-07-14T05:07:45'),obspy.UTCDateTime('2018-07-15T13:25:50'),obspy.UTCDateTime('2018-07-14T04:13:18')],
         [obspy.UTCDateTime('2018-07-12T05:13:25'),obspy.UTCDateTime('2018-07-13T00:43:12'),obspy.UTCDateTime('2018-07-14T05:08:45'),obspy.UTCDateTime('2018-07-15T13:26:50'),obspy.UTCDateTime('2018-07-14T04:14:18')]):

    makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=19.420908, ampscale=1, plot=True, savedate=True, folder='All_EQbig')
