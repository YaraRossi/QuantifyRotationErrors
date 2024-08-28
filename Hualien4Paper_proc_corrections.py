import numpy
import obspy
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy import read, read_inventory, UTCDateTime
import matplotlib.pyplot as plt
from numpy import mean
from attitudeequation import earth_rotationrate, attitude_equation_simple
from functions import makeAnglesHualien_lat_v3, correctAccelerationHualien_v2, filter_plotly_maxy_Hualien_v2
from roots import get_rootsHualien
Hroot_originaldata, Hroot_savefig, Hroot_processeddata = get_rootsHualien()


#station_name, Lat = 'NA01', 24.46760
station_name, Lat = 'MDSA0', 24.02305
starttime = UTCDateTime('2024-04-02T23:58:05')
endtime = UTCDateTime('2024-04-02T23:59:55')


for station_name, Lat, response in zip(['NA01', 'MDSA0'],[24.46760, 24.02305], [418675, 419430]):

    #makeAnglesHualien_lat_v3(station_name=station_name, starttime=starttime, endtime=endtime, ampscale = 1,latitude=Lat, fromdata = True, plot=False, savedate=True, folder = 'All_EQ')

    #correctAccelerationHualien_v2(station_name, starttime, endtime, ampscale=1, response =response,plot=True, savedate=False, folder='All_EQ')

    both_maxi, both_ts = filter_plotly_maxy_Hualien_v2(station_name, starttime, endtime, ampscale=1,
                                                       response =response, magnitude=7.4, lpfreq=0.1,
                                                       hpfreq=0.1, plot=True, show=False)

