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


max_rot_lp = []
max_rot_hp = []
max_disp_lp = []
max_disp_hp = []
max_acc_lp = []
max_acc_hp = []
ts_rot_lp = []
ts_rot_hp = []
ts_disp_lp = []
ts_disp_hp = []
ts_acc_lp = []
ts_acc_hp = []

for station_name, Lat, response in zip(['NA01', 'MDSA0'],[24.46760, 24.02305], [418675, 419430]):

    #makeAnglesHualien_lat_v3(station_name=station_name, starttime=starttime, endtime=endtime, ampscale = 1,latitude=Lat, fromdata = True, plot=True, savedate=True, folder = 'All_EQ')

    #correctAccelerationHualien_v2(station_name, starttime, endtime, ampscale=1, response =response,plot=True, savedate=False, folder='All_EQ')

    '''both_maxi, both_ts = filter_plotly_maxy_Hualien_v2(station_name, starttime, endtime, ampscale=1,
                                                       response =response, magnitude=7.4, lpfreq=0.1,
                                                       hpfreq=0.1, plot=True, show=True)'''
    ampscale = 1
    magnitude = 7.4
    lpfreq = 0.1
    hpfreq = 0.1
    plot = True
    show = True

    # make figure
    fig, axs = plt.subplots(3,1)
    # load data
    df = 50
    lenmean = 4 * df
    root_save = Hroot_processeddata
    obs_acc_ = read('%s/TW.%s..HLE.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLN.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLZ.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    # inv = read_inventory(Hroot_originaldata + '/station.xml')
    axs[0].plot(obs_acc_.select(channel='HLE')[0].times('matplotlib'), obs_acc_.select(channel='HLE')[0].data, color='k')
    axs[1].plot(obs_acc_.select(channel='HLN')[0].times('matplotlib'), obs_acc_.select(channel='HLN')[0].data, color='k')
    axs[2].plot(obs_acc_.select(channel='HLZ')[0].times('matplotlib'), obs_acc_.select(channel='HLZ')[0].data, color='k')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime - 5, endtime + 5)
    axs[0].plot(obs_acc_.select(channel='HLE')[0].times('matplotlib'), obs_acc_.select(channel='HLE')[0].data, color='grey')
    axs[1].plot(obs_acc_.select(channel='HLN')[0].times('matplotlib'), obs_acc_.select(channel='HLN')[0].data, color='grey')
    axs[2].plot(obs_acc_.select(channel='HLZ')[0].times('matplotlib'), obs_acc_.select(channel='HLZ')[0].data, color='grey')

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    # obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    # obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True)

    axs[0].plot(obs_acc_.select(channel='HLE')[0].times('matplotlib'), obs_acc_.select(channel='HLE')[0].data, color='lightgrey')
    axs[1].plot(obs_acc_.select(channel='HLN')[0].times('matplotlib'), obs_acc_.select(channel='HLN')[0].data, color='lightgrey')
    axs[2].plot(obs_acc_.select(channel='HLZ')[0].times('matplotlib'), obs_acc_.select(channel='HLZ')[0].data, color='lightgrey')

    obs_acc_ = obs_acc_.interpolate(sampling_rate=df,starttime=starttime)

    axs[0].plot(obs_acc_.select(channel='HLE')[0].times('matplotlib'), obs_acc_.select(channel='HLE')[0].data, color='lightblue')
    axs[1].plot(obs_acc_.select(channel='HLN')[0].times('matplotlib'), obs_acc_.select(channel='HLN')[0].data, color='lightblue')
    axs[2].plot(obs_acc_.select(channel='HLZ')[0].times('matplotlib'), obs_acc_.select(channel='HLZ')[0].data, color='lightblue')

    for tr in obs_acc_:
        tr.data = tr.data * ampscale / response

    obs_acc_ = obs_acc_.slice(starttime, endtime)
    # obs_acc_.plot()

    fig, axs = plt.subplots(3,1)
    axs[0].plot(obs_acc_.select(channel='HLE')[0].times('matplotlib'), obs_acc_.select(channel='HLE')[0].data, color='blue')
    axs[1].plot(obs_acc_.select(channel='HLN')[0].times('matplotlib'), obs_acc_.select(channel='HLN')[0].data, color='blue')
    axs[2].plot(obs_acc_.select(channel='HLZ')[0].times('matplotlib'), obs_acc_.select(channel='HLZ')[0].data, color='blue')

    obs_acc = numpy.vstack([obs_acc_.select(channel='HLE')[0].data,
                            obs_acc_.select(channel='HLN')[0].data,
                            obs_acc_.select(channel='HLZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :lenmean]), numpy.mean(obs_acc[1, :lenmean]), numpy.mean(obs_acc[2, :lenmean])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    for filter_type, freq, ii, col in zip(['lowpass', 'highpass'], [lpfreq, hpfreq], range(2), ['lightgrey','lightblue']):
        obs_acc_lp = obs_acc_.copy().filter(filter_type, freq=freq, zerophase=True)
        axs[0].plot(obs_acc_lp.select(channel='HLE')[0].times('matplotlib'), obs_acc_lp.select(channel='HLE')[0].data, color=col)
        axs[1].plot(obs_acc_lp.select(channel='HLN')[0].times('matplotlib'), obs_acc_lp.select(channel='HLN')[0].data, color=col)
        axs[2].plot(obs_acc_lp.select(channel='HLZ')[0].times('matplotlib'), obs_acc_lp.select(channel='HLZ')[0].data, color=col)

    plt.show()