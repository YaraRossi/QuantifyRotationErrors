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


def eq_kilauea(min_mag=4):
    root = root_originaldata
    file_ = pd.read_csv('%s/Kilauea_EQ_201807_3MW.txt' % root, sep=',')

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
def makeAnglesKilauea_lat_v3(date_name='date_name', starttime='starttime', endtime='endtime', ampscale = 1, latitude=1, plot=True, savedate=False, folder = 'putsomethinghere'):
    """
        Process seismic data to calculate attitude angles and rotation rates, and optionally plot or save the results.

        Parameters:
        date_name (str): The date for data retrieval.
        starttime (str): The start time of the data slice.
        endtime (str): The end time of the data slice.
        ampscale (float): Scaling factor applied to the rotation rates. Default is 1.
        plot (bool): Whether to generate plots of the results. Default is True.
        savedate (bool): Whether to save the processed data. Default is False.

        Returns:
        None

        Notes:
        - This function processes seismic data from Kilauea volcano to calculate angles and rotation rates. While
          taking into account the attitude equation and a correction for the earth's rotation rate.
        - It applies preprocessing corrections to the data, such as
            a) slicing, removing sensitivity, rotating to correct orientation, and demeaning.
            b) calculating the Earth's rotation rate either due to the location (Latitude) or the data itself.
            b) scaling the signal on the data, to simulate larger earthquakes.
        - Euler angles and Euler rotation rates are computed using the attitude_equation_simple function.
        - If plot is True, plots of rotation rates and attitude angles are generated.
        - If savedate is True, processed data is saved.
        """
    root_save = '%s/%s' %(root_processeddata,folder)
    obs_rate  = read('%s/Kilauea_%s_HJ1.mseed' % (root_originaldata,date_name))
    obs_rate += read('%s/Kilauea_%s_HJ2.mseed' % (root_originaldata,date_name))
    obs_rate += read('%s/Kilauea_%s_HJ3.mseed' % (root_originaldata,date_name))
    inv = read_inventory(root_originaldata+'/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_rate = obs_rate.slice(starttime, endtime)

    #obs_rate.plot()

    #### 1.3 ####
    # correct stuff:
    # scale data from nrad/s to rad/s
    obs_rate.remove_sensitivity(inventory=inv)

    #obs_rate.plot()

    # orient the sensor correctly. it is oriented 1.8° wrong
    obs_rate.rotate(method='->ZNE', inventory=inv, components=["123"])

    #obs_rate.plot()



    ##############################################################################
    #### 2.0 Process data for attitude correction and get uncorrected angles and rotation rate.
    #### 2.1 Earth's rotation rate####
    # calculate earth rotation rate at that location:
    earth_rr = earth_rotationrate(Latitude=latitude) #19.420908
    # get earth rotation rate from data
    e_rr_fromdata = [mean(obs_rate.select(channel='HJE')[0].data[0:2000]),
                     mean(obs_rate.select(channel='HJN')[0].data[0:2000]),
                     mean(obs_rate.select(channel='HJZ')[0].data[0:2000])]
    print('EQ at time: ', starttime)
    print('Earth Rot rate from Calculations: ', earth_rr)
    print('Earth Rot rate from Data: ', e_rr_fromdata)

    #### 2.2 Demeaning and Scaling of data ####
    # demean obsrate for direct angle calculation, but save in different name.
    obs_rate_demean = obs_rate.copy()
    obs_rate_demean.select(channel='HJE')[0].data = (obs_rate_demean.select(channel='HJE')[0].data-e_rr_fromdata[0])*ampscale
    obs_rate_demean.select(channel='HJN')[0].data = (obs_rate_demean.select(channel='HJN')[0].data-e_rr_fromdata[1])*ampscale
    obs_rate_demean.select(channel='HJZ')[0].data = (obs_rate_demean.select(channel='HJZ')[0].data-e_rr_fromdata[2])*ampscale

    #### 2.3 integrate rotation rate to angles ####
    # instead of doing: obs_angle = obs_rate_demean.copy().integrate()

    # pre attitude correction rotation rate
    pre_ac_rr = numpy.vstack([obs_rate_demean.select(channel='HJE')[0].data,
                              obs_rate_demean.select(channel='HJN')[0].data,
                              obs_rate_demean.select(channel='HJZ')[0].data])
    length = len(pre_ac_rr[0, :])
    obs_angle = numpy.zeros((3, length))
    dt = 1/obs_rate[0].stats.sampling_rate,
    for i in range(1, length):
        obs_angle[:,i] = obs_angle[:, i-1] + dt * pre_ac_rr[:, i-1]

    # pre attitude correction angles
    pre_ac_a = obs_angle

    #### 2.4 Attitude Correction and Earth rotation rate Correction ####
    # apply the attitude correction and the correction for earth's rotation rate.
    for_ac_rr = numpy.vstack([obs_rate_demean.select(channel='HJE')[0].data,
                              obs_rate_demean.select(channel='HJN')[0].data,
                              obs_rate_demean.select(channel='HJZ')[0].data])
    # add the earth rotation rate back.
    for i in range(len(for_ac_rr[0,:])):
        if latitude == 19.420908:
            for_ac_rr[:, i] = for_ac_rr[:, i] + e_rr_fromdata # use earthspin taken from rotational sensor
        else:
            for_ac_rr[:,i] = for_ac_rr[:,i] + earth_rr # use estimated earthspin from latitude calculation

    # run function to get a variety of angles and rotation rate, using a variety of rotation correction schemes
    if latitude == 19.420908:
        euler_a, rot_a_err, euler_rr, rot_rr_err, euler_a_tot, euler_rr_tot = attitude_equation_simple(
            dt=1 / obs_rate[0].stats.sampling_rate,
            obs_rate=for_ac_rr, earth_rr=e_rr_fromdata)  # e_rr_fromdata taken from rotational sensor
    else:
        euler_a, rot_a_err, euler_rr, rot_rr_err, euler_a_tot, euler_rr_tot = attitude_equation_simple(
            dt=1/obs_rate[0].stats.sampling_rate,
            obs_rate=for_ac_rr, earth_rr=earth_rr) #earth_rr from latitude calculation

    ##############################################################################
    #### 3.0 Save stuff ####
    MSEED_obs_a = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_obs_a.select(channel=ch)[0].data = pre_ac_a[i, :]
    MSEED_euler_a = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_euler_a.select(channel=ch)[0].data = euler_a[i, :]
    MSEED_rot_a_err = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_rot_a_err.select(channel=ch)[0].data = rot_a_err[i, :]
    MSEED_euler_a_tot = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_euler_a_tot.select(channel=ch)[0].data = euler_a_tot[i, :]
    MSEED_obs_rr = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_obs_rr.select(channel=ch)[0].data = pre_ac_rr[i, :]
    MSEED_euler_rr = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_euler_rr.select(channel=ch)[0].data = euler_rr[i, :]
    MSEED_rot_rr_err = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_rot_rr_err.select(channel=ch)[0].data = rot_rr_err[i, :]
    MSEED_euler_rr_tot = obs_rate.copy()
    for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
        MSEED_euler_rr_tot.select(channel=ch)[0].data = euler_rr_tot[i, :]

    DATA = [pre_ac_a, pre_ac_rr, euler_a, rot_a_err, euler_rr, rot_rr_err, euler_a_tot, euler_rr_tot]
    NAME = ['obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot',
            'euler_rr_tot']

    if savedate:
        for data, name in zip(DATA, NAME):
            mseed = obs_rate.copy()
            for ch, i in zip(['HJE', 'HJN', 'HJZ'], range(3)):
                mseed.select(channel=ch)[0].data = data[i, :]
                starttime_str = str(starttime)
                new_starttime = starttime_str.replace(':', '_')
                new_starttime = new_starttime.replace('-', '_')
                new_starttime = new_starttime.replace('.', '_')
                mseed.select(channel=ch).write(root_save + '/Kilauea_' + str(new_starttime) + '_'+str(ampscale)+
                                               '_lat' + str(latitude) + '_' + name + '_' + ch + '.mseed')

    ##############################################################################
    #### 4.0 plot stuff ####
    if plot:
        filtertype = ['highpass','lowpass']
        direction = ['East','North','Up']

        fig, axs = plt.subplots(3, 2, figsize=(11,5))
        fig.suptitle('Rotation rate [rad/s]')
        for j in range(2):
            MSEED_obs_rr_f = MSEED_obs_rr.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_rr_f = MSEED_euler_rr.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_rr_err_f = MSEED_euler_rr_tot.copy().filter(filtertype[j], freq=0.1)
            ch = ['HJE','HJN','HJZ']
            for i in range(3):
                ax = axs[i,j]
                ax.plot(MSEED_obs_rr_f.select(channel=ch[i])[0].times(),
                        MSEED_obs_rr_f.select(channel=ch[i])[0].data, label='obs_rr')
                ax.plot(MSEED_euler_rr_f.select(channel=ch[i])[0].times(),
                        MSEED_euler_rr_f.select(channel=ch[i])[0].data, label='euler_rr')
                ax.plot(MSEED_euler_rr_err_f.select(channel=ch[i])[0].times(),
                        MSEED_euler_rr_err_f.select(channel=ch[i])[0].data, '--', label='euler_rr_tot')
                if j == 0:
                    ax.set_ylabel(direction[i])
            axs[0,j].set_title('%s at 0.1 Hz' % filtertype[j])
            ax.set_xlabel('Time [s]')
        ax.legend()
        #plt.savefig('RotationRate_%s.png' % date_name, dpi=300, bbox_inches='tight')

        fig, axs = plt.subplots(3, 2, figsize=(11,5))
        fig.suptitle('Differences in Correction schemes: Rotation rate [rad/s]')
        for j in range(2):
            MSEED_obs_rr_f = MSEED_obs_rr.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_rr_f = MSEED_euler_rr.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_rr_err_f = MSEED_euler_rr_tot.copy().filter(filtertype[j], freq=0.1)
            ch = ['HJE','HJN','HJZ']
            for i in range(3):
                ax = axs[i,j]
                ax.plot(MSEED_obs_rr_f.select(channel=ch[i])[0].times(),
                        MSEED_obs_rr_f.select(channel=ch[i])[0].data - MSEED_euler_rr_f.select(channel=ch[i])[0].data,
                        label='diff obs und euler')
                ax.plot(MSEED_euler_rr_f.select(channel=ch[i])[0].times(),
                        MSEED_euler_rr_f.select(channel=ch[i])[0].data - MSEED_euler_rr_err_f.select(channel=ch[i])[0].data,
                        '--', label='diff euler und euler earth corr.')
                ax.plot(MSEED_euler_rr_err_f.select(channel=ch[i])[0].times(),
                        MSEED_obs_rr_f.select(channel=ch[i])[0].data - MSEED_euler_rr_err_f.select(channel=ch[i])[0].data,
                        '-.', label='diff obs und euler earth corr.')
                if j == 0:
                    ax.set_ylabel(direction[i])
            axs[0,j].set_title('%s at 0.1 Hz' % filtertype[j])
            ax.set_xlabel('Time [s]')
        ax.legend()
        #plt.savefig('Diff_Corr_Schemes_RotationRate_%s.png' % date_name, dpi=300, bbox_inches='tight')

        fig, axs = plt.subplots(3, 2, figsize=(11,5))
        fig.suptitle('Angle [rad]')
        for j in range(2):
            MSEED_obs_a_f = MSEED_obs_a.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_a_f = MSEED_euler_a.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_a_err_f = MSEED_euler_a_tot.copy().filter(filtertype[j], freq=0.1)
            ch = ['HJE', 'HJN', 'HJZ']
            for i in range(3):
                ax = axs[i, j]
                ax.plot(MSEED_obs_a_f.select(channel=ch[i])[0].times(), MSEED_obs_a_f.select(channel=ch[i])[0].data,
                        label='obs_a')
                ax.plot(MSEED_euler_a_f.select(channel=ch[i])[0].times(), MSEED_euler_a_f.select(channel=ch[i])[0].data,
                        label='euler_a')
                ax.plot(MSEED_euler_a_err_f.select(channel=ch[i])[0].times(),
                        MSEED_euler_a_err_f.select(channel=ch[i])[0].data,
                        '--', label='euler_a_tot')
                if j == 0:
                    ax.set_ylabel(direction[i])
            axs[0,j].set_title('%s at 0.1 Hz' % filtertype[j])
            ax.set_xlabel('Time [s]')
        ax.legend()
        #plt.savefig('Angles_%s.png' % date_name, dpi=300, bbox_inches='tight')

        fig, axs = plt.subplots(3, 2, figsize=(11,5))
        fig.suptitle('Differences in Correction schemes: Angle [rad]')
        for j in range(2):
            MSEED_obs_a_f = MSEED_obs_a.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_a_f = MSEED_euler_a.copy().filter(filtertype[j], freq=0.1)
            MSEED_euler_a_err_f = MSEED_euler_a_tot.copy().filter(filtertype[j], freq=0.1)
            ch = ['HJE', 'HJN', 'HJZ']
            for i in range(3):
                ax = axs[i, j]
                ax.plot(MSEED_obs_a_f.select(channel=ch[i])[0].times(),
                        MSEED_obs_a_f.select(channel=ch[i])[0].data - MSEED_euler_a_f.select(channel=ch[i])[0].data,
                        label='diff obs und euler')
                ax.plot(MSEED_euler_a_f.select(channel=ch[i])[0].times(),
                        MSEED_euler_a_f.select(channel=ch[i])[0].data - MSEED_euler_a_err_f.select(channel=ch[i])[
                            0].data,
                        '--', label='diff euler und euler earth corr.')
                ax.plot(MSEED_euler_a_err_f.select(channel=ch[i])[0].times(),
                        MSEED_obs_a_f.select(channel=ch[i])[0].data - MSEED_euler_a_err_f.select(channel=ch[i])[
                            0].data,
                        '-.', label='diff obs und euler earth corr.')
                if j == 0:
                    ax.set_ylabel(direction[i])
            axs[0,j].set_title('%s at 0.1 Hz' % filtertype[j])
            ax.set_xlabel('Time [s]')
        ax.legend()
        #plt.savefig('Diff_Corr_Schemes_Angles_%s.png' % date_name, dpi=300, bbox_inches='tight')
        plt.show()

############################
#### Start Calculations ####

## 1. amplitude ascaling and latitude changing
# mw 5.3, ml 3.18m mw 5.3
for date_name, starttime, endtime in zip(['201807140400','201807120400', '20180713'],
                                         [obspy.UTCDateTime('2018-07-14T05:07:45'),obspy.UTCDateTime('2018-07-12T05:12:25'),obspy.UTCDateTime('2018-07-13T00:42:12')],
                                         [obspy.UTCDateTime('2018-07-14T05:08:45'),obspy.UTCDateTime('2018-07-12T05:13:25'),obspy.UTCDateTime('2018-07-13T00:43:12')]):
    #if date_name != '20180713':
    #    continue
    for ampscale in [0.0001,0.001,0.01,0.1,1,10,100,1000]:
        makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=19.420908, ampscale=ampscale, plot=False, savedate=True, folder='Scaling')

    for latitude in [0,15,20,30,45,60,75,90]:
        makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=latitude, ampscale=1, plot=False, savedate=True,folder='Latitudes')

print ('done with scaling and latitude changing!')

## 2. Other earthquakes
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
        print(starttime[NN], e)

# here I calculate a couple of eq that are larger and I just want original amplitude scale and original latitude.
# Ml 3.18m, Mw5.3, Mw5.3, Mw5.3, Ml4.36
for date_name, starttime, endtime in zip(['201807120400','20180713', '20180714', '20180715', '20180714'],
         [obspy.UTCDateTime('2018-07-12T05:12:25'),obspy.UTCDateTime('2018-07-13T00:42:12'),obspy.UTCDateTime('2018-07-14T05:07:45'),obspy.UTCDateTime('2018-07-15T13:25:50'),obspy.UTCDateTime('2018-07-14T04:13:18')],
         [obspy.UTCDateTime('2018-07-12T05:13:25'),obspy.UTCDateTime('2018-07-13T00:43:12'),obspy.UTCDateTime('2018-07-14T05:08:45'),obspy.UTCDateTime('2018-07-15T13:26:50'),obspy.UTCDateTime('2018-07-14T04:14:18')]):

    makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=19.420908, ampscale=1, plot=True, savedate=True, folder='All_EQ')
