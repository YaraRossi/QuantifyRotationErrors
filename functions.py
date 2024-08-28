import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import matplotlib.pyplot as plt
import obspy
import numpy
from numpy import sin, cos, tan, pi, mean
from obspy import read, read_inventory
from tqdm import tqdm
from roots import get_roots, get_rootsHualien
root_originaldata, root_savefig, root_processeddata = get_roots()
Hroot_originaldata, Hroot_savefig, Hroot_processeddata = get_rootsHualien()
from attitudeequation import rot_vec, attitude_equation_simple,earth_rotationrate

def eq_kilauea(min_mag=4, paper = False):
    dates = ['2018-07-13T00:42:27.110Z', '2018-07-14T04:13:33.600Z', '2018-07-14T05:08:03.680Z',
             '2018-07-15T13:26:05.130Z', '2018-07-12T05:12:41.420Z']

    root = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    file_ = pd.read_csv('%s/Kilauea_EQ_201807_3MW.txt' % root, sep=',')

    if paper:
        file_ = file_[file_.time.isin(dates)]

    file = file_[file_.mag > min_mag]
    '''for i in range(len(file['depth'])):
        if file.loc[i, 'depth'] < 0:
            file.loc[i, 'depth'] = 0'''

    T_lat, T_lon = 19.420908, -155.292023
    model = TauPyModel(model="iasp91")

    # Calculate distance in degrees
    file['dist_degrees'] = locations2degrees(T_lat, T_lon, file['latitude'], file['longitude'])

    # Convert degrees to radians
    file['dist_radians'] = numpy.deg2rad(file['dist_degrees'])

    # Define Earth's radius in meters
    earth_radius_meters = 6371000  # Average radius of Earth in meters

    # Convert radians to distance in meters
    file['dist'] = file['dist_radians'] * earth_radius_meters

    arrivaltime = []
    for index, row in file.iterrows():
        arrivaltime.append(model.get_travel_times(source_depth_in_km=abs(row['depth']),
                                                  distance_in_degree=row['dist_degrees'])[0].time)
    file['arrivaltime'] = arrivaltime

    return file

# this one is outdated!
'''def makeAnglesKilauea(date_name='date_name', ampscale=1, plot=True):
    # can only handle two dates, for any dates use makeAnglesKilauea_v2

    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    obs_rate  = read('%s/Kilauea_%s_HJ1.mseed' % (root_import,date_name))
    obs_rate += read('%s/Kilauea_%s_HJ2.mseed' % (root_import,date_name))
    obs_rate += read('%s/Kilauea_%s_HJ3.mseed' % (root_import,date_name))
    inv = read_inventory(root_import+'/station.xml')

    #### 1.2 ####
    # slice data to only include EQ

    if date_name == '201807140400':# biggest one:
        starttime=obspy.UTCDateTime('2018-07-14T05:07:45')
        endtime=obspy.UTCDateTime('2018-07-14T05:08:45')

    elif date_name == '201807120400':# smaller one:
        starttime=obspy.UTCDateTime('2018-07-12T05:12:15')
        endtime=obspy.UTCDateTime('2018-07-12T05:13:15')
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
    #### 2.1 ####
    # calculate earth rotation rate at that location:
    earth_rr = earth_rotationrate(Latitude=19.420908)
    # get earth rotation rate from data
    e_rr_fromdata = [mean(obs_rate.select(channel='HJE')[0].data[0:2000]),
                     mean(obs_rate.select(channel='HJN')[0].data[0:2000]),
                     mean(obs_rate.select(channel='HJZ')[0].data[0:2000])]

    #### 2.2 ####
    # demean obsrate for direct angle calculation, but save in different name.
    obs_rate_demean = obs_rate.copy()
    obs_rate_demean.select(channel='HJE')[0].data = (obs_rate_demean.select(channel='HJE')[0].data-e_rr_fromdata[0])*ampscale
    obs_rate_demean.select(channel='HJN')[0].data = (obs_rate_demean.select(channel='HJN')[0].data-e_rr_fromdata[1])*ampscale
    obs_rate_demean.select(channel='HJZ')[0].data = (obs_rate_demean.select(channel='HJZ')[0].data-e_rr_fromdata[2])*ampscale

    # integrate rotation rate to angles:
    # instead of doing: obs_angle = obs_rate_demean.copy().integrate()


    pre_ac_rr = numpy.vstack([obs_rate_demean.select(channel='HJE')[0].data,
                              obs_rate_demean.select(channel='HJN')[0].data,
                              obs_rate_demean.select(channel='HJZ')[0].data])
    length = len(pre_ac_rr[0, :])
    obs_angle = numpy.zeros((3, length))
    dt = 1/obs_rate[0].stats.sampling_rate,
    for i in range(1, length):
        obs_angle[:,i] = obs_angle[:, i-1] + dt * pre_ac_rr[:, i-1]

    pre_ac_a = obs_angle

    #### 2.3 ####
    # apply the attitude correction and the correction for earth's rotation rate.
    for_ac_rr = numpy.vstack([obs_rate_demean.select(channel='HJE')[0].data,
                              obs_rate_demean.select(channel='HJN')[0].data,
                              obs_rate_demean.select(channel='HJZ')[0].data])
    # ad the earth rotation rate back.
    for i in range(len(for_ac_rr[0,:])):
        for_ac_rr[:,i] = for_ac_rr[:,i] + e_rr_fromdata

    euler_a, euler_a_err, euler_rr, euler_rr_err, euler_a_after, euler_rr_after = attitude_equation_simple(dt=1/obs_rate[0].stats.sampling_rate,
                                                             obs_rate=for_ac_rr, earth_rr=e_rr_fromdata) #earth_rr,e_rr_fromdata


    ##############################################################################
    #### 3.0 Save stuff ####
    MSEED_obs_a = obs_rate.copy()
    MSEED_euler_a = obs_rate.copy()
    MSEED_euler_a_err = obs_rate.copy()
    MSEED_obs_rr = obs_rate.copy()
    MSEED_euler_rr = obs_rate.copy()
    MSEED_euler_rr_err = obs_rate.copy()
    MSEED = [MSEED_obs_a,MSEED_obs_rr,MSEED_euler_a,MSEED_euler_a_err,MSEED_euler_rr,MSEED_euler_rr_err]
    DATA = [pre_ac_a,pre_ac_rr,euler_a,euler_a_err,euler_rr,euler_rr_err]
    NAME = ['obs_angle','obs_rr', 'euler_angle','euler_angle_err', 'euler_rr', 'euler_rr_err']

    for mseed, data, name in zip(MSEED,DATA,NAME):
        for ch, i in zip(['HJE','HJN','HJZ'],range(3)):
            mseed.select(channel=ch)[0].data = mseed.select(channel=ch)[0].data*0 + data[i,:]
            mseed.select(channel=ch).write(root_import+'/Kilauea_'+date_name+'_'+str(ampscale)+'_'+name+'_'+ch+'.mseed')

    if plot:

        ##############################################################################
        #### 4.0 ####
        # plot stuff
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Rotation rate [rad/s]')
        for i in range(3):
            ax = axs[i]
            ax.plot(pre_ac_rr[i, :], label='pre_ac_rr')
            ax.plot(euler_rr[i, :], label='euler_rr')
            ax.plot(euler_rr_err[i, :], '--', label='euler_rr_err')
            ax.plot(euler_rr_after[i, :], '-.', label='euler_rr_after')
        ax.legend()

        fig, axs = plt.subplots(3, 1)
        fig.suptitle('ATT Earth Error Rotation rate [rad/s]')
        for i in range(3):
            ax = axs[i]
            ax.plot(pre_ac_rr[i, 1:] - euler_rr[i, 1:], label='diff obs und euler')
            ax.plot(euler_rr[i, 1:] - euler_rr_err[i, 1:], '--', label='diff euler und euler earth corr.')
            ax.plot(pre_ac_rr[i, 1:] - euler_rr_err[i, 1:], '-.', label='diff obs und euler earth corr.')
        ax.legend()

        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Angle [rad]')
        for i in range(3):
            ax = axs[i]
            ax.plot(pre_ac_a[i, :], label='pre_ac_a')
            ax.plot(euler_a[i, :], label='euler_a')
            ax.plot(euler_a_err[i, :], '--', label='euler_a_err')
            ax.plot(euler_a_after[i, :], '-.', label='euler_a_after')
        ax.legend()

        fig, axs = plt.subplots(3, 1)
        fig.suptitle('ATT Earth Error Angle [rad]')
        for i in range(3):
            ax = axs[i]
            ax.plot(pre_ac_a[i, 1:] - euler_a[i, 1:], label='diff obs und euler')
            ax.plot(euler_a[i, 1:] - euler_a_err[i, 1:], '--', label='diff euler und euler earth corr.')
            ax.plot(pre_ac_a[i, 1:] - euler_a_err[i, 1:], '-.', label='diff obs und euler earth corr.')
        ax.legend()

        plt.show()'''

def correctAccelerationKilauea(date_name='date_name', ampscale=1, plot=True):
    ##############################################################################
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_import, date_name))

    inv = read_inventory(root_import + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    if date_name == '201807140400':  # biggest one:
        starttime = obspy.UTCDateTime('2018-07-14T05:07:45')
        endtime = obspy.UTCDateTime('2018-07-14T05:08:45')

    elif date_name == '201807120400':  # smaller one:
        starttime = obspy.UTCDateTime('2018-07-12T05:12:15')
        endtime = obspy.UTCDateTime('2018-07-12T05:13:15')
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data*ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :200]), numpy.mean(obs_acc[1, :200]), numpy.mean(obs_acc[2, :200])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    obs_a_E = read('%s/Kilauea_%s_%s_obs_angle_HJE.mseed' % (root_import, date_name, ampscale))
    obs_a_N = read('%s/Kilauea_%s_%s_obs_angle_HJN.mseed' % (root_import, date_name, ampscale))
    obs_a_Z = read('%s/Kilauea_%s_%s_obs_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/Kilauea_%s_%s_euler_angle_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_N = read('%s/Kilauea_%s_%s_euler_angle_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_Z = read('%s/Kilauea_%s_%s_euler_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/Kilauea_%s_%s_euler_angle_err_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_err_N = read('%s/Kilauea_%s_%s_euler_angle_err_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_err_Z = read('%s/Kilauea_%s_%s_euler_angle_err_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(
        sampling_rate=df)
    euler_a_err = numpy.vstack(
        [euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])
    if plot:
        fig, axs = plt.subplots(3, 1)

        for ax, ch in zip(axs.flat, ['HJE', 'HJN', 'HJZ']):
            for data, liner, color, label in zip([obs_a_all, euler_a_all, euler_a_err_all],
                                                 ['-', '--', '-.', ],
                                                 ['k', 'grey', 'lightgrey'],
                                                 ['obs_a_all', 'euler_a_all', 'euler_a_err_all']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Angle [rad]')
        ax.legend()
        ax.set_xlabel('Time')

    ################################################################################################
    #### 2.0 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_obs_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_err_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

    ################################################################################################
    #### 3.0 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    dis_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    if plot:
        # in acceleration
        fig, axs = plt.subplots(3, 1)
        for ax, ch, i in zip(axs.flat, ['HNE', 'HNN', 'HNZ'], range(3)):
            for data, liner, color, label in zip(
                    [obs_acc_, acc_obs_demean, acc_obs_rc_m, acc_euler_rc_m, acc_euler_err_rc_m],
                    ['-', 'dotted', '--', '-.', 'dotted'],
                    ['k', 'k', 'navy', 'blue', 'lightsteelblue'],
                    ['obs_acc_', 'dis_obs_demean', 'acc_obs_rc_m', 'acc_euler_rc_m', 'acc_euler_err_rc_m']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Acc. [m/s/s]')
        ax.legend()
        ax.set_xlabel('Time')

        # in displacement
        fig, axs = plt.subplots(3, 1)
        for ax, ch in zip(axs.flat, ['HNE', 'HNN', 'HNZ']):
            for data, liner, color, label in zip([disp_obs, dis_obs_demean, disp_obs_rc, disp_euler_rc, disp_euler_err_rc],
                                                 ['-', 'dotted', '--', '-.', 'dotted'],
                                                 ['k', 'k', 'navy', 'blue', 'lightsteelblue'],
                                                 ['disp_obs', 'dis_obs_demean', 'disp_obs_rc', 'disp_euler_rc',
                                                  'disp_euler_err_rc']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Disp. [m]')
        ax.legend()
        ax.set_xlabel('Time')

        plt.show()

# Upgrade from first version:
# now it should be posssible to define the time and day more easily.
# also includes rotation angles corrected for err.
def correctAccelerationKilauea_v2(date_name='date_name', starttime='starttime', endtime='endtime',
                                  ampscale=1, plot=True, savedate=False, folder = 'putsomethinghere'):
    ##############################################################################
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50

    root_save = '%s/%s' %(root_processeddata,folder)
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_originaldata, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_originaldata, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_originaldata, date_name))
    inv = read_inventory(root_originaldata + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data*ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :200]), numpy.mean(obs_acc[1, :200]), numpy.mean(obs_acc[2, :200])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    starttime_str = str(starttime)
    new_starttime = starttime_str.replace(':', '_')
    new_starttime = new_starttime.replace('-', '_')
    new_starttime = new_starttime.replace('.', '_')
    # 'obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot', 'euler_rr_tot'
    obs_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import rot angle with earth rotation correction
    rot_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_all = rot_a_err_E
    rot_a_err_all += rot_a_err_N
    rot_a_err_all += rot_a_err_Z
    rot_a_err_all = rot_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    rot_a_err = numpy.vstack([rot_a_err_all.select(channel='HJE')[0].data, rot_a_err_all.select(channel='HJN')[0].data,
         rot_a_err_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(
        sampling_rate=df)
    euler_a_err = numpy.vstack(
        [euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])
    if plot:
        fig, axs = plt.subplots(3, 1)

        for ax, ch in zip(axs.flat, ['HJE', 'HJN', 'HJZ']):
            for data, liner, color, label in zip([obs_a_all, euler_a_all, rot_a_err_all, euler_a_err_all],
                                                 ['-', '--', '-.', 'dotted'],
                                                 ['k', 'grey', 'lightgrey', 'lightsteelblue'],
                                                 ['obs_a_all', 'euler_a_all', 'rot_a_err_all' ,'euler_a_tot_all']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Angle [rad]')
        ax.legend()
        ax.set_xlabel('Time')

    ################################################################################################
    #### 2.0 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_rot_err_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in tqdm(range(NN)):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * rot_a_err[0, i]
        theta = scale * rot_a_err[1, i]
        psi = scale * rot_a_err[2, i]
        acc_rot_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

    ################################################################################################
    #### 3.0 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_rot_err_rc_m = obs_acc_.copy()
    acc_rot_err_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_rot_err_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_rot_err_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_rot_err_rc = acc_rot_err_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    if plot:
        # in acceleration
        fig, axs = plt.subplots(3, 1)
        for ax, ch, i in zip(axs.flat, ['HNE', 'HNN', 'HNZ'], range(3)):
            for data, liner, color, label in zip(
                    [obs_acc_, acc_obs_demean, acc_obs_rc_m, acc_euler_rc_m, acc_rot_err_rc_m, acc_euler_err_rc_m],
                    ['-', 'dotted', '--', '-.', '--', 'dotted'],
                    ['k', 'k', 'navy', 'blue', 'lightblue', 'lightsteelblue'],
                    ['acc_obs', 'acc_dm', 'acc_dm_rc_rot', 'acc_dm_rc_euler', 'acc_dm_rc_rot_err', 'acc_dm_rc_euler_err']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Acc. [m/s/s]')
        ax.legend()
        ax.set_xlabel('Time')
        fig.savefig('%s/TS_acc_rc_%s.png' %(root_savefig, new_starttime), dpi=300, bbox_inches='tight')

        # in displacement
        fig, axs = plt.subplots(3, 1)
        for ax, ch in zip(axs.flat, ['HNE', 'HNN', 'HNZ']):
            for data, liner, color, label in zip([disp_obs, disp_obs_demean, disp_obs_rc, disp_euler_rc, disp_rot_err_rc, disp_euler_err_rc],
                                                 ['-', 'dotted', '--', '-.', '--', 'dotted'],
                                                 ['k', 'k', 'navy', 'blue', 'lightblue', 'lightsteelblue'],
                                                 ['disp_obs', 'disp_dm', 'disp_dm_rc_rot', 'disp_dm_rc_euler', 'disp_dm_rc_rot_err', 'disp_dm_rc_euler_err']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Disp. [m]')
        ax.legend()
        ax.set_xlabel('Time')
        fig.savefig('%s/TS_disp_rc_%s.png' %(root_savefig, new_starttime), dpi=300, bbox_inches='tight')

        plt.show()
def plotnicelyKilauea(date_name='date_name', ampscale=1, plot=True):
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding'
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_import, date_name))

    inv = read_inventory(root_import + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    if date_name == '201807140400':  # biggest one:
        starttime = obspy.UTCDateTime('2018-07-14T05:07:45')
        endtime = obspy.UTCDateTime('2018-07-14T05:08:45')

    elif date_name == '201807120400':  # smaller one:
        starttime = obspy.UTCDateTime('2018-07-12T05:12:15')
        endtime = obspy.UTCDateTime('2018-07-12T05:13:15')
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data*ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :200]), numpy.mean(obs_acc[1, :200]), numpy.mean(obs_acc[2, :200])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    obs_a_E = read('%s/Kilauea_%s_%s_obs_angle_HJE.mseed' % (root_import, date_name, ampscale))
    obs_a_N = read('%s/Kilauea_%s_%s_obs_angle_HJN.mseed' % (root_import, date_name, ampscale))
    obs_a_Z = read('%s/Kilauea_%s_%s_obs_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/Kilauea_%s_%s_euler_angle_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_N = read('%s/Kilauea_%s_%s_euler_angle_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_Z = read('%s/Kilauea_%s_%s_euler_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/Kilauea_%s_%s_euler_angle_err_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_err_N = read('%s/Kilauea_%s_%s_euler_angle_err_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_err_Z = read('%s/Kilauea_%s_%s_euler_angle_err_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(
        sampling_rate=df)
    euler_a_err = numpy.vstack(
        [euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])

    #### 2.0 Processing of RMSE ####
    #### 2.1 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_obs_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_err_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

    ################################################################################################
    #### 2.2 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    dis_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    #### 2.3 Timeseries difference ####
    time = obs_acc_[0].times()

    # rotation
    base = obs_a
    data = [obs_a, euler_a, euler_a_err]
    NN = len(data)
    TSdiff_rot = [[[], [], []],
                  [[], [], []],
                  [[], [], []]]
    TS_rot = [[[], [], []],
              [[], [], []],
              [[], [], []]]
    for j in range(NN):
        channel = data[j]
        for i in range(3):
            diff = channel[i, :] - base[i, :]
            TSdiff_rot[j][i] = diff
            TS_rot[j][i] = channel[i, :]

    # acceleration
    base = obs_acc
    data = [obs_acc, obs_acc_demean, acc_obs_rc, acc_euler_rc, acc_euler_err_rc]
    NN = len(data)
    TSdiff_acc = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
    TS_acc = [[[], [], []],
              [[], [], []],
              [[], [], []],
              [[], [], []],
              [[], [], []]]
    for j in range(NN):
        channel = data[j]
        for i in range(3):
            diff = channel[i, :] - base[i, :]
            TSdiff_acc[j][i] = diff
            TS_acc[j][i] = channel[i, :]

    # displacement
    base = disp_obs
    data = [disp_obs, dis_obs_demean, disp_obs_rc, disp_euler_rc, disp_euler_err_rc]
    ch = ['HNE', 'HNN', 'HNZ']
    NN = len(data)
    TSdiff_disp = [[[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []]]
    TS_disp = [[[], [], []],
               [[], [], []],
               [[], [], []],
               [[], [], []],
               [[], [], []]]
    for j in range(NN):
        channel = data[j]
        for i in range(3):
            diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
            TSdiff_disp[j][i] = diff
            TS_disp[j][i] = channel.select(channel=ch[i])[0].data

    if plot:
        #### 2.4 Plot stuff ####

        fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
        plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
        plt.suptitle('TimeSeries Comparisons for day: ' + date_name)
        x_ticks = ['obs. trans', 'obs. trans \ndemeaned', 'observed', 'euler', 'euler + earth rc']

        # rotation
        color = ['darkred', 'red', 'lightcoral']
        liner = ['--', '-.', 'dotted']
        direction = ['East', 'North', 'Up']
        for i in range(3):
            ax = axs[0 + i]
            ax.set_ylabel('%s [rad]' % direction[i])
            for j in range(3):
                ax.plot(time, TS_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
        ax.legend(loc='upper left')

        x_ticks = ['observed', 'obs. trans demeaned', 'rc. observed', 'rc. euler', 'rc. euler + earth rc']
        liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted']
        direction = ['East', 'North', 'Up']
        # displacement
        color = ['k', 'k', 'grey', 'darkgrey', 'lightgrey']
        for i in range(3):
            ax = axs[3 + i]
            ax.set_ylabel('%s [m]' % direction[i])
            for j in range(5):
                ax.plot(time, TS_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
        ax.legend(loc='upper left')

        # acceleration
        color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'lightsteelblue']
        for i in range(3):
            ax = axs[6 + i]
            ax.set_ylabel('%s [m/s/s]' % direction[i])
            for j in range(5):
                ax.plot(time, TS_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
        ax.legend(loc='upper left')

        ax.set_xlabel('Time [s]')
        plt.savefig('%s/TS_%s_%s.png' % (root_savefig, ampscale, date_name), dpi=300)

        ########################
        fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
        plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
        plt.suptitle('TimeSeries Difference Comparisons for day: ' + date_name)
        x_ticks = ['obs. trans', 'obs. trans \ndemeaned', 'observed', 'euler', 'euler + earth rc']

        # rotation
        color = ['darkred', 'red', 'lightcoral']
        liner = ['--', '-.', 'dotted']
        direction = ['East', 'North', 'Up']
        for i in range(3):
            ax = axs[0 + i]
            ax.set_ylabel('%s [rad]\nDifference \nto observed' % direction[i])
            for j in range(3):
                ax.plot(time, TSdiff_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
        ax.legend(loc='upper left')

        x_ticks = ['observed', 'obs. trans demeaned', 'rc. observed', 'rc. euler', 'rc. euler + earth rc']
        liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted']
        direction = ['East', 'North', 'Up']
        # displacement
        color = ['k', 'k', 'grey', 'darkgrey', 'lightgrey']
        for i in range(3):
            ax = axs[3 + i]
            ax.set_ylabel('%s [m]\nDifference \nto observed' % direction[i])
            for j in range(5):
                ax.plot(time, TSdiff_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
        ax.legend(loc='upper left')

        # acceleration
        color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'lightsteelblue']
        for i in range(3):
            ax = axs[6 + i]
            ax.set_ylabel('%s [m/s/s]\nDifference \nto observed' % direction[i])
            for j in range(5):
                ax.plot(time, TSdiff_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
        ax.legend(loc='upper left')

        ax.set_xlabel('Time [s]')
        plt.savefig('%s/TSdiff_%s_%s.png' % (root_savefig, ampscale, date_name), dpi=300)

        plt.show()

def filter_plotly_Kilauea(date_name='date_name', ampscale='ampscale', lpfreq = 'freq', hpfreq = 'freq', plot=True, show=False):
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    freq = lpfreq
    lenmean = 4*df
    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding'
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_import, date_name))

    inv = read_inventory(root_import + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    if date_name == '201807140400':  # biggest one:
        starttime = obspy.UTCDateTime('2018-07-14T05:07:45')
        endtime = obspy.UTCDateTime('2018-07-14T05:08:45')

    elif date_name == '201807120400':  # smaller one:
        starttime = obspy.UTCDateTime('2018-07-12T05:12:15')
        endtime = obspy.UTCDateTime('2018-07-12T05:13:15')
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data * ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :lenmean]), numpy.mean(obs_acc[1, :lenmean]), numpy.mean(obs_acc[2, :lenmean])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    obs_a_E = read('%s/Kilauea_%s_%s_obs_angle_HJE.mseed' % (root_import, date_name, ampscale))
    obs_a_N = read('%s/Kilauea_%s_%s_obs_angle_HJN.mseed' % (root_import, date_name, ampscale))
    obs_a_Z = read('%s/Kilauea_%s_%s_obs_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/Kilauea_%s_%s_euler_angle_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_N = read('%s/Kilauea_%s_%s_euler_angle_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_Z = read('%s/Kilauea_%s_%s_euler_angle_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/Kilauea_%s_%s_euler_angle_err_HJE.mseed' % (root_import, date_name, ampscale))
    euler_a_err_N = read('%s/Kilauea_%s_%s_euler_angle_err_HJN.mseed' % (root_import, date_name, ampscale))
    euler_a_err_Z = read('%s/Kilauea_%s_%s_euler_angle_err_HJZ.mseed' % (root_import, date_name, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(
        sampling_rate=df)
    euler_a_err = numpy.vstack(
        [euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])

    #### 2.0 Processing of RMSE ####
    #### 2.1 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_obs_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi
        # acc_euler_err_rc[:,i] = rot_vec(phi, theta, psi, gravi)-gravi

    ################################################################################################
    #### 2.2 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    #### 3.0 filter the shit out of this ####
    # 3.1 Lowpass and highpass filter
    both_maxi = [[],[]]
    for filter_type, freq, ii in zip(['lowpass','highpass'],[lpfreq,hpfreq], range(2)):
        # Angles
        obs_a_all_lp = obs_a_all.copy().filter(filter_type, freq = freq, zerophase = True)
        euler_a_all_lp = euler_a_all.copy().filter(filter_type, freq = freq, zerophase = True)
        euler_a_err_all_lp = euler_a_err_all.copy().filter(filter_type, freq = freq, zerophase = True)
        # Disp [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp]
        disp_obs_lp = disp_obs.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_obs_demean_lp = disp_obs_demean.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_obs_rc_lp = disp_obs_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_euler_rc_lp = disp_euler_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_euler_err_rc_lp = disp_euler_err_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        # Acc [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp]
        obs_acc_lp = obs_acc_.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_obs_demean_lp = acc_obs_demean.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_obs_rc_m_lp = acc_obs_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_euler_rc_m_lp = acc_euler_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_euler_err_rc_m_lp = acc_euler_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)

        #### 2.3 Timeseries difference ####
        time = obs_acc_[0].times()

        # rotation
        base = obs_a_all_lp
        data = [obs_a_all_lp, euler_a_all_lp, euler_a_err_all_lp]
        NN = len(data)
        ch = ['HJE','HJN','HJZ']
        TSdiff_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_rot = [[[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_rot[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_rot[j][i] = mini
                else:
                    TSmax_rot[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_rot[j][i] = mini
                    else:
                        TSmax_rot[j][i] = maxi

                TS_rot[j][i] = channel.select(channel=ch[i])[0].data

        # acceleration
        base = obs_acc_lp
        data = [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp]
        NN = len(data)
        ch = ['HNE', 'HNN', 'HNZ']
        TSdiff_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_acc = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_acc[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_acc[j][i] = mini
                else:
                    TSmax_acc[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_acc[j][i] = mini
                    else:
                        TSmax_acc[j][i] = maxi
                TS_acc[j][i] = channel.select(channel=ch[i])[0].data

        # displacement
        base = disp_obs_lp
        data = [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp]
        ch = ['HNE', 'HNN', 'HNZ']
        NN = len(data)
        TSdiff_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TSmax_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TS_disp = [[[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_disp[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_disp[j][i] = mini
                else:
                    TSmax_disp[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_disp[j][i] = mini
                    else:
                        TSmax_disp[j][i] = maxi

                TS_disp[j][i] = channel.select(channel=ch[i])[0].data



        if plot:
            #### 2.4 Plot stuff ####

            ######################## Plot Time Series
            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            plt.suptitle('TimeSeries Comparisons for day: ' + date_name)
            x_ticks = ['obs. trans', 'obs. trans \ndemeaned', 'observed', 'euler', 'euler + earth rc']

            # rotation
            color = ['darkred', 'red', 'lightcoral']
            liner = ['--', '-.', 'dotted']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]' % direction[i])
                for j in range(3):
                    ax.plot(time, TS_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
            ax.legend(loc='upper left')

            x_ticks = ['observed', 'obs. trans demeaned', 'rc. observed', 'rc. euler', 'rc. euler + earth rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]' % direction[i])
                for j in range(5):
                    ax.plot(time, TS_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left')

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'lightsteelblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]' % direction[i])
                for j in range(5):
                    ax.plot(time, TS_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left')

            ax.set_xlabel('Time [s]')
            plt.savefig('%s/TS_%s__%s_%s.png' % (root_savefig, filter_type, ampscale, date_name), dpi=300)

            ######################## Plot Difference

            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            plt.suptitle('TimeSeries Difference Comparisons for day: ' + date_name)
            x_ticks = ['obs. trans', 'obs. trans \ndemeaned', 'observed', 'euler', 'euler + earth rc']

            # rotation
            color = ['darkred', 'red', 'lightcoral']
            liner = ['--', '-.', 'dotted']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]\nDifference \nto observed' % direction[i])
                for j in range(3):
                    ax.plot(time, TSdiff_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    ax.axhline(TSmax_rot[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            x_ticks = ['observed', 'obs. trans demeaned', 'rc. observed', 'rc. euler', 'rc. euler + earth rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]\nDifference \nto observed' % direction[i])
                for j in range(5):
                    ax.plot(time, TSdiff_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.axhline(TSmax_disp[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'lightsteelblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]\nDifference \nto observed' % direction[i])
                for j in range(5):
                    ax.plot(time, TSdiff_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.axhline(TSmax_acc[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            ax.set_xlabel('Time [s]')
            plt.savefig('%s/TSdiff_%s_%s_%s.png' % (root_savefig, filter_type, ampscale, date_name), dpi=300)
            if show:
                plt.show()

        both_maxi[ii] = [TSmax_rot,TSmax_disp,TSmax_acc]
        # TSmax_disp
        # disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp
        # all three components

        # TSmax_acc
        # obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp
        # all three components

        # TSmax_rot
        # obs_a_all_lp, euler_a_all_lp, euler_a_err_all_lp
        # all three components
    return both_maxi

# below here are new functions that will work with more data from various EQ's.

####################
# Upgrade from v2
# latitude and scaling can be changed here simultaneously:
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


def filter_plotly_maxy_Kilauea(date_name='date_name', starttime='starttime', endtime='endtime', folder='mag_int',
                               ampscale='ampscale',  magnitude='magnitude', lpfreq = 'freq', hpfreq = 'freq',
                               plot=True, show=False):
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    freq = lpfreq
    lenmean = 4*df
    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    root_processed = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/Processed'
    root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4Correction/%s' %folder
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_import, date_name))

    inv = read_inventory(root_import + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data * ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :lenmean]), numpy.mean(obs_acc[1, :lenmean]), numpy.mean(obs_acc[2, :lenmean])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    starttime_str = str(starttime)
    new_starttime = starttime_str.replace(':', '_')
    new_starttime = new_starttime.replace('-', '_')
    new_starttime = new_starttime.replace('.', '_')
    # 'obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot', 'euler_rr_tot'
    obs_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import rot angle with earth rotation correction
    rot_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_all = rot_a_err_E
    rot_a_err_all += rot_a_err_N
    rot_a_err_all += rot_a_err_Z
    rot_a_err_all = rot_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    rot_a_err = numpy.vstack([rot_a_err_all.select(channel='HJE')[0].data, rot_a_err_all.select(channel='HJN')[0].data,
                              rot_a_err_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJN.mseed' % ( root_processeddata, new_starttime, ampscale))
    euler_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a_err = numpy.vstack([euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])

    #### 2.0 Processing of RMSE ####
    #### 2.1 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_rot_err_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * rot_a_err[0, i]
        theta = scale * rot_a_err[1, i]
        psi = scale * rot_a_err[2, i]
        acc_rot_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

    ################################################################################################
    #### 2.2 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_rot_err_rc_m = obs_acc_.copy()
    acc_rot_err_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_rot_err_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_rot_err_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_rot_err_rc = acc_rot_err_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    #### 3.0 filter the shit out of this ####
    # 3.1 Lowpass and highpass filter
    both_maxi = [[],[]]
    for filter_type, freq, ii in zip(['lowpass','highpass'],[lpfreq,hpfreq], range(2)):
        # Angles
        obs_a_all_lp = obs_a_all.copy().filter(filter_type, freq = freq, zerophase = True)
        euler_a_all_lp = euler_a_all.copy().filter(filter_type, freq = freq, zerophase = True)
        rot_a_err_all_lp = rot_a_err_all.copy().filter(filter_type, freq = freq, zerophase = True)
        euler_a_err_all_lp = euler_a_err_all.copy().filter(filter_type, freq = freq, zerophase = True)
        # Disp [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp]
        disp_obs_lp = disp_obs.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_obs_demean_lp = disp_obs_demean.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_obs_rc_lp = disp_obs_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_euler_rc_lp = disp_euler_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_rot_err_rc_lp = disp_rot_err_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        disp_euler_err_rc_lp = disp_euler_err_rc.copy().filter(filter_type, freq = freq, zerophase = True)
        # Acc [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp]
        obs_acc_lp = obs_acc_.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_obs_demean_lp = acc_obs_demean.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_obs_rc_m_lp = acc_obs_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_euler_rc_m_lp = acc_euler_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_rot_err_rc_m_lp = acc_rot_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)
        acc_euler_err_rc_m_lp = acc_euler_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = True)

        #### 2.3 Timeseries difference ####
        time = obs_acc_[0].times()

        # rotation
        base = obs_a_all_lp
        data = [obs_a_all_lp, euler_a_all_lp, rot_a_err_all_lp, euler_a_err_all_lp]
        NN = len(data)
        ch = ['HJE','HJN','HJZ']
        TSdiff_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_rot = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_rot[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_rot[j][i] = mini
                else:
                    TSmax_rot[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_rot[j][i] = mini
                    else:
                        TSmax_rot[j][i] = maxi

                TS_rot[j][i] = channel.select(channel=ch[i])[0].data

        # acceleration
        base = obs_acc_lp
        data = [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_rot_err_rc_m_lp, acc_euler_err_rc_m_lp]
        NN = len(data)
        ch = ['HNE', 'HNN', 'HNZ']
        TSdiff_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_acc = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_acc[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_acc[j][i] = mini
                else:
                    TSmax_acc[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_acc[j][i] = mini
                    else:
                        TSmax_acc[j][i] = maxi
                TS_acc[j][i] = channel.select(channel=ch[i])[0].data

        # displacement
        base = disp_obs_lp
        data = [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp]
        ch = ['HNE', 'HNN', 'HNZ']
        NN = len(data)
        TSdiff_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TSmax_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TS_disp = [[[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data - base.select(channel=ch[i])[0].data
                TSdiff_disp[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff[:-df])
                maxi = numpy.max(diff[:-df])
                if numpy.abs(mini) > maxi:
                    TSmax_disp[j][i] = mini
                else:
                    TSmax_disp[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data)
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data)
                    if numpy.abs(mini) > maxi:
                        TSmax_disp[j][i] = mini
                    else:
                        TSmax_disp[j][i] = maxi

                TS_disp[j][i] = channel.select(channel=ch[i])[0].data



        if plot:
            #### 2.4 Plot stuff ####

            ######################## Plot Time Series
            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            plt.suptitle('TimeSeries Comparisons for an M' + str(magnitude))
            x_ticks = ['obs. ', 'obs. \ndemeaned', 'rot', 'euler', 'rot + earth rc', 'euler + earth rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]' % direction[i])
                for j in range(4):
                    ax.plot(time, TS_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
            ax.legend(loc='upper left')

            x_ticks = ['obs.', 'obs. demeaned', 'rc. rot', 'rc. euler', 'rc. rot + earth rc', 'rc. euler + earth rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left')

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left')

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            plt.savefig('%s/TS_%s_%s_%s.png' % (root_savefig, filter_type, ampscale, str(new_starttime)), dpi=300)

            ######################## Plot Difference

            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            plt.suptitle('TimeSeries Difference Comparisons for an M' + str(magnitude))
            x_ticks = ['obs.', 'obs. \ndemeaned', 'rot', 'euler', 'rot + earth rc', 'euler + earth rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]\nDifference \nto observed' % direction[i])
                for j in range(4):
                    ax.plot(time, TSdiff_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    #ax.axhline(TSmax_rot[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            x_ticks = ['obs.', 'obs. demeaned', 'rc. observed', 'rc. euler', 'rc. rot + earth rc', 'rc. euler + earth rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]\nDifference \nto observed' % direction[i])
                for j in range(6):
                    ax.plot(time, TSdiff_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #ax.axhline(TSmax_disp[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]\nDifference \nto observed' % direction[i])
                for j in range(6):
                    ax.plot(time, TSdiff_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #ax.axhline(TSmax_acc[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left')

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            plt.savefig('%s/TSdiff_%s_%s_%s.png' % (root_savefig, filter_type, ampscale, str(new_starttime)), dpi=300)
            if show:
                plt.show()

        both_maxi[ii] = [TSmax_rot,TSmax_disp,TSmax_acc]
        # TSmax_disp
        # disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp
        # all three components

        # TSmax_acc
        # obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp
        # all three components

        # TSmax_rot
        # obs_a_all_lp, euler_a_all_lp, euler_a_err_all_lp
        # all three components
    return both_maxi

# This one is exactly the same as the original version, but also outputs the timesereis and not just the min and max.
def filter_plotly_maxy_Kilauea_v2(date_name='date_name', starttime='starttime', endtime='endtime', folder='mag_int',
                               ampscale='ampscale',  magnitude='magnitude', lpfreq = 'freq', hpfreq = 'freq',
                               plot=True, show=False):
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    freq = lpfreq
    lenmean = 4*df
    root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
    root_processed = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/Processed'
    root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4Correction/%s' %folder
    obs_acc_ = read('%s/Kilauea_%s_HNE.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNN.mseed' % (root_import, date_name))
    obs_acc_ += read('%s/Kilauea_%s_HNZ.mseed' % (root_import, date_name))

    inv = read_inventory(root_import + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime, endtime)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
    for tr in obs_acc_:
        tr.data = tr.data * ampscale

    obs_acc = numpy.vstack([obs_acc_.select(channel='HNE')[0].data,
                            obs_acc_.select(channel='HNN')[0].data,
                            obs_acc_.select(channel='HNZ')[0].data])
    # demean the data
    offset_obs_acc = numpy.array(
        [numpy.mean(obs_acc[0, :lenmean]), numpy.mean(obs_acc[1, :lenmean]), numpy.mean(obs_acc[2, :lenmean])])
    obs_acc_demean = obs_acc.copy()
    obs_acc_demean[0, :] = obs_acc_demean[0, :] - offset_obs_acc[0]
    obs_acc_demean[1, :] = obs_acc_demean[1, :] - offset_obs_acc[1]
    obs_acc_demean[2, :] = obs_acc_demean[2, :] - offset_obs_acc[2]

    #### 1.4 import rotation data ####
    # import observed angles
    starttime_str = str(starttime)
    new_starttime = starttime_str.replace(':', '_')
    new_starttime = new_starttime.replace('-', '_')
    new_starttime = new_starttime.replace('.', '_')
    # 'obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot', 'euler_rr_tot'
    obs_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_obs_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    #obs_a_all.plot()
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    #obs_a_all.plot()
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import rot angle with earth rotation correction
    rot_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJN.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    rot_a_err_all = rot_a_err_E
    rot_a_err_all += rot_a_err_N
    rot_a_err_all += rot_a_err_Z
    rot_a_err_all = rot_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    rot_a_err = numpy.vstack([rot_a_err_all.select(channel='HJE')[0].data, rot_a_err_all.select(channel='HJN')[0].data,
                              rot_a_err_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJE.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_N = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJN.mseed' % ( root_processeddata, new_starttime, ampscale))
    euler_a_err_Z = read('%s/All_EQ/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJZ.mseed' % (root_processeddata, new_starttime, ampscale))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(sampling_rate=df)
    euler_a_err = numpy.vstack([euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])

    #### 2.0 Processing of RMSE ####
    #### 2.1 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_rot_err_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * rot_a_err[0, i]
        theta = scale * rot_a_err[1, i]
        psi = scale * rot_a_err[2, i]
        acc_rot_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

    ################################################################################################
    #### 2.2 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HNE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HNN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HNZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HNE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HNN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HNZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_rot_err_rc_m = obs_acc_.copy()
    acc_rot_err_rc_m.select(channel='HNE')[0].data = acc_euler_rc[0, :]
    acc_rot_err_rc_m.select(channel='HNN')[0].data = acc_euler_rc[1, :]
    acc_rot_err_rc_m.select(channel='HNZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HNE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HNN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HNZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_rot_err_rc = acc_rot_err_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    #### 3.0 filter the shit out of this ####
    # 3.1 Lowpass and highpass filter
    both_maxi = [[],[]]
    both_ts = [[],[]]
    for filter_type, freq, ii in zip(['lowpass','highpass'],[lpfreq,hpfreq], range(2)):
        # Angles
        obs_a_all_lp = obs_a_all.copy().filter(filter_type, freq = freq, zerophase = False)
        euler_a_all_lp = euler_a_all.copy().filter(filter_type, freq = freq, zerophase = False)
        rot_a_err_all_lp = rot_a_err_all.copy().filter(filter_type, freq = freq, zerophase = False)
        euler_a_err_all_lp = euler_a_err_all.copy().filter(filter_type, freq = freq, zerophase = False)

        # Disp [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp]
        disp_obs_lp = disp_obs.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_obs_demean_lp = disp_obs_demean.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_obs_rc_lp = disp_obs_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_euler_rc_lp = disp_euler_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_rot_err_rc_lp = disp_rot_err_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_euler_err_rc_lp = disp_euler_err_rc.copy().filter(filter_type, freq = freq, zerophase = False)

        # Acc [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp]
        obs_acc_lp = obs_acc_.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_obs_demean_lp = acc_obs_demean.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_obs_rc_m_lp = acc_obs_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_euler_rc_m_lp = acc_euler_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_rot_err_rc_m_lp = acc_rot_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_euler_err_rc_m_lp = acc_euler_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)

        #### 2.3 Timeseries difference ####

        # rotation
        base = obs_a_all_lp
        data = [obs_a_all_lp, euler_a_all_lp, rot_a_err_all_lp, euler_a_err_all_lp]
        NN = len(data)
        start, end = int(df), int((2*df))
        time = obs_acc_[0].times()[start:-end]
        ch = ['HJE','HJN','HJZ']
        TSdiff_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_rot = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_rot[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_rot[j][i] = mini
                else:
                    TSmax_rot[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_rot[j][i] = mini
                    else:
                        TSmax_rot[j][i] = maxi

                TS_rot[j][i] = channel.select(channel=ch[i])[0].data[start:-end]

        # acceleration
        base = acc_obs_demean_lp
        data = [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_rot_err_rc_m_lp, acc_euler_err_rc_m_lp]
        NN = len(data)
        ch = ['HNE', 'HNN', 'HNZ']
        TSdiff_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_acc = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_acc[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_acc[j][i] = mini
                else:
                    TSmax_acc[j][i] = maxi

                # have the first two be the amplitude of motion and not the difference which is just 0.
                if j in [0,1]:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_acc[j][i] = mini
                    else:
                        TSmax_acc[j][i] = maxi
                TS_acc[j][i] = channel.select(channel=ch[i])[0].data[start:-end]

        # displacement
        base = disp_obs_demean_lp
        data = [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp]
        ch = ['HNE', 'HNN', 'HNZ']
        NN = len(data)
        TSdiff_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TSmax_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TS_disp = [[[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_disp[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_disp[j][i] = mini
                else:
                    TSmax_disp[j][i] = maxi

                # have the first two be the amplitude of motion and not the difference which is just 0.
                if j in [0,1]:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_disp[j][i] = mini
                    else:
                        TSmax_disp[j][i] = maxi

                TS_disp[j][i] = channel.select(channel=ch[i])[0].data[start:-end]



        if plot:
            #### 2.4 Plot stuff ####
            fontsize = 8

            ######################## Plot Time Series
            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            if magnitude < 5:
                plt.suptitle('Timeseries Comparisons for an Ml ' + str(magnitude))
            elif magnitude > 5:
                plt.suptitle('Timeseries Comparisons for an Mw ' + str(magnitude))
            x_ticks = ['obs. ', 'obs. \ndemeaned', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]' % direction[i])
                for j in range(4):
                    ax.plot(time, TS_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
            ax.legend(loc='upper left', fontsize=fontsize)

            x_ticks = ['obs.', 'obs. demeaned', 'rc. rot', 'rc. euler', 'rc. rot + spin rc', 'rc. euler + spin rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #top = numpy.max([TS_disp[1][i],TS_disp[2][i],TS_disp[3][i],TS_disp[4][i],TS_disp[5][i]])*1.2
                    #bottom = numpy.min([TS_disp[1][i],TS_disp[2][i],TS_disp[3][i],TS_disp[4][i],TS_disp[5][i]])*1.2
                    #ax.set_ylim(top= top, bottom = bottom)
            ax.legend(loc='upper left', fontsize=fontsize)

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            ax.set_xlim(left=0, right=44)
            plt.savefig('%s/TS_%s_%s_%s.png' % (root_savefig, filter_type, ampscale, str(new_starttime)), dpi=300)

            ######################## Plot Difference

            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            if magnitude < 5:
                plt.suptitle('TimeSeries Difference Comparisons for an Ml ' + str(magnitude))
            elif magnitude > 5:
                plt.suptitle('TimeSeries Difference Comparisons for an Mw ' + str(magnitude))
            x_ticks = ['obs.', 'obs. \ndemeaned', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel(r'%s [$\frac{rad}{rad}\%%$]' % direction[i])
                for j in range(4):
                    #ax.plot(time, TSdiff_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    ax.plot(time, TSdiff_rot[j][i][:]/max(abs(numpy.asarray(TS_rot[0][i][:])))*100, linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    #ax.axhline(TSmax_rot[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            x_ticks = ['obs.', 'obs. demeaned', 'rc. observed', 'rc. euler', 'rc. rot + spin rc', 'rc. euler + spin rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                #ax.set_ylabel(rf'{direction[i]} [$\\frac{{m}}{{m}}\\%$]')
                ax.set_ylabel(r'%s [$\frac{m}{m}\%%$]' % direction[i])
                for j in range(6):
                    #ax.plot(time, TSdiff_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.plot(time, TSdiff_disp[j][i][:]/max(abs(numpy.asarray(TS_disp[0][i][:])))*100, linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #ax.axhline(TSmax_disp[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                #ax.set_ylabel('%s [m/s/s]' % direction[i])
                #ax.set_ylabel(rf'{direction[i]} [$\\frac{{m/s/s}}{{m/s/s}}\\%$]')
                ax.set_ylabel(r'%s [$\frac{m/s/s}{m/s/s}\%%$]' % direction[i])
                for j in range(6):
                    #ax.plot(time, TSdiff_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.plot(time, TSdiff_acc[j][i][:]/max(abs(numpy.asarray(TS_acc[0][i][:])))*100, linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #ax.axhline(TSmax_acc[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            ax.set_xlim(left=0, right=44)
            plt.savefig('%s/TSdiff_%s_%s_%s.png' % (root_savefig, filter_type, ampscale, str(new_starttime)), dpi=300)
            if show:
                plt.show()

        both_maxi[ii] = [TSmax_rot,TSmax_disp,TSmax_acc]
        both_ts[ii] = [TS_rot, TS_disp, TS_acc]
        # TSmax_disp
        # disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp
        # all three components

        # TSmax_acc
        # obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp
        # all three components

        # TSmax_rot
        # obs_a_all_lp, euler_a_all_lp, euler_a_err_all_lp
        # all three components
    return both_maxi, both_ts
# Output as: lowpass, highpass. then disp, rot and acc, then each rotation correction type, then all three components.

# make function that work on Hualien EQ data

# NA01
def makeAnglesHualien_lat_v3(station_name='station_name', starttime='starttime', endtime='endtime', ampscale = 1, latitude=1, fromdata = True, plot=True, savedate=False, folder = 'putsomethinghere'):
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
    root_save = '%s/%s' %(Hroot_processeddata,folder)
    obs_rate  = read('%s/TW.%s.01.HJE.D.2024.093.MSEED' % (Hroot_originaldata,station_name))
    obs_rate += read('%s/TW.%s.01.HJN.D.2024.093.MSEED' % (Hroot_originaldata,station_name))
    obs_rate += read('%s/TW.%s.01.HJZ.D.2024.093.MSEED' % (Hroot_originaldata,station_name))
    df = 50
    #inv = read_inventory(Hroot_originaldata+'/station.xml') doesn't exist
    #obs_rate = obs_rate.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)

    #### 1.2 ####
    # slice data to only include EQ
    obs_rate = obs_rate.slice(starttime-5, endtime+5)

    #obs_rate.plot()

    #### 1.3 ####
    # correct stuff:
    # scale data from nrad/s to rad/s
    #obs_rate.remove_sensitivity(inventory=inv)
    for tr in obs_rate:
        tr.data = tr.data/1e9

    #obs_rate.plot()

    # orient the sensor correctly. it is oriented 1.8° wrong
    #obs_rate.rotate(method='->ZNE', inventory=inv, components=["ENZ"])


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
        if fromdata:
            for_ac_rr[:, i] = for_ac_rr[:, i] + e_rr_fromdata # use earthspin taken from rotational sensor
        else:
            for_ac_rr[:,i] = for_ac_rr[:,i] + earth_rr # use estimated earthspin from latitude calculation

    # run function to get a variety of angles and rotation rate, using a variety of rotation correction schemes
    if fromdata:
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
                mseed.select(channel=ch).write(root_save + '/Hualien_' + str(new_starttime) + '_'+str(ampscale)+
                                               '_station' + str(station_name) + '_' + name + '_' + ch + '.mseed')

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
    return

def correctAccelerationHualien_v2(station_name='station_name', starttime='starttime', endtime='endtime',
                                  ampscale=1, response ='response', plot=True, savedate=False, folder = 'putsomethinghere'):
    ##############################################################################
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    lenmean = 4 * df
    root_save = Hroot_processeddata
    obs_acc_ = read('%s/TW.%s..HLE.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLN.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLZ.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    #inv = read_inventory(Hroot_originaldata + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime - 5, endtime + 5) # starttime - 5, endtime + 5

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    #obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    #obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    for tr in obs_acc_:
        tr.data = tr.data*ampscale / response

    obs_acc_ = obs_acc_.slice(starttime, endtime)
    #obs_acc_.plot()

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

    #### 1.4 import rotation data ####
    # import observed angles
    starttime_str = str(starttime)
    new_starttime = starttime_str.replace(':', '_')
    new_starttime = new_starttime.replace('-', '_')
    new_starttime = new_starttime.replace('.', '_')
    # 'obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot', 'euler_rr_tot'
    obs_a_E = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_N = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    obs_a_all = obs_a_all.slice(starttime, endtime)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_N = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    euler_a_all = euler_a_all.slice(starttime, endtime)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import rot angle with earth rotation correction
    rot_a_err_E = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_N = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_all = rot_a_err_E
    rot_a_err_all += rot_a_err_N
    rot_a_err_all += rot_a_err_Z
    rot_a_err_all = rot_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    rot_a_err_all = rot_a_err_all.slice(starttime, endtime)
    rot_a_err = numpy.vstack([rot_a_err_all.select(channel='HJE')[0].data, rot_a_err_all.select(channel='HJN')[0].data,
         rot_a_err_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_err_N = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_err_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).resample(
        sampling_rate=df)
    euler_a_err_all = euler_a_err_all.slice(starttime, endtime)
    euler_a_err = numpy.vstack(
        [euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])
    if plot:
        fig, axs = plt.subplots(3, 1)

        for ax, ch in zip(axs.flat, ['HJE', 'HJN', 'HJZ']):
            for data, liner, color, label in zip([obs_a_all, euler_a_all, rot_a_err_all, euler_a_err_all],
                                                 ['-', '--', '-.', 'dotted'],
                                                 ['k', 'grey', 'lightgrey', 'lightsteelblue'],
                                                 ['obs_a_all', 'euler_a_all', 'rot_a_err_all' ,'euler_a_tot_all']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Angle [rad]')
        ax.legend()
        ax.set_xlabel('Time')
        fig.savefig('%s/TS_rot_rc_%s.png' % (root_savefig, station_name), dpi=300, bbox_inches='tight')

    ################################################################################################
    #### 2.0 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_rot_err_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in tqdm(range(NN)):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * rot_a_err[0, i]
        theta = scale * rot_a_err[1, i]
        psi = scale * rot_a_err[2, i]
        acc_rot_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

    ################################################################################################
    #### 3.0 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HLE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HLN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HLZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HLE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HLN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HLZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HLE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HLN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HLZ')[0].data = acc_euler_rc[2, :]

    acc_rot_err_rc_m = obs_acc_.copy()
    acc_rot_err_rc_m.select(channel='HLE')[0].data = acc_euler_rc[0, :]
    acc_rot_err_rc_m.select(channel='HLN')[0].data = acc_euler_rc[1, :]
    acc_rot_err_rc_m.select(channel='HLZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HLE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HLN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HLZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_rot_err_rc = acc_rot_err_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    if plot:
        # in acceleration
        fig, axs = plt.subplots(3, 1)
        for ax, ch, i in zip(axs.flat, ['HLE', 'HLN', 'HLZ'], range(3)):
            for data, liner, color, label in zip(
                    [obs_acc_, acc_obs_demean, acc_obs_rc_m, acc_euler_rc_m, acc_rot_err_rc_m, acc_euler_err_rc_m],
                    ['-', 'dotted', '--', '-.', '--', 'dotted'],
                    ['k', 'k', 'navy', 'blue', 'lightblue', 'lightsteelblue'],
                    ['acc_obs', 'acc_dm', 'acc_dm_rc_rot', 'acc_dm_rc_euler', 'acc_dm_rc_rot_err', 'acc_dm_rc_euler_err']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Acc. [m/s/s]')
        ax.legend()
        ax.set_xlabel('Time')
        fig.savefig('%s/TS_acc_rc_%s.png' %(root_savefig, station_name), dpi=300, bbox_inches='tight')

        # in displacement
        fig, axs = plt.subplots(3, 1)
        for ax, ch in zip(axs.flat, ['HLE', 'HLN', 'HLZ']):
            for data, liner, color, label in zip([disp_obs, disp_obs_demean, disp_obs_rc, disp_euler_rc, disp_rot_err_rc, disp_euler_err_rc],
                                                 ['-', 'dotted', '--', '-.', '--', 'dotted'],
                                                 ['k', 'k', 'navy', 'blue', 'lightblue', 'lightsteelblue'],
                                                 ['disp_obs', 'disp_dm', 'disp_dm_rc_rot', 'disp_dm_rc_euler', 'disp_dm_rc_rot_err', 'disp_dm_rc_euler_err']):
                ax.plot(data.select(channel=ch)[0].times('matplotlib'), data.select(channel=ch)[0].data, linestyle=liner,
                        color=color, label=label)
                ax.set_ylabel('Disp. [m]')
        ax.legend()
        ax.set_xlabel('Time')
        fig.savefig('%s/TS_disp_rc_%s.png' %(root_savefig, station_name), dpi=300, bbox_inches='tight')

        plt.show()

def filter_plotly_maxy_Hualien_v2(station_name='station_name', starttime='starttime', endtime='endtime',
                               ampscale='ampscale', response='responese', magnitude='magnitude', lpfreq = 'freq', hpfreq = 'freq',
                               plot=True, show=False):
    #### 1.0 Import data and perform some preprocessing
    #### 1.1 ####
    # load data
    df = 50
    lenmean = 4 * df
    root_save = Hroot_processeddata
    obs_acc_ = read('%s/TW.%s..HLE.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLN.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    obs_acc_ += read('%s/TW.%s..HLZ.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
    # inv = read_inventory(Hroot_originaldata + '/station.xml')

    #### 1.2 ####
    # slice data to only include EQ
    obs_acc_ = obs_acc_.slice(starttime-5, endtime+5)

    #### 1.3 ####
    # correct stuff:
    # remove response from accelerations
    # obs_acc_ = obs_acc_.remove_response(inventory=inv, output='ACC')
    # obs_acc_ = obs_acc_.rotate(method='->ZNE', inventory=inv, components=["ZNE"])
    obs_acc_ = obs_acc_.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    for tr in obs_acc_:
        tr.data = tr.data * ampscale / response

    obs_acc_ = obs_acc_.slice(starttime, endtime)
    #obs_acc_.plot()

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

    #### 1.4 import rotation data ####
    # import observed angles
    starttime_str = str(starttime-5)
    new_starttime = starttime_str.replace(':', '_')
    new_starttime = new_starttime.replace('-', '_')
    new_starttime = new_starttime.replace('.', '_')
    # 'obs_angle', 'obs_rr', 'euler_angle', 'rot_angle_err', 'euler_rr', 'rot_rr_err', 'euler_angle_tot', 'euler_rr_tot'
    obs_a_E = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_N = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_obs_angle_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    obs_a_all = obs_a_E
    obs_a_all += obs_a_N
    obs_a_all += obs_a_Z
    #obs_a_all.plot()
    obs_a_all = obs_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    obs_a_all = obs_a_all.slice(starttime, endtime)
    obs_a = numpy.vstack([obs_a_all.select(channel='HJE')[0].data, obs_a_all.select(channel='HJN')[0].data,
                          obs_a_all.select(channel='HJZ')[0].data])

    # import euler angle no earth rotation correction
    euler_a_E = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_N = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_all = euler_a_E
    euler_a_all += euler_a_N
    euler_a_all += euler_a_Z
    euler_a_all = euler_a_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    euler_a_all = euler_a_all.slice(starttime, endtime)
    euler_a = numpy.vstack([euler_a_all.select(channel='HJE')[0].data, euler_a_all.select(channel='HJN')[0].data,
                            euler_a_all.select(channel='HJZ')[0].data])

    # import rot angle with earth rotation correction
    rot_a_err_E = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_N = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJN.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_rot_angle_err_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    rot_a_err_all = rot_a_err_E
    rot_a_err_all += rot_a_err_N
    rot_a_err_all += rot_a_err_Z
    rot_a_err_all = rot_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    rot_a_err_all = rot_a_err_all.slice(starttime, endtime)
    rot_a_err = numpy.vstack([rot_a_err_all.select(channel='HJE')[0].data, rot_a_err_all.select(channel='HJN')[0].data,
                              rot_a_err_all.select(channel='HJZ')[0].data])

    # import euler angle with earth rotation correction
    euler_a_err_E = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJE.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_err_N = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJN.mseed' % ( root_save, new_starttime, ampscale, station_name))
    euler_a_err_Z = read('%s/All_EQ/Hualien_%s_%s_station%s_euler_angle_tot_HJZ.mseed' % (root_save, new_starttime, ampscale, station_name))
    euler_a_err_all = euler_a_err_E
    euler_a_err_all += euler_a_err_N
    euler_a_err_all += euler_a_err_Z
    euler_a_err_all = euler_a_err_all.filter('lowpass', freq=df / 2, corners=8, zerophase=True).interpolate(sampling_rate=df, starttime=starttime)
    euler_a_err_all = euler_a_err_all.slice(starttime, endtime)
    euler_a_err = numpy.vstack([euler_a_err_all.select(channel='HJE')[0].data, euler_a_err_all.select(channel='HJN')[0].data,
         euler_a_err_all.select(channel='HJZ')[0].data])

    #### 2.0 Processing of RMSE ####
    #### 2.1 Rotation correction ####
    gravi = numpy.array([0, 0, 9.81])
    NN = len(obs_acc_demean[0, :])
    acc_obs_rc = numpy.zeros((3, NN))
    acc_euler_rc = numpy.zeros((3, NN))
    acc_rot_err_rc = numpy.zeros((3, NN))
    acc_euler_err_rc = numpy.zeros((3, NN))
    scale = -1
    for i in range(NN):
        data = obs_acc_demean[:, i]

        phi = scale * obs_a[0, i]
        theta = scale * obs_a[1, i]
        psi = scale * obs_a[2, i]
        acc_obs_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a[0, i]
        theta = scale * euler_a[1, i]
        psi = scale * euler_a[2, i]
        acc_euler_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * rot_a_err[0, i]
        theta = scale * rot_a_err[1, i]
        psi = scale * rot_a_err[2, i]
        acc_rot_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

        phi = scale * euler_a_err[0, i]
        theta = scale * euler_a_err[1, i]
        psi = scale * euler_a_err[2, i]
        acc_euler_err_rc[:, i] = rot_vec(phi, theta, psi, data + gravi) - gravi

    ################################################################################################
    #### 2.2 Integration to displacement ####
    acc_obs_demean = obs_acc_.copy()
    acc_obs_demean.select(channel='HLE')[0].data = obs_acc_demean[0, :]
    acc_obs_demean.select(channel='HLN')[0].data = obs_acc_demean[1, :]
    acc_obs_demean.select(channel='HLZ')[0].data = obs_acc_demean[2, :]

    acc_obs_rc_m = obs_acc_.copy()
    acc_obs_rc_m.select(channel='HLE')[0].data = acc_obs_rc[0, :]
    acc_obs_rc_m.select(channel='HLN')[0].data = acc_obs_rc[1, :]
    acc_obs_rc_m.select(channel='HLZ')[0].data = acc_obs_rc[2, :]

    acc_euler_rc_m = obs_acc_.copy()
    acc_euler_rc_m.select(channel='HLE')[0].data = acc_euler_rc[0, :]
    acc_euler_rc_m.select(channel='HLN')[0].data = acc_euler_rc[1, :]
    acc_euler_rc_m.select(channel='HLZ')[0].data = acc_euler_rc[2, :]

    acc_rot_err_rc_m = obs_acc_.copy()
    acc_rot_err_rc_m.select(channel='HLE')[0].data = acc_euler_rc[0, :]
    acc_rot_err_rc_m.select(channel='HLN')[0].data = acc_euler_rc[1, :]
    acc_rot_err_rc_m.select(channel='HLZ')[0].data = acc_euler_rc[2, :]

    acc_euler_err_rc_m = obs_acc_.copy()
    acc_euler_err_rc_m.select(channel='HLE')[0].data = acc_euler_err_rc[0, :]
    acc_euler_err_rc_m.select(channel='HLN')[0].data = acc_euler_err_rc[1, :]
    acc_euler_err_rc_m.select(channel='HLZ')[0].data = acc_euler_err_rc[2, :]

    NN = len(obs_acc[0, :])
    disp_obs_demean = acc_obs_demean.copy().integrate().integrate()
    disp_obs = obs_acc_.copy().integrate().integrate()
    disp_obs_rc = acc_obs_rc_m.copy().integrate().integrate()
    disp_euler_rc = acc_euler_rc_m.copy().integrate().integrate()
    disp_rot_err_rc = acc_rot_err_rc_m.copy().integrate().integrate()
    disp_euler_err_rc = acc_euler_err_rc_m.copy().integrate().integrate()

    #### 3.0 filter the shit out of this ####
    # 3.1 Lowpass and highpass filter
    both_maxi = [[],[]]
    both_ts = [[],[]]
    for filter_type, freq, ii in zip(['lowpass','highpass'],[lpfreq,hpfreq], range(2)):
        # Angles
        obs_a_all_lp = obs_a_all.copy().filter(filter_type, freq = freq, zerophase = False)
        euler_a_all_lp = euler_a_all.copy().filter(filter_type, freq = freq, zerophase = False)
        rot_a_err_all_lp = rot_a_err_all.copy().filter(filter_type, freq = freq, zerophase = False)
        euler_a_err_all_lp = euler_a_err_all.copy().filter(filter_type, freq = freq, zerophase = False)

        # Disp [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp]
        disp_obs_lp = disp_obs.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_obs_demean_lp = disp_obs_demean.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_obs_rc_lp = disp_obs_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_euler_rc_lp = disp_euler_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_rot_err_rc_lp = disp_rot_err_rc.copy().filter(filter_type, freq = freq, zerophase = False)
        disp_euler_err_rc_lp = disp_euler_err_rc.copy().filter(filter_type, freq = freq, zerophase = False)

        # Acc [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp]
        obs_acc_lp = obs_acc_.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_obs_demean_lp = acc_obs_demean.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_obs_rc_m_lp = acc_obs_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_euler_rc_m_lp = acc_euler_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_rot_err_rc_m_lp = acc_rot_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)
        acc_euler_err_rc_m_lp = acc_euler_err_rc_m.copy().filter(filter_type, freq = freq, zerophase = False)

        #### 2.3 Timeseries difference ####

        # rotation
        base = obs_a_all_lp
        data = [obs_a_all_lp, euler_a_all_lp, rot_a_err_all_lp, euler_a_err_all_lp]
        NN = len(data)
        start, end = int(df), int((2*df))
        time = obs_acc_[0].times()[start:-end]
        ch = ['HJE','HJN','HJZ']
        TSdiff_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_rot = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_rot = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_rot[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_rot[j][i] = mini
                else:
                    TSmax_rot[j][i] = maxi

                # have the first one be the amplitude of motion and not the difference which is just 0.
                if j == 0:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_rot[j][i] = mini
                    else:
                        TSmax_rot[j][i] = maxi

                TS_rot[j][i] = channel.select(channel=ch[i])[0].data[start:-end]

        # acceleration
        base = acc_obs_demean_lp
        data = [obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_rot_err_rc_m_lp, acc_euler_err_rc_m_lp]
        NN = len(data)
        ch = ['HLE', 'HLN', 'HLZ']
        TSdiff_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TSmax_acc = [[[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []],
                      [[], [], []]]
        TS_acc = [[[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []],
                  [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_acc[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_acc[j][i] = mini
                else:
                    TSmax_acc[j][i] = maxi

                # have the first two be the amplitude of motion and not the difference which is just 0.
                if j in [0,1]:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_acc[j][i] = mini
                    else:
                        TSmax_acc[j][i] = maxi
                TS_acc[j][i] = channel.select(channel=ch[i])[0].data[start:-end]

        # displacement
        base = disp_obs_demean_lp
        data = [disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp]
        ch = ['HLE', 'HLN', 'HLZ']
        NN = len(data)
        TSdiff_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TSmax_disp = [[[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []],
                       [[], [], []]]
        TS_disp = [[[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []],
                   [[], [], []]]
        for j in range(NN):
            channel = data[j]
            for i in range(3):
                diff = channel.select(channel=ch[i])[0].data[start:-end] - base.select(channel=ch[i])[0].data[start:-end]
                TSdiff_disp[j][i] = diff

                # find extreme value:
                mini = numpy.min(diff)
                maxi = numpy.max(diff)
                if numpy.abs(mini) > maxi:
                    TSmax_disp[j][i] = mini
                else:
                    TSmax_disp[j][i] = maxi

                # have the first two be the amplitude of motion and not the difference which is just 0.
                if j in [0,1]:
                    mini = numpy.min(channel.select(channel=ch[i])[0].data[start:-end])
                    maxi = numpy.max(channel.select(channel=ch[i])[0].data[start:-end])
                    if numpy.abs(mini) > maxi:
                        TSmax_disp[j][i] = mini
                    else:
                        TSmax_disp[j][i] = maxi

                TS_disp[j][i] = channel.select(channel=ch[i])[0].data[start:-end]



        if plot:
            #### 2.4 Plot stuff ####
            fontsize = 8

            ######################## Plot Time Series
            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            if magnitude < 5:
                plt.suptitle('Timeseries Comparisons for an Ml ' + str(magnitude))
            elif magnitude > 5:
                plt.suptitle('Timeseries Comparisons for an Mw ' + str(magnitude))
            x_ticks = ['obs. ', 'obs. \ndemeaned', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel('%s [rad]' % direction[i])
                for j in range(4):
                    ax.plot(time, TS_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
            ax.legend(loc='upper left', fontsize=fontsize)

            x_ticks = ['obs.', 'obs. demeaned', 'rc. rot', 'rc. euler', 'rc. rot + spin rc', 'rc. euler + spin rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                ax.set_ylabel('%s [m]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    top = numpy.max([TS_disp[1][i],TS_disp[2][i],TS_disp[3][i],TS_disp[4][i],TS_disp[5][i]])*1.2
                    bottom = numpy.min([TS_disp[1][i],TS_disp[2][i],TS_disp[3][i],TS_disp[4][i],TS_disp[5][i]])*1.2
                    if top == 0:
                        top = bottom*0.1
                    elif bottom == 0:
                        bottom = top*0.1
                    ax.set_ylim(top= top, bottom = bottom)
            ax.legend(loc='upper left', fontsize=fontsize)

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                ax.set_ylabel('%s [m/s/s]' % direction[i])
                for j in range(6):
                    ax.plot(time, TS_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            ax.set_xlim(left=0, right=104)
            plt.savefig('%s/TS_%s_%s_%s.png' % (Hroot_savefig, filter_type, ampscale, station_name), dpi=300)

            ######################## Plot Difference

            fig, axs = plt.subplots(9, 1, figsize=(11, 12), sharex=True)
            plt.subplots_adjust(hspace=0, top=0.95, bottom=0.07)
            if magnitude < 5:
                plt.suptitle('Timeseries Difference Comparisons for an Ml ' + str(magnitude))
            elif magnitude > 5:
                plt.suptitle('Timeseries Difference Comparisons for an Mw ' + str(magnitude))
            x_ticks = ['obs.', 'obs. \ndemeaned', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']

            # rotation
            color = ['darkred', 'red', 'tomato', 'lightcoral']
            liner = ['--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            for i in range(3):
                ax = axs[0 + i]
                ax.set_ylabel(r'%s [$\frac{rad}{rad}\%%$]' % direction[i])
                for j in range(4):
                    #ax.plot(time, TSdiff_rot[j][i], linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    ax.plot(time, TSdiff_rot[j][i][:]/TSmax_rot[0][i]*100, linestyle=liner[j], color=color[j], label=x_ticks[2 + j])
                    #ax.axhline(TSmax_rot[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            x_ticks = ['obs.', 'obs. demeaned', 'rc. observed', 'rc. euler', 'rc. rot + spin rc', 'rc. euler + spin rc']
            liner = ['-', (0, (3, 1, 1, 1, 1, 1)), '--', '-.', 'dotted', '-']
            direction = ['East', 'North', 'Up']
            # displacement
            color = ['k', 'k', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
            for i in range(3):
                ax = axs[3 + i]
                #ax.set_ylabel(rf'{direction[i]} [$\\frac{{m}}{{m}}\\%$]')
                ax.set_ylabel(r'%s [$\frac{m}{m}\%%$]' % direction[i])
                for j in range(6):
                    #ax.plot(time, TSdiff_disp[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.plot(time, TSdiff_disp[j][i][:]/TSmax_disp[0][i]*100, linestyle=liner[j], color=color[j], label=x_ticks[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            # acceleration
            color = ['midnightblue', 'midnightblue', 'blue', 'cornflowerblue', 'deepskyblue', 'lightblue']
            for i in range(3):
                ax = axs[6 + i]
                #ax.set_ylabel('%s [m/s/s]' % direction[i])
                #ax.set_ylabel(rf'{direction[i]} [$\\frac{{m/s/s}}{{m/s/s}}\\%$]')
                ax.set_ylabel(r'%s [$\frac{m/s/s}{m/s/s}\%%$]' % direction[i])
                for j in range(6):
                    #ax.plot(time, TSdiff_acc[j][i], linestyle=liner[j], color=color[j], label=x_ticks[j])
                    ax.plot(time, TSdiff_acc[j][i][:]/TSmax_acc[0][i]*100, linestyle=liner[j], color=color[j], label=x_ticks[j])
                    #ax.axhline(TSmax_acc[j][i], linestyle=(0, (5, 10)), color=color[j])
            ax.legend(loc='upper left', fontsize=fontsize)

            ax.set_xlabel('Time since %s [s]' % str(new_starttime))
            ax.set_xlim(left=0, right=104)
            plt.savefig('%s/TSdiff_%s_%s_%s.png' % (Hroot_savefig, filter_type, ampscale, station_name), dpi=300)
            if show:
                plt.show()

        both_maxi[ii] = [TSmax_rot,TSmax_disp,TSmax_acc]
        both_ts[ii] = [TS_rot, TS_disp, TS_acc]
        # TSmax_disp
        # disp_obs_lp, disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_euler_err_rc_lp
        # all three components

        # TSmax_acc
        # obs_acc_lp, acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_euler_err_rc_m_lp
        # all three components

        # TSmax_rot
        # obs_a_all_lp, euler_a_all_lp, euler_a_err_all_lp
        # all three components
    return both_maxi, both_ts