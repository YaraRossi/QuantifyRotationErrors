import numpy
import obspy
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy import read, read_inventory, UTCDateTime
import matplotlib.pyplot as plt
from numpy import mean
from attitudeequation import earth_rotationrate, attitude_equation_simple
from functions import makeAnglesKilauea_lat_v3, correctAccelerationKilauea_v2,filter_plotly_maxy_Kilauea,filter_plotly_maxy_Kilauea_v2
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()


def eq_kilauea(min_mag=3):
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

############################
#### Start Calculations ####

## 2. Other earthquakes
# get time of various EQ's:
minmag = 4
ml318 = True

info_eq = eq_kilauea(min_mag=minmag)
info_eq3 = eq_kilauea(min_mag=3.17)
ampscale=1

# get start and end times of Earthquakes:
date, arrival, starttime, endtime, magnitude, distance = [], [], [], [], [], []

for date_time,arrival_time, mags, dist in zip(info_eq['time'],info_eq['arrivaltime'],info_eq['mag'],info_eq['dist']):

    date_correct = date_time[0:4]+date_time[5:7]+date_time[8:10]
    date.append(date_correct)
    arrival.append(arrival_time)
    if mags < 4:
        starttime.append(UTCDateTime(date_time)+arrival_time-15)
        endtime.append(UTCDateTime(date_time)+arrival_time+30)
    else:
        starttime.append(UTCDateTime(date_time)+arrival_time-15)
        endtime.append(UTCDateTime(date_time)+arrival_time+30)
    magnitude.append(mags)
    distance.append(dist)

if ml318:
    iindex = info_eq3.index[info_eq3['time'].astype(str).str.contains('2018-07-12T05:12')].tolist()
    info_eq_ = info_eq3.loc[iindex[0]]
    date_time, arrival_time, mags, dist = info_eq_['time'], info_eq_['arrivaltime'], info_eq_['mag'], info_eq_['dist']
    date_correct = date_time[0:4]+date_time[5:7]+date_time[8:10]
    date.append(date_correct)
    arrival.append(arrival_time)
    if mags < 4:
        starttime.append(UTCDateTime(date_time)+arrival_time-15)
        endtime.append(UTCDateTime(date_time)+arrival_time+30)
    else:
        starttime.append(UTCDateTime(date_time)+arrival_time-15)
        endtime.append(UTCDateTime(date_time)+arrival_time+30)
    magnitude.append(mags)
    distance.append(dist)

'''
for NN in range(len(info_eq['time'])):
    try:
        date_ = date[NN]
        date_name = date_[0:4]+date_[5:7]+date_[8:10]
    except Exception as e:
        print(starttime[NN], e)'''

# here I calculate a couple of eq that are larger and I just want original amplitude scale and original latitude.
# Ml 3.18m, Mw5.3, Mw5.3, Mw5.3, Ml4.36

available_EQ = []
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
for date_name, starttime, endtime, magnitude, distance in zip(date,starttime, endtime, magnitude, distance):
    '''try:
        makeAnglesKilauea_lat_v3(date_name,starttime,endtime,latitude=19.420908, ampscale=1,
                                 plot=False, savedate=True, folder='All_EQ')
        if magnitude > 4:
            for ampscale in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                makeAnglesKilauea_lat_v3(date_name, starttime, endtime, latitude=19.420908, ampscale=ampscale,
                                         plot=False, savedate=True, folder='Scaling')

            for latitude in [0, 15, 20, 30, 45, 60, 75, 90]:
                makeAnglesKilauea_lat_v3(date_name, starttime, endtime, latitude=latitude, ampscale=1, plot=False,
                                         savedate=True, folder='Latitudes')
        elif '2018-07-12T05:12' in str(starttime):

            for ampscale in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                makeAnglesKilauea_lat_v3(date_name, starttime, endtime, latitude=19.420908, ampscale=ampscale,
                                         plot=False, savedate=True, folder='Scaling')

            for latitude in [0, 15, 20, 30, 45, 60, 75, 90]:
                makeAnglesKilauea_lat_v3(date_name, starttime, endtime, latitude=latitude, ampscale=1, plot=False,
                                         savedate=True, folder='Latitudes')

    except:
        print('no data for times: ' + date_name)
        continue'''

    print('Now perform the corrections on the accelerations')
    try:
        #correctAccelerationKilauea_v2(date_name, starttime, endtime, ampscale=1,
        #                              plot=True, savedate=False, folder='All_EQ')
        mag_int = int(magnitude)
        #both_maxi = filter_plotly_maxy_Kilauea(date_name, starttime, endtime, folder=mag_int, ampscale=1, magnitude=magnitude,
        #                           lpfreq=0.1, hpfreq=0.1, plot=False, show=False)
        both_maxi, both_ts = filter_plotly_maxy_Kilauea_v2(date_name, starttime, endtime, folder=mag_int, ampscale=1, magnitude=magnitude,
                                   lpfreq=0.1, hpfreq=0.1, plot=True, show=False)

        # now save all the information from the EQ that have both data, and have been made into figures:
        #
        available_EQ.append([date_name, starttime, endtime, magnitude, distance])
        max_rot_lp.append([both_maxi[0][0][0],both_maxi[0][0][1],both_maxi[0][0][2],both_maxi[0][0][3]]) # lowpass
        max_rot_hp.append([both_maxi[1][0][0],both_maxi[1][0][1],both_maxi[1][0][2],both_maxi[1][0][3]]) # highpass

        max_disp_lp.append([both_maxi[0][1][0],both_maxi[0][1][1],both_maxi[0][1][2],both_maxi[0][1][3],both_maxi[0][1][4],both_maxi[0][1][5]]) # lowpass
        max_disp_hp.append([both_maxi[1][1][0],both_maxi[1][1][1],both_maxi[1][1][2],both_maxi[1][1][3],both_maxi[1][1][4],both_maxi[1][1][5]]) # highpass

        max_acc_lp.append([both_maxi[0][2][0],both_maxi[0][2][1],both_maxi[0][2][2],both_maxi[0][2][3],both_maxi[0][2][4],both_maxi[0][2][5]]) # lowpass
        max_acc_hp.append([both_maxi[1][2][0],both_maxi[1][2][1],both_maxi[1][2][2],both_maxi[1][2][3],both_maxi[1][2][4],both_maxi[1][2][5]]) # highpas

        # Timeseries
        ts_rot_lp.append([both_ts[0][0][0], both_ts[0][0][1], both_ts[0][0][2], both_ts[0][0][3]])  # lowpass
        ts_rot_hp.append([both_ts[1][0][0], both_ts[1][0][1], both_ts[1][0][2], both_ts[1][0][3]])  # highpass

        ts_disp_lp.append([both_ts[0][1][0], both_ts[0][1][1], both_ts[0][1][2], both_ts[0][1][3], both_ts[0][1][4], both_ts[0][1][5]])  # lowpass
        ts_disp_hp.append([both_ts[1][1][0], both_ts[1][1][1], both_ts[1][1][2], both_ts[1][1][3], both_ts[1][1][4], both_ts[1][1][5]])  # highpass

        ts_acc_lp.append([both_ts[0][2][0], both_ts[0][2][1], both_ts[0][2][2], both_ts[0][2][3], both_ts[0][2][4], both_ts[0][2][5]])  # lowpass
        ts_acc_hp.append([both_ts[1][2][0], both_ts[1][2][1], both_ts[1][2][2], both_ts[1][2][3], both_ts[1][2][4], both_ts[1][2][5]])  # highpas
    except:
        print('no data for times: ' + date_name)
        continue

# Now plot the error over max displacement etc.
color = ['red', 'k', 'grey']
marker = ['d','*','.']
labels = ['euler', 'rot + spin rc', 'euler + spin rc']
# euler_a_all_lp, rot_a_err_all_lp, euler_a_err_all_lp
# Rotation
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex='col')
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
for neq in range(len(max_rot_hp)):
    absolut_hp = numpy.asarray(max_rot_hp[neq][0])
    absolut_lp = numpy.asarray(max_rot_lp[neq][0])
    for j in range(1,len(max_rot_hp[neq])):
        maximum_error_hp = numpy.asarray(max_rot_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_rot_lp[neq][j])
        for i in range(3):
            axs[i,0].scatter(abs(absolut_hp[i]), abs(maximum_error_hp)[i]/abs(absolut_hp[i])*100, color=color[j-1], marker=marker[j-1])
            axs[i,1].scatter(abs(absolut_lp[i]), abs(maximum_error_lp)[i]/abs(absolut_lp[i])*100, color=color[j-1], marker=marker[j-1])

axs[0, 0].set_title('a) highpass 0.1 Hz', loc='left')
axs[0, 1].set_title('b) lowpass 0.1 Hz', loc='left')
axs[0, 0].set_ylabel('East error [%]')
axs[1, 0].set_ylabel('North error [%]')
axs[2, 0].set_ylabel('Up error [%]')
axs[2, 0].set_xlabel('max amplitude [rad]')
axs[2, 1].set_xlabel('max amplitude [rad]')
if minmag <4:
    for j in range(2):
        for i in range(3):
            axs[i, j].set_yscale('log')
            axs[i, j].set_xscale('log')

custom_lines = [plt.Line2D([0], [0], color=color[i], marker=marker[i], linestyle='', label=labels[i]) for i in range(len(labels))]
fig.legend(handles=custom_lines, loc='upper right', ncol=len(labels))

fig.savefig('%s/Angle_error_M%s_4paper.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')


# Comparing to no rotation correction for displacement.
#disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp
color = ['midnightblue','cornflowerblue','red', 'k', 'grey']
marker = ['X','D','d','*','.']
labels = ['demean', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex='col')
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
for neq in range(len(max_disp_hp)):
    absolut_hp = numpy.asarray(max_disp_hp[neq][0])
    absolut_lp = numpy.asarray(max_disp_lp[neq][0])
    for j in range(1,len(max_disp_hp[neq])):
        maximum_error_hp = numpy.asarray(max_disp_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_disp_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp)[i]/abs(absolut_hp[i]) * 100,color=color[j - 1], marker=marker[j - 1])
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp)[i]/abs(absolut_lp[i]) * 100,color=color[j - 1], marker=marker[j - 1])

axs[0, 0].set_title('a) highpass 0.1 Hz', loc='left')
axs[0, 1].set_title('b) lowpass 0.1 Hz', loc='left')
axs[0, 0].set_ylabel('East error [%]')
axs[1, 0].set_ylabel('North error [%]')
axs[2, 0].set_ylabel('Up error [%]')
axs[2, 0].set_xlabel('max amplitude [m]')
axs[2, 1].set_xlabel('max amplitude [m]')
if minmag <4:
    for j in range(2):
        for i in range(3):
            axs[i, j].set_yscale('log')
            axs[i, j].set_xscale('log')
custom_lines = [plt.Line2D([0], [0], color=color[i], marker=marker[i], linestyle='', label=labels[i]) for i in range(len(labels))]
fig.legend(handles=custom_lines, loc='upper right', ncol=len(labels))

fig.savefig('%s/Disp_error_M%s_4paper.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')
'''
# Comparing to simple rotation correction for displacement.
#disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp
color = ['red', 'grey', 'k']
marker = ['d','.','*']
labels = ['euler', 'rot + spin', 'euler + spin']
fig, axs = plt.subplots(3,2, sharex=True)
plt.subplots_adjust(hspace=0)
for neq in range(len(max_disp_hp)):
    absolut_hp = numpy.asarray(max_disp_hp[neq][2])
    absolut_lp = numpy.asarray(max_disp_lp[neq][2])
    for j in range(3,len(max_disp_hp[neq])):
        maximum_error_hp = numpy.asarray(max_disp_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_disp_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp)[i]/abs(absolut_hp[i]) * 100,color=color[j - 3], marker=marker[j - 3])
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp)[i]/abs(absolut_lp[i]) * 100,color=color[j - 3], marker=marker[j - 3])

axs[0, 0].set_title('highpass 0.1 Hz')
axs[0, 1].set_title('lowpass 0.1 Hz')
axs[0, 0].set_ylabel('East error [%]')
axs[1, 0].set_ylabel('North error [%]')
axs[2, 0].set_ylabel('Up error [%]')
axs[2, 0].set_xlabel('max amplitude [m]')
axs[2, 1].set_xlabel('max amplitude [m]')

custom_lines = [plt.Line2D([0], [0], color=color[i], marker=marker[i], linestyle='', label=labels[i]) for i in range(len(labels))]
fig.legend(handles=custom_lines, loc='upper center', ncol=len(labels))

fig.savefig('%s/Disp2_error_M%s_4paper.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')'''


# acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_rot_err_rc_m_lp, acc_euler_err_rc_m_lp
color = ['midnightblue','cornflowerblue','red', 'k', 'grey']
marker = ['X','D','d','*','.']
labels = ['demean', 'rot', 'euler', 'rot + spin rc', 'euler + spin rc']
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex='col')
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
for neq in range(len(max_acc_hp)):
    absolut_hp = numpy.asarray(max_acc_hp[neq][0])
    absolut_lp = numpy.asarray(max_acc_lp[neq][0])
    for j in range(1,len(max_acc_hp[neq])):
        maximum_error_hp = numpy.asarray(max_acc_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_acc_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp)[i]/ abs(absolut_hp[i]) * 100,color=color[j - 1], marker=marker[j - 1])
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp)[i]/ abs(absolut_lp[i]) * 100,color=color[j - 1], marker=marker[j - 1])

axs[0, 0].set_title('a) highpass 0.1 Hz', loc='left')
axs[0, 1].set_title('b) lowpass 0.1 Hz', loc='left')
axs[0, 0].set_ylabel('East error [%]')
axs[1, 0].set_ylabel('North error [%]')
axs[2, 0].set_ylabel('Up error [%]')
axs[2, 0].set_xlabel('max amplitude [m/s/s]')
axs[2, 1].set_xlabel('max amplitude [m/s/s]')
if minmag <4:
    for j in range(2):
        for i in range(3):
            axs[i, j].set_yscale('log')
            axs[i, j].set_xscale('log')
custom_lines = [plt.Line2D([0], [0], color=color[i], marker=marker[i], linestyle='', label=labels[i]) for i in range(len(labels))]
fig.legend(handles=custom_lines, loc='upper right', ncol=len(labels))

fig.savefig('%s/Acc_error_M%s_4paper.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')


plt.show()