import numpy
from obspy import read, read_inventory, UTCDateTime
import matplotlib.pyplot as plt
from functions import eq_kilauea,filter_plotly_maxy_Kilauea_v2, filter_plotly_maxy_Hualien_v2
from roots import get_roots, get_rootsHualien
Hroot_originaldata, Hroot_savefig, Hroot_processeddata = get_rootsHualien()
root_originaldata, root_savefig, root_processeddata = get_roots()


####################################################################################################################
########################################## Data processing for Kilauea EQ's ########################################
####################################################################################################################

## 2. Other earthquakes
# get time of various EQ's:
minmag = 3
ml318 = True

info_eq = eq_kilauea(min_mag=minmag, paper=True)
ampscale=1

# get start and end times of Earthquakes:
date, arrival, starttime, endtime, magnitude, distance = [], [], [], [], [], []

for date_time,arrival_time, mags, dist in zip(info_eq['time'],info_eq['arrivaltime'],info_eq['mag'],info_eq['dist']):

    date_correct = date_time[0:4]+date_time[5:7]+date_time[8:10]
    date.append(date_correct)
    arrival.append(arrival_time)
    starttime.append(UTCDateTime(date_time)+arrival_time-15)
    endtime.append(UTCDateTime(date_time)+arrival_time+30)
    magnitude.append(mags)
    distance.append(dist)

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

    mag_int = int(magnitude)
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
    ts_acc_hp.append([both_ts[1][2][0], both_ts[1][2][1], both_ts[1][2][2], both_ts[1][2][3], both_ts[1][2][4], both_ts[1][2][5]])  # highpass


####################################################################################################################
########################################## Data processing for Hualien EQ's ########################################
####################################################################################################################

#station_name, Lat = 'NA01', 24.46760
station_name, Lat = 'MDSA0', 24.02305
starttime = UTCDateTime('2024-04-02T23:58:10')
endtime = UTCDateTime('2024-04-02T23:59:55')


Hmax_rot_lp = []
Hmax_rot_hp = []
Hmax_disp_lp = []
Hmax_disp_hp = []
Hmax_acc_lp = []
Hmax_acc_hp = []
Hts_rot_lp = []
Hts_rot_hp = []
Hts_disp_lp = []
Hts_disp_hp = []
Hts_acc_lp = []
Hts_acc_hp = []

for station_name, Lat, response in zip(['NA01', 'MDSA0'],[24.46760, 24.02305], [419430*1.02, 418675*0.51]):

    both_maxi, both_ts = filter_plotly_maxy_Hualien_v2(station_name, starttime, endtime, ampscale=1,
                                                       response =response, magnitude=7.4, lpfreq=0.1,
                                                       hpfreq=0.1, plot=True, show=False)
    # now save all the information from the EQ that have both data, and have been made into figures:
    #
    Hmax_rot_lp.append([both_maxi[0][0][0], both_maxi[0][0][1], both_maxi[0][0][2], both_maxi[0][0][3]])  # lowpass
    Hmax_rot_hp.append([both_maxi[1][0][0], both_maxi[1][0][1], both_maxi[1][0][2], both_maxi[1][0][3]])  # highpass

    Hmax_disp_lp.append([both_maxi[0][1][0], both_maxi[0][1][1], both_maxi[0][1][2], both_maxi[0][1][3], both_maxi[0][1][4], both_maxi[0][1][5]])  # lowpass
    Hmax_disp_hp.append([both_maxi[1][1][0], both_maxi[1][1][1], both_maxi[1][1][2], both_maxi[1][1][3], both_maxi[1][1][4], both_maxi[1][1][5]])  # highpass

    Hmax_acc_lp.append([both_maxi[0][2][0], both_maxi[0][2][1], both_maxi[0][2][2], both_maxi[0][2][3], both_maxi[0][2][4] ,both_maxi[0][2][5]])  # lowpass
    Hmax_acc_hp.append([both_maxi[1][2][0], both_maxi[1][2][1], both_maxi[1][2][2], both_maxi[1][2][3], both_maxi[1][2][4], both_maxi[1][2][5]])  # highpas

    # Timeseries
    Hts_rot_lp.append([both_ts[0][0][0], both_ts[0][0][1], both_ts[0][0][2], both_ts[0][0][3]])  # lowpass
    Hts_rot_hp.append([both_ts[1][0][0], both_ts[1][0][1], both_ts[1][0][2], both_ts[1][0][3]])  # highpass

    Hts_disp_lp.append([both_ts[0][1][0], both_ts[0][1][1], both_ts[0][1][2], both_ts[0][1][3], both_ts[0][1][4], both_ts[0][1][5]])  # lowpass
    Hts_disp_hp.append([both_ts[1][1][0], both_ts[1][1][1], both_ts[1][1][2], both_ts[1][1][3], both_ts[1][1][4], both_ts[1][1][5]])  # highpass

    Hts_acc_lp.append([both_ts[0][2][0], both_ts[0][2][1], both_ts[0][2][2], both_ts[0][2][3], both_ts[0][2][4], both_ts[0][2][5]])  # lowpass
    Hts_acc_hp.append([both_ts[1][2][0], both_ts[1][2][1], both_ts[1][2][2], both_ts[1][2][3], both_ts[1][2][4], both_ts[1][2][5]])  # highpass

####################################################################################################################
######################################### Figure for both Hualien & Kilaues ########################################
####################################################################################################################

# Now plot the error over max displacement etc.
color = ['red', 'k', 'grey']
marker = ['d', '*', '.']
labels = ['misorientation rot.', 'rot + spin rc', 'misorientation rot. + spin rc']
# euler_a_all_lp, rot_a_err_all_lp, euler_a_err_all_lp
# Rotation
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex='col')
plt.subplots_adjust(hspace=0.07, wspace=0.07, right=0.98, top=0.84)
# Kilauea
for neq in range(len(max_rot_hp)):
    absolut_hp = numpy.asarray(max_rot_hp[neq][1])
    absolut_lp = numpy.asarray(max_rot_lp[neq][1])
    for j in range(1,len(max_rot_hp[neq])):
        maximum_error_hp = numpy.asarray(max_rot_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_rot_lp[neq][j])
        for i in range(3):
            axs[i,0].scatter(abs(absolut_hp[i]), abs(maximum_error_hp[i])/abs(absolut_hp[i])*100, color=color[j-1], marker=marker[j-1], s=20)
            axs[i,1].scatter(abs(absolut_lp[i]), abs(maximum_error_lp[i])/abs(absolut_lp[i])*100, color=color[j-1], marker=marker[j-1], s=20)
# Hualien
for neq in range(len(Hmax_rot_hp)):
    absolut_hp = numpy.asarray(Hmax_rot_hp[neq][1])
    absolut_lp = numpy.asarray(Hmax_rot_lp[neq][1])
    for j in range(1,len(Hmax_rot_hp[neq])):
        maximum_error_hp = numpy.asarray(Hmax_rot_hp[neq][j])
        maximum_error_lp = numpy.asarray(Hmax_rot_lp[neq][j])
        for i in range(3):
            axs[i,0].scatter(abs(absolut_hp[i]), abs(maximum_error_hp[i])/abs(absolut_hp[i])*100, color=color[j-1], marker=marker[j-1], s=80)
            axs[i,1].scatter(abs(absolut_lp[i]), abs(maximum_error_lp[i])/abs(absolut_lp[i])*100, color=color[j-1], marker=marker[j-1], s=80)

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

fig.savefig('%s/Angle_error_M%s_4paperKH.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')


# Comparing to no rotation correction for displacement.
#disp_obs_demean_lp, disp_obs_rc_lp, disp_euler_rc_lp, disp_rot_err_rc_lp, disp_euler_err_rc_lp
color = ['cornflowerblue', 'red', 'k', 'grey']
marker = ['D', 'd', '*', '.']
labels = ['rot', 'misorientation rot.', 'rot + spin rc', 'misorientation rot. + spin rc']
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.07, right=0.98, top=0.84)
# Kilauea
for neq in range(len(max_disp_hp)):
    absolut_hp = numpy.asarray(max_disp_hp[neq][1])
    absolut_lp = numpy.asarray(max_disp_lp[neq][1])
    for j in range(2,len(max_disp_hp[neq])):
        maximum_error_hp = numpy.asarray(max_disp_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_disp_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp[i])/abs(absolut_hp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=20)
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp[i])/abs(absolut_lp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=20)
# Hualien
for neq in range(len(Hmax_disp_hp)):
    absolut_hp = numpy.asarray(Hmax_disp_hp[neq][1])
    absolut_lp = numpy.asarray(Hmax_disp_lp[neq][1])
    for j in range(2,len(Hmax_disp_hp[neq])):
        maximum_error_hp = numpy.asarray(Hmax_disp_hp[neq][j])
        maximum_error_lp = numpy.asarray(Hmax_disp_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp[i])/abs(absolut_hp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=80)
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp[i])/abs(absolut_lp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=80)

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

fig.savefig('%s/Disp_error_M%s_4paperKH.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')

# acc_obs_demean_lp, acc_obs_rc_m_lp, acc_euler_rc_m_lp, acc_rot_err_rc_m_lp, acc_euler_err_rc_m_lp
color = ['cornflowerblue', 'red', 'k', 'grey']
marker = ['D', 'd', '*', '.']
labels = ['rot', 'misorientation rot.', 'rot + spin rc', 'misorientation rot. + spin rc']
fig, axs = plt.subplots(3,2, figsize=(11,5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.07, right=0.98, top=0.84)
# Kilauea
for neq in range(len(max_acc_hp)):
    absolut_hp = numpy.asarray(max_acc_hp[neq][1])
    absolut_lp = numpy.asarray(max_acc_lp[neq][1])
    for j in range(2,len(max_acc_hp[neq])): # 2 because start at 'rot'
        maximum_error_hp = numpy.asarray(max_acc_hp[neq][j])
        maximum_error_lp = numpy.asarray(max_acc_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp[i])/ abs(absolut_hp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=20)
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp[i])/ abs(absolut_lp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=20)
# Hualien
for neq in range(len(Hmax_acc_hp)):
    absolut_hp = numpy.asarray(Hmax_acc_hp[neq][1])
    absolut_lp = numpy.asarray(Hmax_acc_lp[neq][1])
    for j in range(2,len(Hmax_acc_hp[neq])): # 2 because start at 'rot'
        maximum_error_hp = numpy.asarray(Hmax_acc_hp[neq][j])
        maximum_error_lp = numpy.asarray(Hmax_acc_lp[neq][j])
        for i in range(3):
            axs[i, 0].scatter(abs(absolut_hp[i]),abs(maximum_error_hp[i])/ abs(absolut_hp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=80)
            axs[i, 1].scatter(abs(absolut_lp[i]),abs(maximum_error_lp[i])/ abs(absolut_lp[i]) * 100,color=color[j - 2], marker=marker[j - 2], s=80)

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

fig.savefig('%s/Acc_error_M%s_4paperKH.png' %(root_savefig,minmag), dpi=300, bbox_inches='tight')


plt.show()