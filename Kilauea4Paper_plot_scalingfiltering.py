import os
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from numpy import mean, sqrt
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()

root_save = '%s/Scaling' % root_processeddata
root_alleq = '%s/All_EQ' % root_processeddata
root_alleqbig = '%s/All_EQbig' % root_processeddata
#root_save = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/Processed/Scaling'
#root_alleq = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/Processed/All_EQ'
#root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'

NAME = ['obs_angle' ,'obs_rr', 'euler_angle' ,'euler_angle_err', 'euler_rr', 'euler_rr_err']

date = '2018_07_12T05_12_15_000000Z' #'2018_07_13T00_41_57_610339Z' #'2018_07_13T00_41_30_000000Z' #'2018_07_12T05_12_15_000000Z'

freq = 0.1
ss = 20
withother = False # at True, the other EQ's will be plotted.
fig1, axs1 = plt.subplots(3,2, figsize=(11,5))#, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
#plt.suptitle('Rotation Angle')
fig2, axs2 = plt.subplots(3,2, figsize=(11,5))#, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
#plt.suptitle('Rotation Rate')
dates = ['2018_07_13T00_42_12_610339Z', '2018_07_14T04_13_18_908030Z', '2018_07_14T05_07_49_173848Z',
         '2018_07_15T13_25_50_535633Z', '2018_07_12T05_12_26_921263Z']
for ampscale in [0.001,0.01,0.1,1,10,100,1000]:
    if withother:
        if ampscale == 1:
            '''for date, color1, color2 in zip(['2018_07_13T00_41_57_610339Z','2018_07_13T00_41_30_000000Z',
                                             '2018_07_12T05_12_15_000000Z', '2018_07_14T05_07_45_000000Z'], \
                                            ['pink', 'rebeccapurple','red','indianred'],
                                            ['grey', 'darkgrey', 'k', 'lightgrey']):'''

            #for file, marker_circle, marker_star in zip(os.listdir(root_save),marker_circle,marker_star):
            for file in os.listdir(root_alleq):
                if file[8:35] in dates:
                    if '_1_lat19.420908_obs_angle_HJE' in file:
                        date = file[8:35]
                        color1 = 'grey'
                        color2 = 'indianred'
                        # Load Data
                        obs_angle = read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJE.mseed' % (root_alleq, date, ampscale))
                        obs_angle += read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJN.mseed' % (root_alleq, date, ampscale))
                        obs_angle += read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJZ.mseed' % (root_alleq, date, ampscale))

                        euler_angle = read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJE.mseed' % (root_alleq, date, ampscale))
                        euler_angle += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJN.mseed' % (root_alleq, date, ampscale))
                        euler_angle += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJZ.mseed' % (root_alleq, date, ampscale))

                        rot_angle_err = read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJE.mseed' % (root_alleq, date, ampscale))
                        rot_angle_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJN.mseed' % (root_alleq, date, ampscale))
                        rot_angle_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJZ.mseed' % (root_alleq, date, ampscale))

                        euler_angle_tot = read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJE.mseed' % (root_alleq, date, ampscale))
                        euler_angle_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJN.mseed' % (root_alleq, date, ampscale))
                        euler_angle_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJZ.mseed' % (root_alleq, date, ampscale))

                        obs_rr = read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJE.mseed' % (root_alleq, date, ampscale))
                        obs_rr += read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJN.mseed' % (root_alleq, date, ampscale))
                        obs_rr += read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJZ.mseed' % (root_alleq, date, ampscale))

                        euler_rr = read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJE.mseed' % (root_alleq, date, ampscale))
                        euler_rr += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJN.mseed' % (root_alleq, date, ampscale))
                        euler_rr += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJZ.mseed' % (root_alleq, date, ampscale))

                        rot_rr_err = read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJE.mseed' % (root_alleq, date, ampscale))
                        rot_rr_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJN.mseed' % (root_alleq, date, ampscale))
                        rot_rr_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJZ.mseed' % (root_alleq, date, ampscale))

                        euler_rr_tot = read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJE.mseed' % (root_alleq, date, ampscale))
                        euler_rr_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJN.mseed' % (root_alleq, date, ampscale))
                        euler_rr_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJZ.mseed' % (root_alleq, date, ampscale))

                        # Filter Data
                        obs_angle_lp = obs_angle.copy().filter('lowpass', freq=freq)
                        obs_angle_hp = obs_angle.copy().filter('highpass', freq=freq)

                        euler_angle_lp = euler_angle.copy().filter('lowpass', freq=freq)
                        euler_angle_hp = euler_angle.copy().filter('highpass', freq=freq)

                        rot_angle_err_lp = rot_angle_err.copy().filter('lowpass', freq=freq)
                        rot_angle_err_hp = rot_angle_err.copy().filter('highpass', freq=freq)

                        euler_angle_tot_lp = euler_angle_tot.copy().filter('lowpass', freq=freq)
                        euler_angle_tot_hp = euler_angle_tot.copy().filter('highpass', freq=freq)

                        obs_rr_lp = obs_rr.copy().filter('lowpass', freq=freq)
                        obs_rr_hp = obs_rr.copy().filter('highpass', freq=freq)

                        euler_rr_lp = euler_rr.copy().filter('lowpass', freq=freq)
                        euler_rr_hp = euler_rr.copy().filter('highpass', freq=freq)

                        rot_rr_err_lp = rot_rr_err.copy().filter('lowpass', freq=freq)
                        rot_rr_err_hp = rot_rr_err.copy().filter('highpass', freq=freq)

                        euler_rr_tot_lp = euler_rr_tot.copy().filter('lowpass', freq=freq)
                        euler_rr_tot_hp = euler_rr_tot.copy().filter('highpass', freq=freq)

                        if ampscale == 1:
                            s = ss * 10
                        else:
                            s = ss
                        ### Angle
                        # highpass
                        for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs1[0, 0], axs1[1, 0], axs1[2, 0]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
                            max_obs_angle_hp = max(abs(obs_angle_hp.select(channel=ch)[0].data))
                            y = sqrt(max(abs(obs_angle_hp.select(channel='HJE')[0].data))**2+max(abs(obs_angle_hp.select(channel='HJN')[0].data))**2+max(abs(obs_angle_hp.select(channel='HJZ')[0].data))**2)
                            diff1 = max(abs(obs_angle_hp.select(channel=ch)[0].data - euler_angle_hp.select(channel=ch)[0].data))
                            diff2 = max(abs(obs_angle_hp.select(channel=ch)[0].data - rot_angle_err_hp.select(channel=ch)[0].data))
                            ax.scatter(y, diff1 / max_obs_angle_hp * 100, s=s, marker='d', c=color2)
                            ax.scatter(y, diff2 / max_obs_angle_hp * 100, s=s, marker='*', c=color1)
                        # lowpass
                        for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs1[0, 1], axs1[1, 1], axs1[2, 1]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
                            max_obs_angle_lp = max(abs(obs_angle_lp.select(channel=ch)[0].data))
                            y = sqrt(max(abs(obs_angle_lp.select(channel='HJE')[0].data))**2+max(abs(obs_angle_lp.select(channel='HJN')[0].data))**2+max(abs(obs_angle_lp.select(channel='HJZ')[0].data))**2)
                            diff1 = max(abs(obs_angle_lp.select(channel=ch)[0].data - euler_angle_lp.select(channel=ch)[0].data))
                            diff2 = max(abs(obs_angle_lp.select(channel=ch)[0].data - rot_angle_err_lp.select(channel=ch)[0].data))
                            ax.scatter(y, diff1 / max_obs_angle_lp * 100, s=s, marker='d', c=color2)
                            ax.scatter(y, diff2 / max_obs_angle_lp * 100, s=s, marker='*', c=color1)

                        ### Rotationrate
                        # highpass
                        for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 0], axs2[1, 0], axs2[2, 0]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
                            max_obs_rr_hp = max(abs(obs_rr_hp.select(channel=ch)[0].data))
                            y = sqrt(max(abs(obs_rr_hp.select(channel='HJE')[0].data))**2+max(abs(obs_rr_hp.select(channel='HJN')[0].data))**2+max(abs(obs_rr_hp.select(channel='HJZ')[0].data))**2)
                            diff1 = max(abs(obs_rr_hp.select(channel=ch)[0].data - euler_rr_hp.select(channel=ch)[0].data))
                            diff2 = max(abs(obs_rr_hp.select(channel=ch)[0].data - rot_rr_err_hp.select(channel=ch)[0].data))
                            ax.scatter(y, diff1 / max_obs_rr_hp * 100, s=s, marker='d', c=color2)
                            ax.scatter(y, diff2 / max_obs_rr_hp * 100, s=s, marker='*', c=color1)
                        # lowpass
                        for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 1], axs2[1, 1], axs2[2, 1]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
                            max_obs_rr_lp = max(abs(obs_rr_lp.select(channel=ch)[0].data))
                            y = sqrt(max(abs(obs_rr_lp.select(channel='HJE')[0].data))**2+max(abs(obs_rr_lp.select(channel='HJN')[0].data))**2+max(abs(obs_rr_lp.select(channel='HJZ')[0].data))**2)
                            diff1 = max(abs(obs_rr_lp.select(channel=ch)[0].data - euler_rr_lp.select(channel=ch)[0].data))
                            diff2 = max(abs(obs_rr_lp.select(channel=ch)[0].data - rot_rr_err_lp.select(channel=ch)[0].data))
                            ax.scatter(y, diff1 / max_obs_rr_lp * 100, s=s, marker='d', c=color2)
                            ax.scatter(y, diff2 / max_obs_rr_lp * 100, s=s, marker='*', c=color1)


    # Load Data
    date = '2018_07_12T05_12_26_921263Z'
    obs_angle = read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJE.mseed' % (root_save, date, ampscale))
    obs_angle += read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJN.mseed' % (root_save, date, ampscale))
    obs_angle += read('%s/Kilauea_%s_%s_lat19.420908_obs_angle_HJZ.mseed' % (root_save, date, ampscale))

    euler_angle = read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJE.mseed' % (root_save, date, ampscale))
    euler_angle += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJN.mseed' % (root_save, date, ampscale))
    euler_angle += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_HJZ.mseed' % (root_save, date, ampscale))

    rot_angle_err = read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJE.mseed' % (root_save, date, ampscale))
    rot_angle_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJN.mseed' % (root_save, date, ampscale))
    rot_angle_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_angle_err_HJZ.mseed' % (root_save, date, ampscale))

    euler_angle_tot = read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJE.mseed' % (root_save, date, ampscale))
    euler_angle_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJN.mseed' % (root_save, date, ampscale))
    euler_angle_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_angle_tot_HJZ.mseed' % (root_save, date, ampscale))

    obs_rr = read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJE.mseed' % (root_save, date, ampscale))
    obs_rr += read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJN.mseed' % (root_save, date, ampscale))
    obs_rr += read('%s/Kilauea_%s_%s_lat19.420908_obs_rr_HJZ.mseed' % (root_save, date, ampscale))

    euler_rr = read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJE.mseed' % (root_save, date, ampscale))
    euler_rr += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJN.mseed' % (root_save, date, ampscale))
    euler_rr += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_HJZ.mseed' % (root_save, date, ampscale))

    rot_rr_err = read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJE.mseed' % (root_save, date, ampscale))
    rot_rr_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJN.mseed' % (root_save, date, ampscale))
    rot_rr_err += read('%s/Kilauea_%s_%s_lat19.420908_rot_rr_err_HJZ.mseed' % (root_save, date, ampscale))

    euler_rr_tot = read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJE.mseed' % (root_save, date, ampscale))
    euler_rr_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJN.mseed' % (root_save, date, ampscale))
    euler_rr_tot += read('%s/Kilauea_%s_%s_lat19.420908_euler_rr_tot_HJZ.mseed' % (root_save, date, ampscale))

    # Filter Data
    obs_angle_lp = obs_angle.copy().filter('lowpass', freq=freq)
    obs_angle_hp = obs_angle.copy().filter('highpass', freq=freq)

    euler_angle_lp = euler_angle.copy().filter('lowpass', freq=freq)
    euler_angle_hp = euler_angle.copy().filter('highpass', freq=freq)

    rot_angle_err_lp = rot_angle_err.copy().filter('lowpass', freq=freq)
    rot_angle_err_hp = rot_angle_err.copy().filter('highpass', freq=freq)

    euler_angle_tot_lp = euler_angle_tot.copy().filter('lowpass', freq=freq)
    euler_angle_tot_hp = euler_angle_tot.copy().filter('highpass', freq=freq)

    obs_rr_lp = obs_rr.copy().filter('lowpass', freq=freq)
    obs_rr_hp = obs_rr.copy().filter('highpass', freq=freq)

    euler_rr_lp = euler_rr.copy().filter('lowpass', freq=freq)
    euler_rr_hp = euler_rr.copy().filter('highpass', freq=freq)

    rot_rr_err_lp = rot_rr_err.copy().filter('lowpass', freq=freq)
    rot_rr_err_hp = rot_rr_err.copy().filter('highpass', freq=freq)

    euler_rr_tot_lp = euler_rr_tot.copy().filter('lowpass', freq=freq)
    euler_rr_tot_hp = euler_rr_tot.copy().filter('highpass', freq=freq)

    color1 = 'k'
    color2 = 'red'

    s = ss
    ### Angle
    # highpass
    for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs1[0, 0], axs1[1, 0], axs1[2, 0]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
        max_obs_angle_hp = max(abs(obs_angle_hp.select(channel=ch)[0].data))
        y = sqrt(max(abs(obs_angle_hp.select(channel='HJE')[0].data))**2+max(abs(obs_angle_hp.select(channel='HJN')[0].data))**2+max(abs(obs_angle_hp.select(channel='HJZ')[0].data))**2)
        diff1 = max(abs(obs_angle_hp.select(channel=ch)[0].data - euler_angle_hp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_angle_hp.select(channel=ch)[0].data - rot_angle_err_hp.select(channel=ch)[0].data))
        ax.scatter(y, diff1 / max_obs_angle_hp * 100, s=s, marker='d', c=color2)
        ax.scatter(y, diff2 / max_obs_angle_hp * 100, s=s, marker='*', c=color1)
    # lowpass
    for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs1[0, 1], axs1[1, 1], axs1[2, 1]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
        max_obs_angle_lp = max(abs(obs_angle_lp.select(channel=ch)[0].data))
        y = sqrt(max(abs(obs_angle_lp.select(channel='HJE')[0].data))**2+max(abs(obs_angle_lp.select(channel='HJN')[0].data))**2+max(abs(obs_angle_lp.select(channel='HJZ')[0].data))**2)
        diff1 = max(abs(obs_angle_lp.select(channel=ch)[0].data - euler_angle_lp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_angle_lp.select(channel=ch)[0].data - rot_angle_err_lp.select(channel=ch)[0].data))
        ax.scatter(y, diff1 / max_obs_angle_lp * 100, s=s, marker='d', c=color2)
        ax.scatter(y, diff2 / max_obs_angle_lp * 100, s=s, marker='*', c=color1)

    ### Rotationrate
    # highpass
    for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 0], axs2[1, 0], axs2[2, 0]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
        max_obs_rr_hp = max(abs(obs_rr_hp.select(channel=ch)[0].data))
        y = sqrt(max(abs(obs_rr_hp.select(channel='HJE')[0].data))**2+max(abs(obs_rr_hp.select(channel='HJN')[0].data))**2+max(abs(obs_rr_hp.select(channel='HJZ')[0].data))**2)
        diff1 = max(abs(obs_rr_hp.select(channel=ch)[0].data - euler_rr_hp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_rr_hp.select(channel=ch)[0].data - rot_rr_err_hp.select(channel=ch)[0].data))
        ax.scatter(y, diff1 / max_obs_rr_hp * 100, s=s, marker='d', c=color2)
        ax.scatter(y, diff2 / max_obs_rr_hp * 100, s=s, marker='*', c=color1)
    # lowpass
    for ch, ax, ch1, ch2 in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 1], axs2[1, 1], axs2[2, 1]], ['HJN','HJE','HJE'], ['HJZ','HJZ','HJN']):
        max_obs_rr_lp = max(abs(obs_rr_lp.select(channel=ch)[0].data))
        y = sqrt(max(abs(obs_rr_lp.select(channel='HJE')[0].data))**2+max(abs(obs_rr_lp.select(channel='HJN')[0].data))**2+max(abs(obs_rr_lp.select(channel='HJZ')[0].data))**2)
        diff1 = max(abs(obs_rr_lp.select(channel=ch)[0].data - euler_rr_lp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_rr_lp.select(channel=ch)[0].data - rot_rr_err_lp.select(channel=ch)[0].data))
        ax.scatter(y, diff1 / max_obs_rr_lp * 100, s=s, marker='d', c=color2)
        ax.scatter(y, diff2 / max_obs_rr_lp * 100, s=s, marker='*', c=color1)


for ax, unit in zip([axs1,axs2],['[rad]','[rad/s]']):
    for j in range(2):
        for i in range(3):
            ax[i, j].set_yscale('log')
            ax[i, j].set_xscale('log')

    ax[0, 0].set_title('a) highpass 0.1 Hz', loc='left')
    ax[0, 1].set_title('b) lowpass 0.1 Hz', loc='left')

    ax[2, 0].set_xlabel('$\sqrt{E^2+N^2+Z^2}$ %s' % unit)
    ax[2, 1].set_xlabel('$\sqrt{E^2+N^2+Z^2}$ %s' % unit)

    ax[0, 0].set_ylabel('East Error [%]')
    ax[1, 0].set_ylabel('North Error [%]')
    ax[2, 0].set_ylabel('Z Error [%]')

color = ['red', 'k']
marker = ['d', '*']
labels = ['euler', 'rot + spin rc']
custom_lines = [plt.Line2D([0], [0], color=color[i], marker=marker[i], linestyle='', label=labels[i]) for i in range(len(labels))]
fig2.legend(handles=custom_lines, loc='upper right', ncol=len(labels))
fig1.legend(handles=custom_lines, loc='upper right', ncol=len(labels))

if withother:
    fig1.savefig('%s/Kilauea_%s_ScalingFiltering_a_add.png' % (root_savefig, date), dpi=300)
    fig2.savefig('%s/Kilauea_%s_ScalingFiltering_rr_add.png' % (root_savefig, date), dpi=300)
else:
    fig1.savefig('%s/Kilauea_%s_ScalingFiltering_a.png' %(root_savefig,date), dpi=300)
    fig2.savefig('%s/Kilauea_%s_ScalingFiltering_rr.png' %(root_savefig,date), dpi=300)

plt.show()



