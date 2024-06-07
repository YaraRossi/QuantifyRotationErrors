import os

import numpy
import obspy
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from numpy import mean
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()

root_save = '%s/Latitudes' % root_processeddata
#root_save = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/Processed/Latitudes'
#root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'

NAME = ['obs_angle' ,'obs_rr', 'euler_angle' ,'euler_angle_err', 'euler_rr', 'euler_rr_err']

date = '2018_07_12T05_12_15_000000Z'#'2018_07_13T00_41_57_000000Z' #'2018_07_12T05_12_15_000000Z'
freq = 0.1
ss = 20
fig1, axs1 = plt.subplots(3,2, figsize=(11,5))#, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
plt.suptitle('Rotation Angle')
fig2, axs2 = plt.subplots(3,2, figsize=(11,5))#, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.07, wspace=0.25, right=0.98)
plt.suptitle('Rotation Rate')

for latitude in [0,15,20,30,45,60,75,90]:

    # Load Data
    obs_angle = read('%s/Kilauea_%s_1_lat%s_obs_angle_HJE.mseed' %(root_save,date,latitude))
    obs_angle+= read('%s/Kilauea_%s_1_lat%s_obs_angle_HJN.mseed' %(root_save,date,latitude))
    obs_angle+= read('%s/Kilauea_%s_1_lat%s_obs_angle_HJZ.mseed' %(root_save,date,latitude))

    euler_angle = read('%s/Kilauea_%s_1_lat%s_euler_angle_HJE.mseed' %(root_save,date,latitude))
    euler_angle+= read('%s/Kilauea_%s_1_lat%s_euler_angle_HJN.mseed' %(root_save,date,latitude))
    euler_angle+= read('%s/Kilauea_%s_1_lat%s_euler_angle_HJZ.mseed' %(root_save,date,latitude))

    rot_angle_err = read('%s/Kilauea_%s_1_lat%s_rot_angle_err_HJE.mseed' %(root_save,date,latitude))
    rot_angle_err+= read('%s/Kilauea_%s_1_lat%s_rot_angle_err_HJN.mseed' %(root_save,date,latitude))
    rot_angle_err+= read('%s/Kilauea_%s_1_lat%s_rot_angle_err_HJZ.mseed' %(root_save,date,latitude))

    euler_angle_tot = read('%s/Kilauea_%s_1_lat%s_euler_angle_tot_HJE.mseed' %(root_save,date,latitude))
    euler_angle_tot+= read('%s/Kilauea_%s_1_lat%s_euler_angle_tot_HJN.mseed' %(root_save,date,latitude))
    euler_angle_tot+= read('%s/Kilauea_%s_1_lat%s_euler_angle_tot_HJZ.mseed' %(root_save,date,latitude))

    obs_rr = read('%s/Kilauea_%s_1_lat%s_obs_rr_HJE.mseed' %(root_save,date,latitude))
    obs_rr+= read('%s/Kilauea_%s_1_lat%s_obs_rr_HJN.mseed' %(root_save,date,latitude))
    obs_rr+= read('%s/Kilauea_%s_1_lat%s_obs_rr_HJZ.mseed' %(root_save,date,latitude))

    euler_rr = read('%s/Kilauea_%s_1_lat%s_euler_rr_HJE.mseed' %(root_save,date,latitude))
    euler_rr+= read('%s/Kilauea_%s_1_lat%s_euler_rr_HJN.mseed' %(root_save,date,latitude))
    euler_rr+= read('%s/Kilauea_%s_1_lat%s_euler_rr_HJZ.mseed' %(root_save,date,latitude))

    rot_rr_err = read('%s/Kilauea_%s_1_lat%s_rot_rr_err_HJE.mseed' %(root_save,date,latitude))
    rot_rr_err+= read('%s/Kilauea_%s_1_lat%s_rot_rr_err_HJN.mseed' %(root_save,date,latitude))
    rot_rr_err+= read('%s/Kilauea_%s_1_lat%s_rot_rr_err_HJZ.mseed' %(root_save,date,latitude))

    euler_rr_tot = read('%s/Kilauea_%s_1_lat%s_euler_rr_tot_HJE.mseed' %(root_save,date,latitude))
    euler_rr_tot+= read('%s/Kilauea_%s_1_lat%s_euler_rr_tot_HJN.mseed' %(root_save,date,latitude))
    euler_rr_tot+= read('%s/Kilauea_%s_1_lat%s_euler_rr_tot_HJZ.mseed' %(root_save,date,latitude))

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

    if latitude == 20:
        s = ss * 10
    else:
        s = ss
    ### Angle
    # highpass
    for ch,ax in zip(['HJE','HJN','HJZ'],[axs1[0,0],axs1[1,0],axs1[2,0]]):
        max_obs_angle_hp = max(abs(obs_angle_hp.select(channel=ch)[0].data))
        diff1 = max(abs(obs_angle_hp.select(channel=ch)[0].data - euler_angle_hp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_angle_hp.select(channel=ch)[0].data - rot_angle_err_hp.select(channel=ch)[0].data))
        ax.scatter(latitude,diff1/max_obs_angle_hp*100, s=s, marker='d', c='k')
        ax.scatter(latitude,diff2/max_obs_angle_hp*100, s=s, marker='*', c='red')
    # lowpass
    for ch,ax in zip(['HJE','HJN','HJZ'],[axs1[0,1],axs1[1,1],axs1[2,1]]):
        max_obs_angle_lp = max(abs(obs_angle_lp.select(channel=ch)[0].data))
        diff1 = max(abs(obs_angle_lp.select(channel=ch)[0].data - euler_angle_lp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_angle_lp.select(channel=ch)[0].data - rot_angle_err_lp.select(channel=ch)[0].data))
        ax.scatter(latitude,diff1/max_obs_angle_lp*100, s=s, marker='d', c='k')
        ax.scatter(latitude,diff2/max_obs_angle_lp*100, s=s, marker='*', c='red')

    ### Rotationrate
    # highpass
    for ch, ax in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 0], axs2[1, 0], axs2[2, 0]]):
        max_obs_rr_hp = max(abs(obs_rr_hp.select(channel=ch)[0].data))
        diff1 = max(abs(obs_rr_hp.select(channel=ch)[0].data - euler_rr_hp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_rr_hp.select(channel=ch)[0].data - rot_rr_err_hp.select(channel=ch)[0].data))
        ax.scatter(latitude, diff1/max_obs_rr_hp*100, s=s, marker='d', c='k')
        ax.scatter(latitude, diff2/max_obs_rr_hp*100, s=s, marker='*', c='red')
    # lowpass
    for ch, ax in zip(['HJE', 'HJN', 'HJZ'], [axs2[0, 1], axs2[1, 1], axs2[2, 1]]):
        max_obs_rr_lp = max(abs(obs_rr_lp.select(channel=ch)[0].data))
        diff1 = max(abs(obs_rr_lp.select(channel=ch)[0].data - euler_rr_lp.select(channel=ch)[0].data))
        diff2 = max(abs(obs_rr_lp.select(channel=ch)[0].data - rot_rr_err_lp.select(channel=ch)[0].data))
        ax.scatter(latitude, diff1/max_obs_rr_lp*100, s=s, marker='d', c='k')
        ax.scatter(latitude, diff2/max_obs_rr_lp*100, s=s, marker='*', c='red')


for ax in [axs1,axs2]:
    for j in range(2):
        for i in range(3):
            ax[i, j].set_yscale('log')
            #ax[i, j].axvline(x=19.420908, c='grey',linestyle='--')
            #ax[i, j].set_xscale('log')

    ax[0, 0].set_title('highpass 0.1 Hz')
    ax[0, 1].set_title('lowpass 0.1 Hz')

    ax[2, 0].set_xlabel('latitude [°]')
    ax[2, 1].set_xlabel('latitude [°]')

    ax[0, 0].set_ylabel('East Error [%]')
    ax[1, 0].set_ylabel('North Error [%]')
    ax[2, 0].set_ylabel('Z Error [%]')

fig1.savefig('%s/Kilauea_%s_LatitudeFiltering_a.png' %(root_savefig,date), dpi=300)
fig2.savefig('%s/Kilauea_%s_LatitudeFiltering_rr.png' %(root_savefig,date), dpi=300)
plt.show()



