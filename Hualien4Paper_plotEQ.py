import obspy
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import numpy
from numpy import mean
from matplotlib.ticker import MultipleLocator
from roots import get_roots, get_rootsHualien

Hroot_originaldata, Hroot_savefig, Hroot_processeddata = get_rootsHualien()
root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
root_save = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'
#station_name, tickloc_rad = 'NA01', [0.00002, 0.00002, 0.00004]
station_name, tickloc_rad = 'MDSA0', [0.0001, 0.00005, 0.0001]
obs_rate = read('%s/TW.%s.01.HJE.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
obs_rate += read('%s/TW.%s.01.HJN.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
obs_rate += read('%s/TW.%s.01.HJZ.D.2024.093.MSEED' % (Hroot_originaldata, station_name))
df=50
lenmean = 4*df
obs_rate.filter('lowpass', freq=int(df / 2), corners=8, zerophase=True).resample(sampling_rate=df)
#### 1.2 ####
# slice data to only include EQ
starttime = obspy.UTCDateTime('2024-04-02T23:58:05')
endtime = obspy.UTCDateTime('2024-04-02T23:59:55')
obs_rate = obs_rate.slice(starttime, endtime)

# obs_rate.plot()

#### 1.3 ####
# correct stuff:
# scale data from nrad/s to rad/s
for tr in obs_rate:
    tr.data = tr.data / 1e9

#obs_rate.plot()
e_rr_fromdata = [mean(obs_rate.select(channel='HJE')[0].data[0:lenmean]),
                     mean(obs_rate.select(channel='HJN')[0].data[0:lenmean]),
                     mean(obs_rate.select(channel='HJZ')[0].data[0:lenmean])]
for ch, offset in zip(['HJE','HJN','HJZ'],e_rr_fromdata):
    obs_rate.select(channel=ch)[0].data = obs_rate.select(channel=ch)[0].data - offset

obs_angle = obs_rate.copy()
obs_angle = obs_angle.integrate()

fig, axs = plt.subplots(3,1, figsize=(11.35,5))
axs[0].set_title('c)', loc='left')
plt.subplots_adjust(hspace=0) #, right=0.8, left=0.2)
for i, ch, dir, tickloc, tickloc_ in zip(range(3),['HJE','HJN','HJZ'], ['East','North','Up'],
                                         [0.001, 0.001, 0.0005],tickloc_rad):
    # rotation rate
    ax = axs[i]
    color='pink'
    trace = obs_rate.select(channel=ch)
    ax.plot(trace[0].times(), trace[0].data, color, linewidth=0.6) #, label='Mw 7.4, Hualien, %s' %station_name)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('%s [rad/s]' %dir, color=color)
    ax.yaxis.set_major_locator(MultipleLocator(tickloc))
    # angles
    ax_ = ax.twinx()
    color='deeppink'
    trace = obs_angle.select(channel=ch)
    ax_.plot(trace[0].times(), trace[0].data, color, linestyle='--', linewidth=0.6)
    ax_.tick_params(axis='y', labelcolor=color)
    ax_.set_ylabel('[rad]', color=color)
    ax_.yaxis.set_major_locator(MultipleLocator(tickloc_))

    ax.set_xlim(left=0, right=110)
    ax.axvline(x=45, color='k', linewidth = 1)

#ax2.text(x=1.2, y=0.005, s='Mw 5.3', c='darkred', size=size)
ax.text(x=87, y=-0.0009, s='Mw 7.4, Hualien, %s' %station_name,c='k')

axs[0].tick_params(axis='x', labelcolor='white')
axs[1].tick_params(axis='x', labelcolor='white')
ax.set_xlabel('Time [s]')

fig.savefig('%s/HualienEQ_%s_20180715T132550' %(root_save,station_name), dpi=300, bbox_inches='tight')
plt.show()

