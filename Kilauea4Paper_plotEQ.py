import obspy
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import numpy
from numpy import mean

root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data'
root_save = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'
date_name = '20180715'
obs_rate = read('%s/Kilauea_%s_HJ1.mseed' % (root_import, date_name))
obs_rate += read('%s/Kilauea_%s_HJ2.mseed' % (root_import, date_name))
obs_rate += read('%s/Kilauea_%s_HJ3.mseed' % (root_import, date_name))
inv = read_inventory(root_import + '/station.xml')

#### 1.2 ####
# slice data to only include EQ
starttime = obspy.UTCDateTime('2018-07-15T13:25:50')
endtime = obspy.UTCDateTime('2018-07-15T13:26:35')
obs_rate = obs_rate.slice(starttime, endtime)

# obs_rate.plot()

#### 1.3 ####
# correct stuff:
# scale data from nrad/s to rad/s
obs_rate.remove_sensitivity(inventory=inv)

# obs_rate.plot()

# orient the sensor correctly. it is oriented 1.8° wrong
obs_rate.rotate(method='->ZNE', inventory=inv, components=["123"])

#obs_rate.plot()
e_rr_fromdata = [mean(obs_rate.select(channel='HJE')[0].data[0:1000]),
                     mean(obs_rate.select(channel='HJN')[0].data[0:1000]),
                     mean(obs_rate.select(channel='HJZ')[0].data[0:1000])]
for ch, offset in zip(['HJE','HJN','HJZ'],e_rr_fromdata):
    obs_rate.select(channel=ch)[0].data = obs_rate.select(channel=ch)[0].data - offset

obs_rate = obs_rate.integrate()

fig, axs = plt.subplots(3,1, figsize=(5,5))
plt.subplots_adjust(hspace=0, right=0.98, left=0.2)
for i, ch, dir in zip(range(3),['HJE','HJN','HJZ'], ['East','North','Up']):
    ax = axs[i]
    trace = obs_rate.select(channel=ch)
    ax.plot(trace[0].times(), trace[0].data, 'red')
    ax.set_ylabel('%s [rad]' %dir)
ax.set_xlabel('Time [s]')
fig.savefig('%s/KilaueaEQ_Mw5_3_20180715T132550' %root_save, dpi=300)
#fig.savefig('%s/KilaueaEQ_Ml3_18_20180712T051241' %root_save, dpi=300)
plt.show()

