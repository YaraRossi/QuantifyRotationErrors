import obspy
from obspy import read, read_inventory
import matplotlib.pyplot as plt

root_import = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Data/'
root_save = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'
date_name = '201807120400'
obs_rate = read('%s/Kilauea_%s_HJ1.mseed' % (root_import, date_name))
obs_rate += read('%s/Kilauea_%s_HJ2.mseed' % (root_import, date_name))
obs_rate += read('%s/Kilauea_%s_HJ3.mseed' % (root_import, date_name))
inv = read_inventory(root_import + '/station.xml')

#### 1.2 ####
# slice data to only include EQ
starttime = obspy.UTCDateTime('2018-07-12T05:12:40')
endtime = obspy.UTCDateTime('2018-07-12T05:13:00')
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

fig, axs = plt.subplots(3,1, figsize=(8,5))
plt.subplots_adjust(hspace=0, right=0.98)
for i, ch, dir in zip(range(3),['HJE','HJN','HJZ'], ['East','North','Up']):
    ax = axs[i]
    trace = obs_rate.select(channel=ch)
    ax.plot(trace[0].times(), trace[0].data, 'red')
    ax.set_ylabel('%s [rad/s]' %dir)
ax.set_xlabel('Time [s]')

fig.savefig('%s/KilaueaEQ_Ml3_18_20180712T051241' %root_save, dpi=300)
plt.show()

