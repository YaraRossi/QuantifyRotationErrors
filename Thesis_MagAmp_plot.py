import matplotlib.pyplot as plt
import numpy
import matplotlib.lines as mlines
import os

root = '/Users/yararossi/Documents/Work/Random_Workstuff/Thesis_plot'
root_savefig = '/Users/yararossi/Documents/Work/Towards_Quantification/3_Projects/AttitudeEquation/Figures_coding/4SSA'

fig, ax = plt.subplots(1,2, figsize=(11,5), sharex=True)
# Add Earth Spin and blueSeis and gravity
size = 'small'
gravi = 9.81
earth_rr = 7.3 *1e-5
blueSeis3A = 25 *1e-9

freq = numpy.logspace(numpy.log10(0.001),numpy.log10(70), num=1000)
blue = numpy.logspace(numpy.log10(3e-10),numpy.log10(1.5e-7), num=1000) # Bernauer et al. 2018


ax[0].axhline(y=gravi, color='blue')
#ax[1].axhline(y=earth_rr, color='blue')
plt.arrow(0.17, 0.75, -0.12, 0, head_width=0.03, head_length=0.03, fc='blue', ec='blue', transform=ax[1].transAxes, zorder=20)
ax[1].plot(freq, blue, color='red')

for filename in os.listdir(root):
    if 'Mag' in filename:
        data = numpy.loadtxt('%s/%s' %(root,filename), delimiter = '\t')

        if '20km' in filename:
            liner = '-'
            color = 'k'
        elif '200km' in filename:
            liner = '-.'
            color = 'grey'
        elif '2000km' in filename:
            liner = '--'
            color = 'brown'
        if '200km' in filename:
            continue
        ax[0].loglog(data[:,0], data[:,1], color=color, linestyle=liner)
        ax[1].loglog(data[:,0], data[:,2], color=color, linestyle=liner)

# Text for Rotations
ax[1].text(x=0.4,y=0.0004,rotation= 35,s='M7.5',c='k', size=size)
ax[1].text(x=0.38,y=2.5*1e-5,rotation= 45,s='M6.5',c='k', size=size)
ax[1].text(x=0.65,y=4*1e-6,rotation= 52,s='M5.5',c='k', size=size)
ax[1].text(x=2.8,y=1.3*1e-5,rotation= 50,s='M4.5',c='k', size=size)
ax[1].text(x=4.5,y=2*1e-6,rotation= 55,s='M2.5',c='k', size=size)
ax[1].text(x=10,y=1.2*1e-6,rotation= 20,s='M1.5 - 10 km',c='k', size=size)

#ax[1].text(x=10,y=3*1e-5,s='M5.5 - 100 km',c='grey', size=size)


ax[1].text(x=0.012,y=8.5*1e-12,rotation= 59,s='M8',c='brown', size=size)
ax[1].text(x=0.025,y=8.5*1e-12,rotation= 59,s='M7',c='brown', size=size)
ax[1].text(x=0.062,y=8.5*1e-12,rotation= 59,s='M6 - 2000 km',c='brown', size=size)

# text for specifics:

ax[0].text(x=0.012,y=gravi*1.2,s='Gravity',c='blue', size=size)
ax[1].text(x=0.012,y=1.3e-4,s='Earth spin',c='blue', size=size)
ax[1].text(x=0.012,y=5e-9,s='blueSeis-3A\nself noise',c='red', size=size)


# Legend

'''line1 = mlines.Line2D([], [], color='k', linestyle='-', label='10 km')
line2 = mlines.Line2D([], [], color='grey', linestyle='-.', label='100 km')
line3 = mlines.Line2D([], [], color='brown', linestyle='--', label='1000 km')
legend_patches = [line1, line2, line3]
legend_labels = ['near field, 10 km', 'regional, 100 km', 'far field, 1000 km']'''

line1 = mlines.Line2D([], [], color='k', linestyle='-', label='10 km')
line3 = mlines.Line2D([], [], color='brown', linestyle='--', label='1000 km')
legend_patches = [line1, line3]
legend_labels = ['near field', 'far field']
ax[1].legend(legend_patches, legend_labels, loc='upper left')

# Add other stuff and things
ax[0].set_ylabel('Acceleration [m/s/s]')
ax[1].set_ylabel('Rotation rate [rad/s]')
ax[0].set_xlabel('Frequency [Hz]')
ax[1].set_xlabel('Frequency [Hz]')
ax[0].grid('on')
ax[1].grid('on')
plt.subplots_adjust(wspace=0.22)
for ax in ax:
    ax.set_xlim(left=0.01, right=100)
plt.savefig('%s/thesis_magamp.png' %root_savefig, dpi=300, bbox_inches='tight')


############# separate figures ################
size = 'small'
gravi = 9.81
earth_rr = 7.3 * 1e-5
blueSeis3A = 25 * 1e-9

freq = numpy.logspace(numpy.log10(0.001), numpy.log10(70), num=1000)
blue = numpy.logspace(numpy.log10(3e-10), numpy.log10(1.5e-7), num=1000)  # Bernauer et al. 2018

# First Figure
fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.axhline(y=gravi, color='blue')
plt.subplots_adjust(left=0.2, right=0.97)
for filename in os.listdir(root):
    if 'Mag' in filename:
        data = numpy.loadtxt(f'{root}/{filename}', delimiter='\t')

        if '20km' in filename:
            liner = '-'
            color = 'k'
        elif '200km' in filename:
            liner = '-.'
            color = 'grey'
        elif '2000km' in filename:
            liner = '--'
            color = 'brown'
        if '200km' in filename:
            continue
        ax1.loglog(data[:, 0], data[:, 1], color=color, linestyle=liner)

ax1.set_ylabel('Acceleration [m/s/s]')
ax1.set_xlabel('Frequency [Hz]')
ax1.grid('on')
ax1.set_xlim(left=0.01, right=100)
ax1.text(x=0.012, y=gravi * 1.2, s='Gravity', c='blue', size=size)
plt.savefig(f'{root_savefig}/thesis_magamp_acceleration.png', dpi=300, bbox_inches='tight')

# Second Figure
fig2, ax2 = plt.subplots(figsize=(5.37, 5))
plt.subplots_adjust(left=0.03, right=0.80, bottom=0.094)
ax2.plot(freq, blue, color='red', linestyle='-.')
ax2.set_title('b)', loc='left')
plt.arrow(0.17, 0.75, -0.12, 0, head_width=0.03, head_length=0.03, fc='blue', ec='blue', transform=ax2.transAxes, zorder=20)


for filename in os.listdir(root):
    if 'Mag' in filename:
        data = numpy.loadtxt(f'{root}/{filename}', delimiter='\t')

        if '20km' in filename:
            liner = '-'
            color = 'k'
        elif '200km' in filename:
            liner = '-.'
            color = 'grey'
        elif '2000km' in filename:
            liner = '--'
            color = 'brown'
        if '200km' in filename:
            continue
        ax2.loglog(data[:, 0], data[:, 2], color=color, linestyle=liner)

# Text for Rotations
ax2.text(x=0.4, y=0.0004, rotation=35, s='M7.5', c='k', size=size)
ax2.text(x=0.38, y=2.5 * 1e-5, rotation=45, s='M6.5', c='k', size=size)
ax2.text(x=0.65, y=4 * 1e-6, rotation=52, s='M5.5', c='k', size=size)
ax2.text(x=2.8, y=1.3 * 1e-5, rotation=50, s='M4.5', c='k', size=size)
ax2.text(x=4.5, y=2 * 1e-6, rotation=55, s='M2.5', c='k', size=size)
ax2.text(x=10, y=1.2 * 1e-6, rotation=20, s='M1.5 - 10 km', c='k', size=size)
ax2.text(x=0.012, y=8.5 * 1e-12, rotation=59, s='M8', c='brown', size=size)
ax2.text(x=0.025, y=8.5 * 1e-12, rotation=59, s='M7', c='brown', size=size)
ax2.text(x=0.062, y=8.5 * 1e-12, rotation=59, s='M6 - 2000 km', c='brown', size=size)
ax2.text(x=0.012, y=1.3e-4, s='Earth spin', c='blue', size=size)
ax2.text(x=0.012, y=5e-9, s='blueSeis-3A\nself noise', c='red', size=size)

# Legend
line1 = mlines.Line2D([], [], color='k', linestyle='-', label='10 km')
line3 = mlines.Line2D([], [], color='brown', linestyle='--', label='1000 km')
legend_patches = [line1, line3]
legend_labels = ['near field @ 10 km', 'far field @ 2000 km']
ax2.legend(legend_patches, legend_labels, loc='lower right')

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')

ax2.set_ylabel('Rotation rate [rad/s]')
ax2.set_xlabel('Frequency [Hz]')
ax2.grid('on')
ax2.set_xlim(left=0.01, right=100)

# adding a specific EQ
plt.scatter(4, 0.006, marker='*', color='darkred', edgecolors='k', s=150)
ax2.text(x=1.2, y=0.005, s='Mw 5.3', c='darkred', size=size)
plt.scatter(4, 0.0015, marker='*', color='pink', edgecolors='k', s=150)
ax2.text(x=1.2, y=0.00065, s='Mw 7.4', c='pink', size=size)

plt.savefig(f'{root_savefig}/thesis_magamp_rotation.png', dpi=300, bbox_inches='tight')

plt.show()