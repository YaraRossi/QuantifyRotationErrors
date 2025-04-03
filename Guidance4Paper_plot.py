import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import LogLocator, LogFormatter



fig, axs = plt.subplots(1,2, figsize=(9,2), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.07, right=0.98)
for ax in axs:
    ax.set_xlim(left = 1e-6, right = 2e-2)
    ax.set_ylim(top=3e1, bottom=1e-4)


# high frequency
x = numpy.asarray([1.8e-8,0.018])
x_2 = numpy.asarray([1.8e-8,0.02])
y_1 = numpy.asarray([1,1])
y_espin = numpy.asarray([0.0016,0.0016])
y_misor = numpy.asarray([1.6e-6,1.6])

axs[0].loglog(x_2,y_1, color = 'lightblue', label='1 % error')
axs[0].loglog(x_2,y_espin, color = 'black', label=r"rot. + spin rc")
axs[0].loglog(x,y_misor, color = 'red', label='misorientation rot.')
axs[0].set_title('a) highpass 0.1 Hz', loc='left')
axs[0].set_xlabel('total angle $\sqrt{E^2+N^2+Z^2}$ [rad]')
axs[0].set_ylabel('Error [%]')
axs[0].axvspan(0.01125, 1, color='grey', alpha=0.5, lw=0, label='correction necessary')
axs[0].legend(loc='upper left')

# low frequency
x = numpy.asarray([1.35e-9,0.00135])
x_2 = numpy.asarray([1.35e-9,0.02])
y_1 = numpy.asarray([1,1])
y_espin = numpy.asarray([0.246,0.246])
y_misor = numpy.asarray([2.5e-5,24.8])

axs[1].loglog(x_2,y_1, color = 'lightblue', label='1 % error')
axs[1].loglog(x_2,y_espin, color = 'black', label=r"rot. + spin rc")
axs[1].loglog(x,y_misor, color = 'red', label='misorientation rot.')
axs[1].set_title('b) lowpass 0.1 Hz', loc='left')
axs[1].set_xlabel('total angle $\sqrt{E^2+N^2+Z^2}$ [rad]')
axs[1].axvspan(5.44e-5, 1, color='grey', alpha=0.5, lw=0)

for ax in axs:
    ax.yaxis.set_major_locator(LogLocator(numticks=7))
    ax.yaxis.set_minor_locator(LogLocator(subs='all', numticks=7))

plt.show()