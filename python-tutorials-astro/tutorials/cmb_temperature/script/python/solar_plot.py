# solar_plot.py - plots the solar spectrum
# by Bjoern Malte Schaefer, GSFP+/Heidelberg, bjoern.malte.schaefer@uni-heidelberg.de
# please be careful, you can't use the variable name lambda for the wave length in python, lambda is reserved

import numpy as np
import pylab as plt

data = np.loadtxt('../data/solar_spectrum.data')
wave = data[:,0]
flux = data[:,1]
flux_gt = data[:,2]
flux_dc = data[:,3]

plt.close()

plt.plot(wave,flux,'r-')
plt.plot(wave,flux_gt,'b-')
plt.plot(wave,flux_dc,'y-')
plt.xlabel('wave length $\lambda$ in nm')
plt.ylabel('flux in $\mathrm{W}/\mathrm{m}^2/\mathrm{nm}$')

plt.show()
