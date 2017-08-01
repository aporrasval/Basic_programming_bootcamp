# rotcurve.py - plots galaxy rotation curves
# by Bjoern Malte Schaefer, bjoern.malte.schaefer@uni-heidelberg.de

import numpy as np
import pylab  as plt

plt.style.use('classic')
plt.close()

# load data
data = np.loadtxt('../data/U11616.data')
r = data[:,0]
v = data[:,1]
e = data[:,2]

# plot data
plt.errorbar(r,v,yerr=e,fmt='ro')

plt.xlabel('radius $r$ in arcsec')
plt.ylabel(r'velocity $\upsilon$ in $\mathrm{km}/\mathrm{s}$')

plt.show()
