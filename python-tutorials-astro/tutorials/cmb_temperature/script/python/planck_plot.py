# planck_plot.py - plots flux as a function of frequency from FIRAS@COBE
# by Bjoern Malte Schaefer

import numpy as np
import pylab as plt

spirou_clight = 299792458               # speed of light
spirou_kboltzmann = 1.3806488e-23       # Boltzmann constant
spirou_hplanck = 6.62606957e-34         # Planck constant

s0 = 2 * spirou_hplanck / spirou_clight**2 * 1e20

plt.close()

data = np.loadtxt('../data/firas_spectrum.data')
nu = data[:,0] * spirou_clight * 1e2    # frequency in 1/s
cmb = data[:,1]                         # flux in MJy/sr
plt.plot(nu,cmb,'ro',label='data')

def planck(nu,t):
        result = s0 * nu**3 / (np.exp(spirou_hplanck * nu / spirou_kboltzmann / t) - 1.0)
        return(result)

t = 3
cmb_try = planck(nu,t)
plt.plot(nu,cmb_try,'b-',label='bad fit, $T=3$ K')

t = 2.725
cmb_try = planck(nu,t)
plt.plot(nu,cmb_try,'g--',label='good fit, $T=2.725$ K')

plt.xlabel(r'frequency $\nu$ in [Hz]')
plt.ylabel(r'flux $S(\nu)$ in [MJy$/$sr]')

plt.legend(loc='upper right')

plt.show()
