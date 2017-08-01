# planck_parabel.py - plots flux as a function of frequency from FIRAS@COBE
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
eee = data[:,3] / 1000                  # error in MJy/sr

ngrid = 1001
delta = 3e-4
tt = np.linspace(-delta,+delta,ngrid)
chi2 = np.zeros(ngrid)

def planck(nu,t):
        result = s0 * nu**3 / (np.exp(spirou_hplanck * nu / spirou_kboltzmann / t) - 1.0)
        return(result)

for i in range(ngrid):
        aux = (cmb - planck(nu,tt[i]+2.72501274898)) / eee
        chi2[i] = np.sum(aux**2)

dof = len(data) - 1

plt.plot(tt,chi2/dof,'b-')
plt.xlim([-delta,+delta])

plt.xlabel('CMB-temperature $T-T_\mathrm{CMB}$')
plt.ylabel('quality of the fit $\Delta\chi^2/\mathrm{dof}$')

plt.show()
