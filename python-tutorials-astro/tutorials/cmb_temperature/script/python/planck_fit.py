# planck_fit.py - fits a Planck-spectrum to FIRAS@COBE data
# by Bjoern Malte Schaefer

import numpy as np
import pylab as plt
import scipy.optimize as opt

spirou_clight = 299792458               # speed of light
spirou_kboltzmann = 1.3806488e-23       # Boltzmann constant
spirou_hplanck = 6.62606957e-34         # Planck constant

s0 = 2 * spirou_hplanck / spirou_clight**2 * 1e20

plt.style.use('classic')
plt.close()

def planck(nu,t):
	result = s0 * nu**3 / (np.exp(spirou_hplanck * nu / spirou_kboltzmann / t) - 1.0)
	return(result)

def wien(nu,t):
	result = s0 * nu**3 / np.exp(spirou_hplanck * nu / spirou_kboltzmann / t)
	return(result)

data = np.loadtxt('../data/firas_spectrum.data')
nu = data[:,0] * spirou_clight * 1e2    # frequency in 1/s
cmb = data[:,1]                         # flux in MJy/sr
eee = data[:,3] / 1000                  # error in MJy/sr

nuscale = 1e9
plt.errorbar(nu/nuscale,cmb,100*eee,fmt='ro',label=r'FIRAS data, errors $\times$ 100')

t = 4

guess = t
[t], covar = opt.curve_fit(planck, nu, cmb, guess)
print(t)

nusmooth = np.linspace(np.min(nu),np.max(nu),1000)

cmb_fit = planck(nusmooth,t)
plt.plot(nusmooth/nuscale,cmb_fit,'g-',label='Planck-law')

guess = t
[t], covar = opt.curve_fit(wien, nu, cmb, guess)
print(t)

cmb_fit = wien(nusmooth,t)
plt.plot(nusmooth/nuscale,cmb_fit,'b-',label='Wien-law')

plt.xlabel(r'frequency $\nu$ in [GHz]')
plt.ylabel(r'energy flux $S(\nu)$ in [MJy$/$sr]')

plt.legend(loc='upper right',numpoints = 1)
plt.show()
