# planck_mcmc.py - using Metropolis-Hastings for sampling from a CMB temperature likelihood
# by Bjoern Malte Schaefer, GSFP+/Heidelberg, bjoern.malte.schaefer@uni-heidelberg.de

import numpy as np
import pylab as plt

spirou_clight = 299792458               # speed of light
spirou_kboltzmann = 1.3806488e-23       # Boltzmann constant
spirou_hplanck = 6.62606957e-34         # Planck constant

s0 = 2 * spirou_hplanck / spirou_clight**2 * 1e20

# set values: isotropic proposal distribution with sigma
delta = 3e-5
sigma = 3e-6
nchain = 100000
accept = 0

plt.close()

data = np.loadtxt('../data/firas_spectrum.data')
nu = data[:,0] * spirou_clight * 1e2    # frequency in 1/s
cmb = data[:,1]                         # flux in MJy/sr
eee = data[:,3] / 1000                  # error in MJy/sr

def planck(nu,t):
        result = s0 * nu**3 / (np.exp(spirou_hplanck * nu / spirou_kboltzmann / t) - 1.0)
        return(result)

# define the chi^2-functional
def chi2(tt):
	result = 0.0
	aux = (cmb - planck(nu,tt+2.72501274898)) / eee
	result += np.sum(aux**2)
	return(result)

# allocate memory
tt = np.zeros(nchain)
chi = np.zeros(nchain)

# initialise chain
tt[0] = 0.0
chi[0] = chi2(tt[0])

# Metropolis-Hastings-chain
for i in range(1,nchain):
	tt_try = np.random.normal(tt[i-1],sigma)
	chi_try = chi2(tt_try)

	if(chi_try < chi[i-1]):
		tt[i] = tt_try
		chi[i] = chi_try
		accept = accept + 1
	else:
		u = np.random.uniform(0.0,1.0)
		if(np.exp(chi[i-1]) > u * np.exp(chi_try)):
			tt[i] = tt_try
			chi[i] = chi_try
			accept = accept + 1
		else:
			tt[i] = tt[i-1]
			chi[i] = chi[i-1]

# print acceptance rate
print(accept/nchain)

# plot
plt.hist(tt,40,normed=True)
plt.xlim([-delta,+delta])

plt.xlabel('CMB temperature $T-T_\mathrm{CMB}$')
plt.ylabel('likelihood $\mathcal{L}(T)$')
plt.show()
