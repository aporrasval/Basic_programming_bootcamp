# supermova_mcmc.py - using MH for sampling from a supernova-likelihood in (omega_m,w)
# by Bjoern Malte Schaefer, GSFP+/Heidelberg, bjoern.malte.schaefer@uni-heidelberg.de

import numpy as np
import pylab as plt
from scipy.integrate import quad

plt.close()

# constants
spirou_clight = 299792458
spirou_parsec = 3.08567758e16
spirou_hubble = 70
spirou_dhubble = spirou_clight / spirou_hubble

# set values: isotropic proposal distribution with sigma
sigma = 0.01
nchain = 2000
accept = 0

# load data: redshift, distance modulus, error
data = np.loadtxt('../data/supernovae.data')
z = data[:,0]
my = data[:,1]
me = data[:,2]
ndata = len(z)

# cosmometry: luminosity distance as an integral, distance modulus
def lumi(z,omega_m,w):
	aux = quad(d_lumi,0,z,args=(omega_m,w))
	result = (1.0 + z) * aux[0]
	return(result)

def d_lumi(x,omega_m,w):
	omega_l = 1.0 - omega_m
	hubble = np.sqrt(omega_m * (1.0 + x)**3 + omega_l * (1.0 + x)**(3.0 * (1.0 + w)))
	result = 1.0 / hubble
	return(result)

def mu(z,omega_m,w):
	aux = lumi(z,omega_m,w)
	result = 5.0 * np.log10(spirou_dhubble * aux) + 10.0
	return(result)

# define the chi^2-functional
def chi2(ome,eos):
	result = 0.0
	for i in range(0,ndata):
		aux = (mu(z[i],ome,eos) - my[i]) / me[i]
		result += aux**2
	return(result)

# allocate memory
ome = np.zeros(nchain)
eos = np.zeros(nchain)
chi = np.zeros(nchain)

# initialise chain
ome[0] = 0.3
eos[0] = -1.0
chi[0] = chi2(ome[0],eos[0])

# Metropolis-Hastings-chain
for i in range(1,nchain):
	ome_try = np.random.normal(ome[i-1],sigma)
	eos_try = np.random.normal(eos[i-1],sigma)
	chi_try = chi2(ome_try,eos_try)

	if(chi_try < chi[i-1]):
		ome[i] = ome_try
		eos[i] = eos_try
		chi[i] = chi_try
		accept = accept + 1
	else:
		u = np.random.uniform(0.0,1.0)
		if(np.exp(chi[i-1]) > u * np.exp(chi_try)):
			ome[i] = ome_try
			eos[i] = eos_try
			chi[i] = chi_try
			accept = accept + 1
		else:
			ome[i] = ome[i-1]
			eos[i] = eos[i-1]
			chi[i] = chi[i-1]

# print acceptance rate
print(accept/nchain)

# plot
plt.plot(ome,eos,'bo',markersize=5,markeredgewidth=0.3)
plt.xlabel('matter density $\Omega_m$')
plt.ylabel('equation of state parameter $w$')
plt.xlim([0.1,0.4])
plt.ylim([-1.2,-0.8])

plt.show()
