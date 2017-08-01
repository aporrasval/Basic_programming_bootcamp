# supernova_plot.py - plots supernova distance redshift relation
# by Bjoern Malte Schaefer

import numpy as np
import pylab as plt
import scipy.optimize as opt
from scipy.integrate import quad

spirou_clight = 299792458
spirou_parsec = 3.08567758e16
spirou_hubble = 70
spirou_dhubble = spirou_clight / spirou_hubble

plt.close()

data = np.loadtxt('../data/supernovae.data')
z = data[:,0]
my = data[:,1]
me = data[:,2]

def lumi(z,omega_m):
        aux = quad(d_lumi,0,z,args=(omega_m))
        result = (1.0 + z) * aux[0]
        return(result)

def d_lumi(x,omega_m):
        omega_l = 1.0 - omega_m
        w = -1.0
        hubble = np.sqrt(omega_m * (1.0 + x)**3 + omega_l * (1.0 + x)**(3.0 * (1.0 + w)))
        result = 1.0 / hubble
        return(result)

def mu(z,omega_m):
        aux = lumi(z,omega_m)
        result = 5.0 * np.log10(spirou_dhubble * aux) + 10
        return(result)

vmu = np.vectorize(mu)

omega = np.linspace(0.1,0.5,100)
chi2 = np.zeros(omega.shape)
n = 0
for om in omega:
	aux = (vmu(z,om) - my)**2 / me**2
	chi2[n]= np.sum(aux)
	n = n + 1
	
dof = len(data) - 1

plt.plot(omega,chi2/dof)
plt.xlabel('matter density $\Omega_m$')
plt.ylabel('quality of the fit $\Delta\chi^2/\mathrm{dof}$')
plt.show()
