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

plt.errorbar(z,my,me,fmt='b.')

def lumi(z,omega_m):
        aux = quad(d_lumi,0,z,args=(omega_m))
        result = (1.0 + z) * aux[0]
        return(spirou_dhubble * result)

def d_lumi(x,omega_m):
        omega_l = 1.0 - omega_m
        w = -1.0
        hubble = np.sqrt(omega_m * (1.0 + x)**3 + omega_l * (1.0 + x)**(3.0 * (1.0 + w)))
        result = 1.0 / hubble
        return(result)

def mu(z,omega_m):
        aux = lumi(z,omega_m)
        result = 5.0 * np.log10(aux) + 10
        return(result)

vmu = np.vectorize(mu)

omega_m = 0.3
mu_try = vmu(z,omega_m)
plt.plot(z,mu_try,'r-',label='$\Omega_m=0.3$',linewidth=2)

omega_m = 0.5
mu_try = vmu(z,omega_m)
plt.plot(z,mu_try,'g-',label='$\Omega_m=0.5$',linewidth=2)

omega_m = 0.7
mu_try = vmu(z,omega_m)
plt.plot(z,mu_try,'c-',label='$\Omega_m=0.7$',linewidth=2)

omega_m = 1.0
mu_try = vmu(z,omega_m)
plt.plot(z,mu_try,'m-',label='$\Omega_m=1.0$',linewidth=2)

plt.legend(loc='lower right')
plt.xlim([0.0,1.6])
plt.xlabel('redshift $z$')
plt.ylabel('distance modulus $\mu$')
plt.show()
