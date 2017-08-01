# supernova_fit.py - plots supernova distance redshift relation
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

data1 = np.loadtxt('../data/supernovae.data')
z1 = data1[:,0]
my1 = data1[:,1]
me1 = data1[:,2]

data2 = np.loadtxt('../data/grbs.data')
z2 = data2[:,0]
my2 = data2[:,1]
me2 = data2[:,2]

z=np.append(z1,z2)
my=np.append(my1,my2)
me=np.append(me1,me2)

plt.errorbar(z,my,me,fmt='b.',label='supernova data')

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
guess = np.array([0.3])
omega_fit, covar = opt.curve_fit(vmu, z, my, guess, me)

mu_fit = vmu(z,omega_fit)
plt.plot(z,mu_fit,'r-',label='best fit cosmology')

tstring = '$\Omega_m=%1.4f \pm %1.4f$' % (omega_fit[0], covar[0,0])
plt.title(tstring)

plt.xlabel('redshift $z$')
plt.ylabel('distance modulus $\mu$')

plt.legend(loc='lower right')
plt.show()
