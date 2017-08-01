# supernova_likelihood.py - plots supernova distance redshift relation
# by Bjoern Malte Schaefer

import numpy as np
import pylab as plt
from scipy.integrate import quad
import matplotlib
import colormaps as cmaps
from matplotlib import cm

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

def lumi(z,omega_m,weos):
        aux = quad(d_lumi,0,z,args=(omega_m,weos))
        result = (1.0 + z) * aux[0]
        return(result)

def d_lumi(x,omega_m,weos):
        omega_l = 1.0 - omega_m
        w = -1.0
        hubble = np.sqrt(omega_m * (1.0 + x)**3 + omega_l * (1.0 + x)**(3.0 * (1.0 + weos)))
        result = 1.0 / hubble
        return(result)

def mu(z,omega_m,weos):
        aux = lumi(z,omega_m,weos)
        result = 5.0 * np.log10(spirou_dhubble * aux) + 10
        return(result)

vmu = np.vectorize(mu)

ngrid = 51
xx = np.linspace(0.1,0.5,ngrid)
yy = np.linspace(-1.4,-0.6,ngrid)
om,we = np.meshgrid(xx,yy,indexing='ij')

chi2 = np.zeros((ngrid,ngrid))

for i in range(ngrid):
	for j in range(ngrid):
		aux = (vmu(z,om[i,j],we[i,j]) - my)**2 / me**2
		chi2[i,j] = np.sum(aux)

chi2best = np.amin(chi2)
like = np.exp(-(chi2-chi2best) / 2.0)
plt.pcolormesh(om,we,like,cmap='jet',shading='gouraud')
plt.colorbar()

plt.xlim([0.1,0.5])
plt.ylim([-1.4,-0.6])

plt.xlabel('matter density $\Omega_m$')
plt.ylabel('dark energy equation of state $w$')
plt.show()
