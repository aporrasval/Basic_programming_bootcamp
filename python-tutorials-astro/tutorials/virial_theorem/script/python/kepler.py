# kepler.py - numerical solution to the Kepler-problem in 1/r^n-potentials and measurement of the energies
# by Bjoern Malte Schaefer, bjoern.malte.schaefer@uni-heidelberg.de

import numpy as np
from scipy import integrate
import pylab as plt

plt.style.use('classic')
plt.close()

# definitions
alpha = 1.1
rini = 1.0
delta = 0.30
xini = [rini,0.0,0.0,np.sqrt(alpha / rini**(alpha+2.0))*(1.0 + delta)]

# differential equation
def dsolver(x,t):
	r = x[0]
	s = x[1]
	phi = x[2]
	psi = x[3]
	return [s,-alpha/r**(alpha+1.0)+r*psi**2,psi,-2.0*s/r*psi]

# numerical solution
time = np.linspace(0.0,1000.0,10000)
solution = integrate.odeint(dsolver,xini,time)

# plot
r = solution[0:-1,0]
s = solution[0:-1,1]
phi = solution[0:-1,2]
psi = solution[0:-1,3]
t = time[0:-1]

x = r*np.cos(phi)
y = r*np.sin(phi)

plt.plot(x,y,'b-')

# kinetic and potential energy
ekin = (s**2 + r**2 * psi**2) / 2
epot = -1.0 / r**alpha
esum = ekin + epot
angmom = r**2 * psi

#plt.plot(t,ekin,'g-',label='kinetic energy')
#plt.plot(t,epot,'b-',label='potential energy')
#plt.plot(t,ekin+epot,'r--',label='total energy')

plt.legend(loc='upper right')
plt.xlabel('time $t$')
plt.show()

r = np.sum(ekin) / np.sum(epot)
print(r)
