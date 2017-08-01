# harmonic.py - numerical solution to the harmonic oscillator equation and measurement of the energies
# by Bjoern Malte Schaefer, bjoern.malte.schaefer@uni-heidelberg.de

import numpy as np
from scipy import integrate
import pylab as plt

plt.style.use('classic')
plt.close()

# definitions
plt.close()

def solvr(Y, t):
	return [+Y[1],-Y[0]]

time = np.linspace(0.0,6.0*np.pi,1000)
solution = integrate.odeint(solvr, [1.0, 0.0], time)

# leave out last element which is identical to the first
x = solution[0:-1,0]
y = solution[0:-1,1]
t = time[0:-1]

#plot
plt.plot(t/np.pi,x,'b-',label='position $x$')
plt.plot(t/np.pi,y,'r-',label='velocity $\dot{x}$')

# compute mean kinetic and potential energies
ekin = y**2/2.0
epot = x**2/2.0

#plt.plot(t/np.pi,ekin,'g-',label='kinetic energy')
#plt.plot(t/np.pi,epot,'b-',label='potential energy')
#plt.plot(t/np.pi,ekin+epot,'r--',label='total energy')

r = np.sum(ekin)/np.sum(epot)
print(r)

plt.xlabel('time $t$')
plt.legend(loc='upper right')
plt.show()
