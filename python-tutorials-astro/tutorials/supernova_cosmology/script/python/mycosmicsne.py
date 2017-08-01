import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('Qt4Agg')
matplotlib.use('tkAgg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.ion()
cHo=4.47*10.**9

alpha=2./3.

universe =raw_input('Choose an universe: Open (O), Flat (F), Closed (C)  =  ')
if universe == "O":
    print ('You Choose an Open Universe')
    curv =input('Choose the curvature radius as a function of the Hubble radius: ')
if universe == 'F':
    print 'You Choose a  Flat Universe'
    curv=0
if universe == 'C':
    print 'You Choose a Closed Universe'
    curv =input('Choose the curvature radius as a function of the Hubble radius: ')

alpha =input('Choose an expansion law a(t)=t**alpha  , alpha = ')
ct=cHo*curv

data = np.loadtxt('supernovae.data')
zsn = data[:,0]
my = data[:,1]
me = data[:,2]

new_list1 = []
new_list2 = []
new_list3 = []
with open ("grbs.data", "r") as myfile:
    data=myfile.readlines()
    list1 = data[0].split( )[1::2]
    list2 = data[1].split( )[0::3]
    list3 = data[1].split( )[2::3]
    for item in list1:
        new_list1.append(float(item))
    for item in list2:
        new_list2.append(float(item))
    for item in list3:
        new_list3.append(float(item))

zgrb  = new_list1
mygrb = new_list2
megrb = new_list3

z = np.arange(0.01, 6.51, 0.01)
if alpha !=1:
    if universe == 'F':
        Dl=-cHo*(alpha)/(alpha-1.)*((1+z)-(1+z)**((2*alpha-1)/alpha))
        Dlsn=-cHo*(alpha)/(alpha-1.)*((1+zsn)-(1+zsn)**((2*alpha-1)/alpha))
    if universe == 'C':
        Dl=ct*(1+z)*np.sin(-(alpha)/(alpha-1.)/curv*(1-(1+z)**((alpha-1)/alpha)))
        Dlsn=ct*(1+zsn)*np.sin(-(alpha)/(alpha-1.)/curv*(1-(1+zsn)**((alpha-1)/alpha)))
    if universe == 'O':
       Dl=ct*(1+z)*np.sinh(-(alpha)/(alpha-1.)/curv*(1-(1+z)**((alpha-1)/alpha)))
       Dlsn=ct*(1+zsn)*np.sinh(-(alpha)/(alpha-1.)/curv*(1-(1+zsn)**((alpha-1)/alpha)))

if alpha ==1:
    if universe == 'F':
        Dl=cHo*(1+z)*np.log(1+z)
        Dlsn=cHo*(1+zsn)*np.log(1+zsn)
    if universe == 'C':
        Dl=ct*(1+z)*np.sin(np.log(1+z))
        Dlsn=ct*(1+zsn)*np.sin(np.log(1+zsn))
    if universe == 'O':
       Dl=ct*(1+z)*np.sinh(np.log(1+z))
       Dlsn=ct*(1+zsn)*np.sinh(np.log(1+zsn))



plt.figure(1)
plt.errorbar(zsn,my,me,fmt='-o',color='blue',label='supernova data')
plt.title("Supernova Data")
plt.xlabel("Z")
plt.ylabel("DM")
mu=5*np.log10(Dl/10.)
plt.plot(z,mu,color='green')
plt.xlim(0,1.5)
plt.ylim(32,47)


plt.figure(2)
plt.errorbar(zsn,my,me,fmt='-o',color='blue')
plt.errorbar(zgrb,mygrb,megrb,fmt='-o',color='red')
plt.title("Supernova + GRBs Data")
plt.xlabel("Z")
plt.ylabel("DM")
mu=5*np.log10(Dl/10.)
plt.plot(z,mu,color='green')



musn=5*np.log10(Dlsn/10.)
plt.figure(3)
plt.clf()
plt.errorbar(np.log10(zsn),my-musn,me,fmt='-o',color='blue',label='residuals')
plt.plot(np.log10(zsn),0.*(np.log10(zsn)),color='red')
plt.title("Supernova Data - Residuals")
plt.xlabel("Log10(Z)")
plt.ylabel("Delta DM")
