import numpy as np
import numpy.matlib
import pylab as pl
from os import listdir
import pickle
import glob
import scipy.optimize
from utils import *

nq = 64*64
nat3 = 6
n = 3
T = 50
filesdir = 'FILESDIR'

#Volume and temperature prefactor from thermal2k
f = open(filesdir+'lambda.f.tk','rb')
lambdua = np.fromfile(f, dtype='float64',count=1, sep=' ')
f.close()

omega = 1/(-lambdua*(T**2 * K_BOLTZMANN_RY))
#BRING TO SI units
omega = omega*RY_TO_METER**3

Aoutfull = np.zeros((nq,nat3))
f = open(filesdir+'Aout.f.tk','rb')
for ii in xrange(nq):
	Aoutfull[ii,:] = np.fromfile(f, dtype='float64',count=nat3, sep=' ')
f.close()

b = np.zeros((nq*nat3,3))
f = open(filesdir+'b.f.tk','rb')
for ii in xrange(nq):
	for kk in xrange(nat3):
		b[ii*nat3+kk,:] = np.fromfile(f, dtype='float64',count=3, sep=' ')
f.close()

w = np.zeros((nq*nat3))
f = open(filesdir+'w.f.tk','rb')
for ii in xrange(nq):
	for kk in xrange(nat3):
		w[ii*nat3+kk] = np.fromfile(f, dtype='float64',count=1, sep=' ')
f.close()
#BRING TO SI units
w =w*RY_TO_JOULE

c = np.zeros((nq*nat3,3))
f = open(filesdir+'c.f.tk','rb')
for ii in xrange(nq):
	for kk in xrange(nat3):
		c[ii*nat3+kk,:] = np.fromfile(f, dtype='float64',count=3, sep=' ')
f.close()
#BRING TO SI units
c = c*RY_TO_METER/RY_TO_SECOND

nnp1 = np.zeros((nq*nat3))
f = open(filesdir+'nnp1.f.tk','rb')
for ii in xrange(nq):
	for kk in xrange(nat3):
		nnp1[ii*nat3+kk] = np.fromfile(f, dtype='float64',count=1, sep=' ')
f.close()

#Rescale A matrix with n(n+1) factor
#A in Lorenzo code is W in our paper, rescaled by n(n+1)
#Will use W moving forward, and A for A = W + diag(v.q*1j)
Aoutfull = np.reshape(Aoutfull,nq*nat3)
#Ain = pickle.load(open('../Ain.y.p',"rb"))
W = np.load('../Ain.d.npy') + np.diag(Aoutfull)
#W = Ain + np.diag(Aoutfull)
nnp1mat = np.tile(nnp1, (W.shape[0], 1))
Wnnp1= W/nnp1mat

#BRING TO SI units
cj = Cj_si(w,T,K_BOLTZMANN_SI)/omega

qx = 2e6

dT = deltaT_ed(w[n:],c[n:,:],cj[n:],Wnnp1[n:,n:]/RY_TO_SECOND,np.array([qx,0,0]),10**OMEGA)

print(dT)
#np.savetxt('dT.'+str(OMEGA)+'.dat',[dT])
pickle.dump(dT,open('dT.'+str(OMEGA)+'.ed.dat',"wb"))




