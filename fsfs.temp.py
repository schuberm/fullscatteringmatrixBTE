import numpy as np
import numpy.matlib
import pylab as pl
from os import listdir
import pickle
import glob
import scipy.linalg
import scipy.misc
from utils import *

nq = 256
nat3 = 6
n = 3
T = 300
filesdir = 'FILESPATH'

#Volume and temperature prefactor from thermal2k
f = open(filesdir+'lambda.f.tk','rb')
lambdua = np.fromfile(f, dtype='float64',count=1, sep=' ')
f.close()

omega = 1/(-lambdua*(T**2 * K_BOLTZMANN_RY))
#BRING TO SI units
omega = omega*RY_TO_METER**3

#Ain_read = np.zeros((nq,nq*nat3*nat3*3))
#f = open(filesdir+'A.f.tk','rb')
#for ii in xrange(nq):
#	field = np.fromfile(f, dtype='float64',count=nq*nat3*nat3*3, sep=' ')
#	Ain_read[ii,:] = field
#f.close()

#Ain_m = np.zeros((nq*nat3,nq*nat3))

#for jj in xrange(nq):
#	for ii in xrange(nq):
#		Ain_m[(ii)*nat3:(ii+1)*nat3,(jj)*nat3:(jj+1)*nat3] = (Ain_read[jj, ii*(nat3*nat3*3): ii*(nat3*nat3*3)+(nat3*nat3*3):3]).reshape((nat3,nat3))

#pickle.dump(Ain_m,open("Ain.p","wb"))

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
Ain = pickle.load(open('Ain.p',"rb"))
W = Ain + np.diag(Aoutfull)
nnp1mat = np.tile(nnp1, (W.shape[0], 1))
Wnnp1= W/nnp1mat

#BRING TO SI units
cj = Cj_si(w,T,K_BOLTZMANN_SI)/omega

vx = c[:,0]
vz = c[:,1]
vz[np.abs(vz)<1e-10] = -10
c[:,1] = vz
c[:,0] = vx

keff = k_eff_fs(w[n:],c[n:,:],cj[n:],Wnnp1[n:,n:]/RY_TO_SECOND,DI)*1/nq
print(keff)
np.savetxt('keff.'+str(DI)+'.dat',keff)


