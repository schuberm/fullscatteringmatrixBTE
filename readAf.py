import numpy as np
import numpy.matlib
import pylab as pl
from os import listdir
import pickle
import glob
import scipy.optimize

#thermal2k constants
RY_TO_JOULE =  0.5* 4.35974394e-18
RY_TO_SECOND = 2* 2.418884326505e-17
RY_TO_METER = 5.2917721092e-11
MASS_DALTON_TO_RY = 0.5*1822.88839
RY_TO_WATT = RY_TO_JOULE / RY_TO_SECOND
RY_TO_WATTMM1KM1 = RY_TO_WATT / RY_TO_METER
K_BOLTZMANN_SI   = 1.3806504E-23
HARTREE_SI       = 4.35974394E-18
RYDBERG_SI       = HARTREE_SI/2.0
K_BOLTZMANN_RY   = K_BOLTZMANN_SI / RYDBERG_SI
HBAR =  1.054571800E-34

def df_bose(x,T,K_BOLTZMANN_RY, tol = 1e-12):
	# temp = np.zeros((x.shape[0]))
	# Tm1 = 1/(T*K_BOLTZMANN_RY)
	# expf = np.exp(x[x>=tol]*Tm1)
	# temp[x>=tol] = Tm1 * expf / (expf-1)**2
	temp = np.zeros((x.shape[0]))
	Tm1 = 1/(T*K_BOLTZMANN_RY)
	expf = np.exp(x[x>=tol]*Tm1)
	temp[x>=tol] = (x[x>=tol]*expf )/( K_BOLTZMANN_RY*T*T*(expf-1)**2)
	return temp

def df_bose_si(x,T,K_BOLTZMANN_SI, tol = 1e-30):
	# temp = np.zeros((x.shape[0]))
	# Tm1 = 1/(T*K_BOLTZMANN_RY)
	# expf = np.exp(x[x>=tol]*Tm1)
	# temp[x>=tol] = Tm1 * expf / (expf-1)**2
	temp = np.zeros((x.shape[0]))
	Tm1 = 1/(T*K_BOLTZMANN_SI)
	expf = np.exp(x[x>=tol]*Tm1)
	temp[x>=tol] = (x[x>=tol]*expf )/( K_BOLTZMANN_SI*T*T*(expf-1)**2)
	return temp

def Cj(x,T,K_BOLTZMANN_RY):
	return x*df_bose(x,T,K_BOLTZMANN_RY)

def Cj_si(x,T,K_BOLTZMANN_SI):
	return x*df_bose_si(x,T,K_BOLTZMANN_SI)

def calc_tk(w,v,cj,Ainv):
	k = np.dot(w*v,np.dot(Ainv,v*cj*1/w))
	return k

def calc_caccum(w,vq,cj,Ainv):
	ctot = np.sum(cj)
	vqt = np.tile(vq,(1,Ainv.shape[1]))
	delta = 1.0*np.ones((Ainv.shape[1]))
	Ainv = np.diag(delta) - 1j*vqt*Ainv
	ca = np.dot(w,np.dot(Ainv,cj/ctot*1/w))
	return ca


nq = 1024
nat3 = 6
n = 3
T = 300
filesdir = '../../getA/32.50/'

#Volume and temperature prefactor from thermal2k
f = open(filesdir+'lambda.f.tk','rb')
lambdua = np.fromfile(f, dtype='float64',count=1, sep=' ')
f.close()

omega = 1/(-lambdua*(T**2 * K_BOLTZMANN_RY))
#BRING TO SI units
omega = omega*RY_TO_METER**3

Ain_read = np.zeros((nq,nq*nat3*nat3*3))
f = open(filesdir+'A.f.tk','rb')
for ii in xrange(nq):
	field = np.fromfile(f, dtype='float64',count=nq*nat3*nat3*3, sep=' ')
	Ain_read[ii,:] = field
f.close()

Ain_m = np.zeros((nq*nat3,nq*nat3))

for jj in xrange(nq):
	for ii in xrange(nq):
		Ain_m[(ii)*nat3:(ii+1)*nat3,(jj)*nat3:(jj+1)*nat3] = (Ain_read[jj, ii*(nat3*nat3*3)+1: ii*(nat3*nat3*3)+(nat3*nat3*3)+1:3]).reshape((nat3,nat3))

pickle.dump(Ain_m,open("Ain.y.p","wb"))

