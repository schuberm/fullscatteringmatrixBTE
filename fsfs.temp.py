import numpy as np
import numpy.matlib
import pylab as pl
from os import listdir
import pickle
import glob
import scipy.linalg
import scipy.misc

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

def expAtaylor(A,n):
	out = np.eye(A.shape[0])
	for ii in xrange(n):
		print(ii)
		out = out +np.linalg.matrix_power(A,ii+1)/scipy.misc.factorial(ii+1)
	return out

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

#SSTG
def k_eff_sstg(w,v,cj,W,q):
	ctot = np.sum(cj)
	q = np.tile(q,(v.shape[0],1))
	ivq = np.sum(v*q*1j, axis=1)
	A = (np.tile(w,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))+ np.diag(ivq)
	Ainv = np.linalg.inv(A)
	D1 = np.diag(w)
	D2 = np.diag(cj/w)
	p = cj/ctot
	num = np.real(np.dot(v.T,np.dot(Ainv,np.dot(D1,np.dot(D2,v)))))
	den = 1 - np.real(np.dot(ivq,np.dot(Ainv,p)))
	return num/den

#FS
def k_eff_fs(w,v,cj,W,d):
	ctot = np.sum(cj)
	vx = np.copy(v[:,0])
	vz = np.copy(v[:,1])
	Dm = np.zeros(vz.shape)
	Dm[v[:,1]<0.0] = 1.0
	n = vz.shape[0]
	V = np.diag(vx/vz)
	A = (np.tile(w/vz,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))
	Ainv = np.linalg.inv(A)*1/d
	expmA= expAtaylor(-A*d,1)
	Dm = np.diag(Dm)
	DeAD = np.dot(Dm,np.dot(expmA,Dm))
	row,column = np.nonzero(DeAD)
	row = np.unique(row)
	column = np.unique(column)
	tmp0 = np.linalg.inv(DeAD[row[:, np.newaxis],column])
	DeAD[row[:, np.newaxis],column] = tmp0
	tmp2 = np.eye(n) - expmA 
	tmp3 = np.eye(n) + np.dot(DeAD,tmp2)
	tmp4 = np.eye(n) - np.dot(Ainv,np.dot(tmp2,tmp3))
	k = d*np.dot(vx.T,np.dot(tmp4,np.dot(Ainv,np.dot(V,cj))))
	return k

def k_bulk(w,v,cj,Winv):
	D1 = np.diag(w)
	D2 = np.diag(cj/w)
	kbulk = np.dot(v.T,np.dot(D1,np.dot(Winv,np.dot(D2,v))))
	return kbulk

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


