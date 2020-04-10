import numpy as np
import numpy.matlib
import pylab as pl
from os import listdir
import pickle
import glob
import scipy.optimize
from utils import *

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

