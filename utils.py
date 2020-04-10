import numpy as np

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

#SSTG
def k_eff(w,v,cj,W,q):
	ctot = np.sum(cj)
	q = np.tile(q,(v.shape[0],1))
	ivq = np.sum(v*q*1j, axis=1)
	#A = W+ np.diag(ivq)
	A = (np.tile(w,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))+ np.diag(ivq)
	#A = np.dot(w,np.dot(W,1/w)) + np.diag(ivq)
	Ainv = np.linalg.inv(A)
	D1 = np.diag(w)
	D2 = np.diag(cj/w)
	p = cj/ctot
	num = np.real(np.dot(v.T,np.dot(Ainv,np.dot(D1,np.dot(D2,v)))))
	den = 1 - np.real(np.dot(ivq,np.dot(Ainv,p)))
	return num/den

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

def deltaT(w,v,cj,W,q,omega):
	ctot = np.sum(cj)
	q = np.tile(q,(v.shape[0],1))
	D = 1j*np.diag(np.sum(v*q, axis=1) + omega)
	A = (np.tile(w,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))+ D
	Ainv = np.linalg.inv(A)
	p = cj/ctot
	num = 1 - 1j*np.sum(np.dot(D,np.dot(Ainv,p)))
	den = 1j*omega*ctot + np.sum(np.dot(D,np.dot(Ainv,np.dot(D,cj))))
	return num/den

def deltaT_ed(w,v,cj,W,q,omega):
	ctot = np.sum(cj)
	q = np.tile(q,(v.shape[0],1))
	D = 1j*np.diag(np.sum(v*q, axis=1) + omega)
	A = (np.tile(w,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))+ D
	Ainv = np.linalg.inv(A)
	p = cj/ctot
	#num = 1 - 1j*np.sum(np.dot(D,np.dot(Ainv,p)))
	#den = 1j*omega*ctot + np.sum(np.dot(D,np.dot(Ainv,np.dot(D,cj))))
	num = np.sum(np.dot(Ainv,p))
	den = np.sum(1j*np.dot(Ainv,np.dot(D,cj)))
	return num/den

def k_eff_p(w,v,cj,W,q,p):
	ctot = np.sum(cj)
	q = np.tile(q,(v.shape[0],1))
	ivq = np.sum(v*q*1j, axis=1)
	#A = W+ np.diag(ivq)
	A = (np.tile(w,(W.shape[0],1))).T*W*(np.tile(1/w,(W.shape[0],1)))+ np.diag(ivq)
	#A = np.dot(w,np.dot(W,1/w)) + np.diag(ivq)
	Ainv = np.linalg.inv(A)
	D1 = np.diag(w)
	D2 = np.diag(cj/w)
	#p = cj/ctot
	num = np.real(np.dot(v.T,np.dot(Ainv,np.dot(D1,np.dot(D2,v)))))
	den = 1 - np.real(np.dot(ivq,np.dot(Ainv,p)))
	return num/den

def k_bulk(w,v,cj,W):
	#Winv = np.linalg.inv(W)
	D1 = np.diag(w)
	D2 = np.diag(cj/w)
	kbulk = np.dot(v.T,np.dot(D1,np.dot(W,np.dot(D2,v))))
	return kbulk

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