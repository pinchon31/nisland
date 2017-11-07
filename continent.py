__all__ = ['c_coalesce','c_migrate','c_mkQ','c_mkB','c_mk_F_iicr','c_main_eigenvalue','c_mk_fixed_K_iicrs','c_mk_fixed_k_iicrs']
import copy
import numpy as np
from scipy import linalg
from partition import *


def c_coalesce(p,c1,c2):
  """
  Starting from a state p, produces a list of new states after coalescence of two genes. 
  Each new state is weigthed following the number of genes in the island where the coalescence occurs.
  Warning: c1 is here a coalescence rate (instead of 1/c1 is the Q-matrix preprint)
  """
  lwq = []
  if p[0]>1:
    s=copy.copy(p)
    s[0]-=1
    wq=[s,0.5*c1*p[0]*(p[0]-1)]
    lwq.append(wq)
  for i in range(1,len(p)):
    if p[i]>1:
      s=copy.copy(p)
      s[i]-=1
      s=c_canForm(s)
      wq = [s,0.5*c2*p[i]*(p[i]-1)]
      lwq.append(copy.copy(wq))
  return lwq

def c_migrate(n,p,M1,M2):
  """
  Starting from a state p, produces a list of new states after migration of one gene. 
  For the continent, the migration rate to an island is M1/(2*(n-1)). For an island, it is M2/2.
  """
  lwq=[]
  if p[0]>0:
    v=0.5*M1/(n-1)
    r=copy.copy(p)
    r[0]-=1
    for j in range(1,len(p)):
      s=copy.copy(r)
      s[j]+=1
      s=c_canForm(s)
      wq=[s,p[0]*v]
      lwq.append(wq)
    if len(r)<n:
      s=copy.copy(r)
      s.append(1)     
      wq=[s,p[0]*v*(n-len(p))]
      lwq.append(wq)
  for i in range(1,len(p)):
    r = copy.copy(p)
    r[i]-=1
    r[0]+=1
    r=c_canForm(r)
    wq=[r,0.5*p[i]*M2]
    lwq.append(wq)
  return lwq

def c_mkQ(n,parms,k):
  """
  Components of the basic Q-matrix for the Tk coalescence time.
  parms=[M1,c1,M2,c2]
  """
  lp = c_parts(k,n)
  lp2 = c_parts(k-1,n)
  n1 = len(lp)
  n2 = len(lp2)
  Q = np.zeros((n1+n2,n1+n2))
  for i in range(n1):
    lwq=c_migrate(n,lp[i],parms[0],parms[2])
    for wq in lwq:
      j=lp.index(wq[0])
      Q[i,j]+=wq[1]
    for wq in c_coalesce(lp[i],parms[1],parms[3]):
      j=lp2.index(wq[0])
      Q[i,n1+j]+=wq[1]
    Q[i,i]=0
    for j in range(n1+n2):
      if j!=i: Q[i,i]-=Q[i,j]
  res=[np.array(Q[0:n1,0:n1])]
  if k>2:
    Q2=np.array(Q[0:n1,n1:(n1+n2)])
  else:
    Q2=np.zeros((4,1))
    for i in range(0,4):
      Q2[i,0]=Q[i,n1]+Q[i,n1+1]
  res.append(Q2) 
  return res
    
def c_main_eigenvalue(n,parms,k):
  """
  The largest stricly eigenvalue of the first component of the basic Tk Q-matrix.
  """
  lmu = np.ndarray.tolist(np.linalg.eigvals(c_mkQ(n,parms,k)[0]))
  lmu.sort()
  return lmu[-1]

def c_mkB(n,parms,k):
  """
  Main component of the limit of exp(tQ) when t tends to infinity. 
  """
  Qk=c_mkQ(n,parms,k)
  B=-np.linalg.solve(Qk[0],Qk[1])
  return B
  
def c_mk_F_iicr(n,parms,K,k,st,t):
  """
  Computes the cumulative distribution function F(t)=P(T_{k,sigma}^{(K],n,M} <= t) and the corresponding IICR
  lambda_{k,sigma}^{(K],n,M}(t) = (1-F(t)/F'(t).
  """
  V = np.zeros((1,len(c_parts(K,n))))
  V[0,st]=1
  for l in range(K,k,-1): V = np.dot(V,c_mkB(n,parms,l))
  Q=c_mkQ(n,parms,k)
  Bkt = np.dot(np.identity(len(c_parts(k,n)))-linalg.expm(t*Q[0]),c_mkB(n,parms,k))
  dBkt= np.dot(Q[0],Bkt)+Q[1]
  dV  = np.dot(V,dBkt)
  V   = np.dot(V,Bkt)
  for l in range(k-1,1,-1):
    Bl = mkB(n,M,l)
    V  = np.dot(V,Bl)
    dV = np.dot(dV,Bl)
  return [V[0,0],k*(k-1)/2*(1-V[0,0])/dV[0,0]]

def c_mk_fixed_K_iicrs(n,parms,K,kmax,st,t0,tmax,dt):
  """
  Computes a set of values of the IICR lambda_{k,sigma}^{(K],n,M}(t) for a fixed value of K and for 2<=k<=kmax<=K.
  The times values are equidistant, t0, t0+dt, t0+2*dt, ... in order to use the semigroup property of the
  semi-groups exp(t*Q(k)): for each value of k, only two matrix exponentials are computed.
  """
  lt = np.arange(t0,tmax+dt,dt)
  llpt= [lt]
  lB = [0,0]+[c_mkB(n,parms,k) for k in range(2,K+1)]
  V0 = np.zeros((1,len(c_parts(K,n))))
  V0[0,st]=1
  for k in range(2,kmax+1):
    lpt=[]
    Ik=np.identity(len(c_parts(k,n)))
    V=copy.copy(V0)
    for l in range(K,k,-1): V = np.dot(V,lB[l])
    W = np.identity(1)
    for l in range(2,k): W = np.dot(lB[l],W)
    Q = c_mkQ(n,parms,k)
    Bkt = np.dot(Ik-linalg.expm(t0*Q[0]),c_mkB(n,parms,k))
    dBkt= np.dot(Q[0],Bkt)+Q[1]
    a = (np.dot(np.dot(V,Bkt),W))[0,0]
    b = (np.dot(np.dot(V,dBkt),W))[0,0]
    lpt = [k*(k-1)/2*(1.0-a)/b]
    eQdt = linalg.expm(dt*Q[0])
    Bkdt = np.dot(Ik-eQdt,lB[k])
    t = t0
    while t<tmax:
      t +=dt
      Bkt = np.dot(eQdt,Bkt)+Bkdt
      dBkt = np.dot(Q[0],Bkt)+Q[1]
      a = (np.dot(np.dot(V,Bkt),W))[0,0]
      b = (np.dot(np.dot(V,dBkt),W))[0,0]
      lpt.append(k*(k-1)/2*(1.0-a)/b)
    llpt.append(copy.copy(lpt))  
  return llpt

def c_mk_fixed_k_iicrs(n,parms,Kmax,k,st,t0,tmax,dt):
  """
  Computes a set of values of the IICR lambda_{k,sigma}^{(K],n,M}(t) for a fixed value of k and for k<=K<=Kmax.
  The times values are equidistant, t0, t0+dt, t0+2*dt, ... in order to use the semigroup property of the
  semi-groups exp(t*Q(k)): here only two matrix exponentials are computed.
  """
  lt = np.arange(t0,tmax+dt,dt)
  res = np.zeros( (len(lt), Kmax-k+2))
  res[:,0]=lt
  lB = [0,0]+[c_mkB(n,parms,l) for l in range(2,Kmax+1)]
  W = np.identity(1)
  for l in range(2,k): W = np.dot(lB[l],W)
  Ik=np.identity(len(c_parts(k,n))) 
  Vleft = [] 
  for K in range(k,Kmax+1):
    V = np.zeros((1,len(c_parts(K,n))))
    V[0,st]=1
    for l in range(K,k,-1): V=np.dot(V,lB[l])
    Vleft.append(copy.copy(V))
  Q = c_mkQ(n,parms,k)
  Bkt = np.dot(Ik-linalg.expm(t0*Q[0]),c_mkB(n,parms,k))
  dBkt= np.dot(Q[0],Bkt)+Q[1]
  for K in range(k,Kmax+1):
    V = Vleft[K-k]
    a = (np.dot( np.dot( V  ,Bkt),W))[0]
    b = (np.dot( np.dot( V ,dBkt),W))[0]
    res[0,K-k+1]=k*(k-1)/2*(1.0-a)/b   
  eQdt = linalg.expm(dt*Q[0])
  Bkdt = np.dot(Ik-eQdt,lB[k])
  t = t0
  r=0
  while t<tmax:
    t +=dt
    r+=1
    Bkt  = np.dot(eQdt,Bkt)+Bkdt
    dBkt = np.dot(Q[0],Bkt)+Q[1]
    for K in range(k,Kmax+1):
      V = Vleft[K-k]
      a = (np.dot( np.dot( V  ,Bkt),W))[0]
      b = (np.dot( np.dot( V ,dBkt),W))[0]
      res[r,K-k+1] = k*(k-1)/2*(1.0-a)/b 
  return res
