__all__ = ['coalesce','migrate','mkQ','mkB','mk_F_iicr','main_eigenvalue','mk_fixed_K_iicrs','mk_fixed_k_iicrs']
import copy
import numpy as np
from scipy import linalg
from partition import *


def coalesce(p):
  """
  Starting from a state p, produces a list of new states after coalescence of two genes. 
  Each new state is weigthed following the number of genes in the island where the coalescence occurs.
  """
  lwq = []
  for i in range(len(p)):
    if p[i]>1:
      s=copy.copy(p)
      s[i]-=1
      s.sort()
      s.reverse()
      wq = [s,p[i]*(p[i]-1)/2]
      lwq.append(copy.copy(wq))
  return lwq



def migrate(n,p):
  """
  Starting from a state p, produces a list of new states after migration of one gene. Each new state is weigthed following the
  number of genes in the island where the migration occurs.
  """
  lwq=[]
  for i in range(len(p)):
    r = copy.copy(p)
    r[i]-=1
    nb = p[i]
    for j in range(len(p)):
      if i!=j:
        s=copy.copy(r)
        s[j]+=1
        s=canForm(s)
        if s!=p:
          wq=[s,nb]
          lwq.append(wq)
    if r[i]>0 and len(r)<n:
      r.append(1)
      r=canForm(r)
      wq = [r,nb*(n-len(p))]
      lwq.append(wq)
  return lwq

def mkQ(n,M,k):
  """
  Components of the basic Q-matrix for the Tk coalescence time.
  """
  v = M/(2*(n-1))
  lp = parts(k,n)
  lp2 = parts(k-1,n)
  n1 = len(lp)
  n2 = len(lp2)
  Q = np.zeros((n1+n2,n1+n2))
  for i in range(n1):
    for wq in migrate(n,lp[i]):
      j=lp.index(wq[0])
      Q[i,j]+=v*wq[1]
    for wq in coalesce(lp[i]):
      j=lp2.index(wq[0])
      Q[i,n1+j]+=wq[1]
    Q[i,i]=0
    for j in range(n1+n2):
      if j!=i: Q[i,i]-=Q[i,j]
  res=[np.array(Q[0:n1,0:n1])]
  res.append(np.array(Q[0:n1,n1:(n1+n2)])) 
  return res
    
def main_eigenvalue(n,M,k):
  """
  The largest stricly eigenvalue of the first component of the basic Tk Q-matrix.
  """
  lmu = np.ndarray.tolist(np.linalg.eigvals(mkQ(n,M,k)[0]))
  lmu.sort()
  return lmu[-1]

def mkB(n,M,k):
  """
  Main component of the limit of exp(tQ) when t tends to infinity. 
  """
  Qk=mkQ(n,M,k)
  B=-np.linalg.solve(Qk[0],Qk[1])
  return B
  
def mk_F_iicr(n,M,K,k,st,t):
  """
  Computes the cumulative distribution function F(t)=P(T_{k,sigma}^{(K],n,M} <= t) and the corresponding IICR
  lambda_{k,sigma}^{(K],n,M}(t) = (1-F(t)/F'(t).
  """
  V = np.zeros((1,len(parts(K,n))))
  V[0,st]=1
  for l in range(K,k,-1): V = np.dot(V,mkB(n,M,l))
  Q=mkQ(n,M,k)
  Bkt = np.dot(np.identity(len(parts(k,n)))-linalg.expm(t*Q[0]),mkB(n,M,k))
  dBkt= np.dot(Q[0],Bkt)+Q[1]
  dV  = np.dot(V,dBkt)
  V   = np.dot(V,Bkt)
  for l in range(k-1,1,-1):
    Bl = mkB(n,M,l)
    V  = np.dot(V,Bl)
    dV = np.dot(dV,Bl)
  return [V[0,0],k*(k-1)/2*(1-V[0,0])/dV[0,0]]

def mk_fixed_K_iicrs(n,M,K,kmax,st,t0,tmax,dt):
  """
  Computes a set of values of the IICR lambda_{k,sigma}^{(K],n,M}(t) for a fixed value of K and for 2<=k<=kmax<=K.
  The times values are equidistant, t0, t0+dt, t0+2*dt, ... in order to use the semigroup property of the
  semi-groups exp(t*Q(k)): for each value of k, only two matrix exponentials are computed.
  """
  lt = np.arange(t0,tmax+dt,dt)
  llpt= [lt]
  lB = [0,0]+[mkB(n,M,k) for k in range(2,K+1)]
  V0 = np.zeros((1,len(parts(K,n))))
  V0[0,st]=1
  for k in range(2,kmax+1):
    lpt=[]
    Ik=np.identity(len(parts(k,n)))
    V=copy.copy(V0)
    for l in range(K,k,-1): V = np.dot(V,lB[l])
    W = np.identity(1)
    for l in range(2,k): W = np.dot(lB[l],W)
    Q = mkQ(n,M,k)
    Bkt = np.dot(Ik-linalg.expm(t0*Q[0]),mkB(n,M,k))
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

def mk_fixed_k_iicrs(n,M,Kmax,k,st,t0,tmax,dt):
  """
  Computes a set of values of the IICR lambda_{k,sigma}^{(K],n,M}(t) for a fixed value of k and for k<=K<=Kmax.
  The times values are equidistant, t0, t0+dt, t0+2*dt, ... in order to use the semigroup property of the
  semi-groups exp(t*Q(k)): here only two matrix exponentials are computed.
  """
  lt = np.arange(t0,tmax+dt,dt)
  res = np.zeros( (len(lt), Kmax-k+2))
  res[:,0]=lt
  lB = [0,0]+[mkB(n,M,l) for l in range(2,Kmax+1)]
  W = np.identity(1)
  for l in range(2,k): W = np.dot(lB[l],W)
  Ik=np.identity(len(parts(k,n))) 
  Vleft = [] 
  for K in range(k,Kmax+1):
    V = np.zeros((1,len(parts(K,n))))
    V[0,st]=1
    for l in range(K,k,-1): V=np.dot(V,lB[l])
    Vleft.append(copy.copy(V))
  Q = mkQ(n,M,k)
  Bkt = np.dot(Ik-linalg.expm(t0*Q[0]),mkB(n,M,k))
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


