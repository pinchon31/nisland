__all__ = ['parts','canForm','c_parts','c_canForm']
import copy

def parts(k,n):
  """
  Partitions of k with at most n parts.
  This code is available on Internet but generally returns a generator.
  """
  if k == 1:
    return [[1]]
  else:
    res=[]
    lp = parts(k-1,n)
    for p in lp: 
      if len(p)==1 or (len(p) > 1 and p[-2] > p[-1]):
        p[-1] += 1
        res.append(copy.copy(p))
        p[-1] -=1
      if len(p)<n:
        p.append(1)
        res.append(copy.copy(p))
  return res

def canForm(p):
  """
  Canonical form of a partition set.
  """
  p[:]=[x for x in p if x!=0]
  p.sort()
  p.reverse()
  return p

def c_parts(k,n):
  """
  Continental_island partitions of an integer in a continent_island model:
  1 continent and n-1 islands (n>=2)
  If p is a partition, p[1] genes in the continent and p[2:-1] is the islands
  """
  res=[[k]]
  for m in range(k-1,-1,-1):
    lp=parts(k-m,n-1)
    for p in lp:
      p.insert(0,m)
      res.append(p)
  return res

def c_canForm(p):
  """
  Canonical form of a c_partition
  """
  np=canForm(p[1:])
  np.insert(0,p[0])
  return np
