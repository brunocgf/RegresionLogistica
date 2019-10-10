import numpy as np

def hess_F(X,y,beta):
  
  m = 1/(1+np.exp(np.dot(X,beta)))
  S = np.dot(m.T,(1-m))
  
  return(np.dot(np.dot(X.T,S),X))