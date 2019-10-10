import numpy as np

def grad_F(X,y,beta):
  
  m = 1/(1+np.exp(np.dot(X,beta)))
    
  return(np.dot(X.T, (m-y)))
  
  
  