import numpy as np

def grad_desc_armijo(X,y):
  
  m,p = X.shape
  beta = np.random.uniform(size = p)
  tau = 0.5
  gamma = 0.1
  
  alfa = tau
  
  p = grad_F(X,y,beta)
  
  while 
  
  return X