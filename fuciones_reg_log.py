import numpy as np


def sigm(X):
  return 1/(1+np.exp(-X))


def LVN(X,y,beta):
  m = sigm(X@beta) 
  return -(y@np.log(m) + (1-y)@np.log(1-m))


def grad_F(X,y,beta):
  m = sigm(X@beta) 
  return X.T@(m-y)
  

def hess_F(X,y,beta): 
  m = sigm(X@beta) 
  S = np.outer(m.T,(1-m))
  
  return (X.T@S)@X


#Metodo de Armijo
def grad_desc_armijo(X,y):
  
  #Definimos los parámetros
  m,p = X.shape
  beta = np.zeros(p)# np.random.uniform(size = p)
  tau = 0.5
  gamma = 0.1
  epsilon = 10**-8 
  alfa = 0.00001 
  niter = 0
  
  #Estos son los valores iniciales
  p = -grad_F(X,y,beta)
  leftf = LVN(X,y,beta + alfa*p)
  rightf = LVN(X,y,beta) + gamma*alfa*(grad_F(X,y,beta)@p)
  
  #Iteraciones hasta que se compla la condición
  while leftf + epsilon < rightf:
    niter += 1
    alfa = alfa*tau
    beta = beta + alfa*p
    p = -grad_F(X,y,beta)
    leftf = LVN(X,y,beta + alfa*p)
    rightf = LVN(X,y,beta) + gamma*alfa*(grad_F(X,y,beta)@p)
    print(LVN(X,y,beta))
  
  return beta, niter 


#Funcion de prediccion
def pred(x, betahat):
  return((sigm(x@betahat)>0.5)*1)
  
  
# Metodo de Newton
def grad_desc_newton(X,y):
  
  #Definimos los parámetros
  m,p = X.shape
  beta = np.random.uniform(size = p)
  tau = 0.5
  gamma = 0.1
  epsilon = 10**-8 
  alfa = 0.000000000000001 
  niter = 0
  
  #Estos son los valores iniciales
  p = -np.linalg.inv(hess_F(X,y,beta))@grad_F(X,y,beta)
  leftf = LVN(X,y,beta + alfa*p)
  rightf = LVN(X,y,beta) + gamma*alfa*(grad_F(X,y,beta)@p)
  
  #Iteraciones hasta que se compla la condición
  while leftf + epsilon < rightf:
    niter += 1
    alfa = alfa*tau
    beta = beta + alfa*p
    p = -grad_F(X,y,beta)
    leftf = LVN(X,y,beta + alfa*p)
    rightf = LVN(X,y,beta) + gamma*alfa*(grad_F(X,y,beta)@p)
    print(LVN(X,y,beta))
  
  return beta, niter

