from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np


X_entrena, y_entrena = datos()
X_prueba, y_prueba = datos(modo='prueba')

#X[:,0] = 1

scaler = preprocessing.MinMaxScaler()

Xst_entrena = scaler.fit_transform(X_entrena)
Xst_prueba = scaler.transform(X_prueba)


betahat_armijo, i = grad_desc_armijo(Xst_entrena,y_entrena)

yhat = pred(Xst_prueba, betahat_armijo)

np.mean(yhat-y_prueba)

#lr = LogisticRegression()
#lr.fit(Xst,y)