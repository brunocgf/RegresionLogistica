from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import datos

X,y = datos()
X_test, y_test = datos(modo='prueba')

X[:,0] = 1

Xsta = preprocessing.scale(X)
scaler = preprocessing.MinMaxScaler()
Xst = scaler.fit_transform(X)
Xst_test = scaler.transform(X_test)


beta, i = grad_desc_armijo(Xst,y)

y_est = sigm(Xst_test@beta)
y_est = sigm(Xst_test@beta0[0,:])


lr = LogisticRegression()
lr.fit(Xst,y)