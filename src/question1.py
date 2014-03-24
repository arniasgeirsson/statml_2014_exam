import modules.common as com
from sklearn import linear_model
import numpy as np

# Read the given train and test data sets
_, i_train, t_train = com.parseData("data/SSFRTrain2014.dt")
_, i_test,  t_test = com.parseData("data/SSFRTest2014.dt")

# http://scikit-learn.org/0.11/auto_examples/linear_model/plot_ols.html
# http://glowingpython.blogspot.dk/2012/03/linear-regression-with-numpy.html

# Create the linear model and train it with the given training data
model = linear_model.LinearRegression()#normalize=True)
model.fit(i_train, t_train)

# Create the linear Phi matrix based on some input data set x
def linearPhi(X):
    return X

# Create the regression weight vector based on some input data set x
# and target vector t and a Phi function
def w(X,T,PHI):
    Phi = PHI(X)
    return np.linalg.inv(Phi.T*Phi)*Phi.T*T

w_0 = model.predict(np.array([0,0,0,0]))
w_1 = model.predict(np.array([1,0,0,0])) - w_0
w_2 = model.predict(np.array([0,1,0,0])) - w_0
w_3 = model.predict(np.array([0,0,1,0])) - w_0
w_4 = model.predict(np.array([0,0,0,1])) - w_0

print "w_0 = ", w_0
print "w_1 = ", w_1
print "w_2 = ", w_2
print "w_3 = ", w_3
print "w_4 = ", w_4

print "Mean square error of the training data set: " \
        , com.MSE(model.predict(i_train), t_train)
print "Mean square error of the test data set: " \
        , com.MSE(model.predict(i_test), t_test)

# print "Weight vector = ", w(i_train,t_train,linearPhi)
