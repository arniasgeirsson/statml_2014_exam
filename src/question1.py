import modules.common as com
from sklearn import linear_model
import numpy as np

# TODO, to normalize?
toNormalize = False
# Read the given train and test data sets
d, i_train, t_train, M = com.parseData("data/SSFRTrain2014.dt",
                                        normalizeX=toNormalize)
_, i_test,  t_test,  _ = com.parseData("data/SSFRTest2014.dt",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

# http://scikit-learn.org/0.11/auto_examples/linear_model/plot_ols.html
# http://glowingpython.blogspot.dk/2012/03/linear-regression-with-numpy.html

# Create the linear model and train it with the given training data
model = linear_model.LinearRegression()#normalize=True)
model.fit(i_train, t_train)

# Create the linear Phi matrix based on some input data set x
def linearPhi(X):
    return np.c_[np.ones(len(X)),X]

# Create the regression weight vector based on some input data set x
# and target vector t and a Phi function
def w(X,T,PHI):
    Phi = PHI(X)
    # (Phi.T*Phi)^-1*Phi.T*T
    return np.dot(np.dot(np.linalg.inv(np.dot(Phi.T,Phi)),Phi.T),T)

# Computes the weight vector based on a linear regression model
# where M is the dimension of the data
def extract_w(model,M):
    w = model.predict(np.r_[[np.zeros(M)], np.diag(np.ones(M))])
    w[1:] = w[1:] - w[0]
    return w

# Extract the weight vector from the model
print "Extracted weight vector:\n\t", extract_w(model,M)
# Calculate the weight vector based on the theoreticly model
print "Calculated weight vector:\n\t", w(i_train,t_train,linearPhi)

print "Mean square error of the training data set: " \
        , com.MSE(model.predict(i_train), t_train)
print "Mean square error of the test data set: " \
        , com.MSE(model.predict(i_test), t_test)
