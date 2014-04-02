import modules.common as com
import numpy as np

# Normalization does not effect anything but the values of the weights in the
# weight vector
toNormalize = False
# Read the given train and test data sets
d, train_X, train_T, M = com.parseData("data/SSFRTrain2014.dt",
                                        normalizeX=toNormalize)
_, test_X,  test_T,  _ = com.parseData("data/SSFRTest2014.dt",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

print "----- Linear regression using custom methods -----"

# Create the linear Phi matrix based on some input data set x
def linearPhi(X):
    return np.c_[np.ones(len(X)),X]

# Create the regression weight vector based on some input data set X
# and target vector T and a Phi function
def w(X,T,PHI):
    Phi = PHI(X)
    # (Phi.T*Phi)^-1*Phi.T*T
    return np.dot(np.dot(np.linalg.inv(np.dot(Phi.T,Phi)),Phi.T),T)

# Predict the outcome of a single point x
def predictOne(x,w):
    return np.dot(w,np.r_[1,x])

# Predict the outcome of all the points in X
def predict(X,w):
    return np.array([predictOne(x,w) for x in X])

# Calculate the weight vector w
w = w(train_X,train_T,linearPhi)

# Calculate the weight vector based on the theoreticly model
print "Calculated weight vector:\n\t", w

# Print the mean square error on the training and test data sets
print "Mean square error of the training data set:\t" \
        , com.MSE(predict(train_X,w), train_T)
print "Mean square error of the test data set:\t\t" \
        , com.MSE(predict(test_X,w), test_T)


print "----- Linear regression using Scikit -----"
# Note that scikit is just used to make sure that my implementation of
# linear regression is correct

from sklearn import linear_model

# Create the linear model and train it with the given training data
model = linear_model.LinearRegression(fit_intercept=True,normalize=False)
model.fit(train_X, train_T)

# Computes the weight vector based on a linear regression model
# where M is the dimension of the data
def extract_w(model,M):
    w = model.predict(np.r_[[np.zeros(M)], np.diag(np.ones(M))])
    w[1:] = w[1:] - w[0]
    return w

# Extract the weight vector from the model
print "Extracted weight vector:\n\t", extract_w(model,M)
# Print the mean square error on the training and test data sets
print "Mean square error of the training data set:\t" \
        , com.MSE(model.predict(train_X), train_T)
print "Mean square error of the test data set:\t\t" \
        , com.MSE(model.predict(test_X), test_T)
