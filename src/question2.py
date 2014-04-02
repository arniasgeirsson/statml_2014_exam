import modules.common as com
from sklearn.svm import SVR
import numpy as np

_, i_train, t_train, _ = com.parseData("data/SSFRTrain2014.dt")
_, i_test,  t_test,  _ = com.parseData("data/SSFRTest2014.dt")

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

# Use something other than a svm
# https://absalon.itslearning.com/ContentArea/ContentArea.aspx?LocationID=52903&LocationType=1

C = 20.0
epsilon = 0.01

model = SVR(kernel = 'rbf', C = C, epsilon = epsilon)
model.fit(i_train,t_train)

print "Mean square error of the training data set: " \
        , com.MSE(model.predict(i_train),t_train)
print "Mean square error of the test data set: " \
        , com.MSE(model.predict(i_test),t_test)
