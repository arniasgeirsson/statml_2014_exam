import modules.common as com
import modules.config as con
from sklearn import svm, grid_search

# Normalize the data as SVMs perform better and faster when the data is
# normalized
toNormalize = True
# Read the given train and test data sets
d, train_X, train_T, _ = com.parseData("data/SGTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)
_, test_X,  test_T,  _ = com.parseData("data/SGTest2014.dt", delimiter=",",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

# As the jaakkola values are static if given the same input values, it is
# therefore unneeded to recompute them with every run, and I have precomputed
# it and hardcoded the values
jaakkolaSigma = 1.81188376031
jaakkola = 0.1523033091

if con.recompute:
    jaakkola = com.jaakkolaGamma(train_X,train_T)

print "Jaakkola sigma: ", jaakkolaSigma
print "Jaakkola gamma: ", jaakkola

# Define the different values for C and gamma
b = 10

Cs = [b**-2,b**-1,1,b,b**2,b**3]
Gs = [jaakkola*b**-3,jaakkola*b**-2,jaakkola*b**-1,jaakkola*b**0,\
        jaakkola*b**1,jaakkola*b**2,jaakkola*b**3]

print "All C values used in gridsearch:\n\t", Cs
print "All gamma values used in gridsearch:\n\t", Gs

parameters = {'kernel':['rbf'], 'C':Cs, 'gamma':Gs}

if not con.recompute:
    parameters = {'kernel':['rbf'], 'C':[1000], 'gamma':[0.01523033091]}

# Use grid search and 5-fold cross validation to find the best hyperparameters
# and fit the svm with the found values
svc = svm.SVC()
clf = grid_search.GridSearchCV(svc, parameters, cv = con.cv,
                                n_jobs = con.n_jobs, verbose=con.verbose)
clf.fit(train_X,train_T)

# Print the most optimized hyperparameters
print "Best C value:\t\t", clf.best_params_['C']
print "Best gamma value:\t", clf.best_params_['gamma']

# Calculate the mean-squared error and the accuracy of the classifier
print "Mean square error of the training data set:\t" \
        , com.MSE(clf.predict(train_X),train_T)
print "Mean square error of the test data set:\t\t" \
        , com.MSE(clf.predict(test_X),test_T)

print "The accuracy of the training data set:\t\t" \
        , com.accuracy(clf.predict(train_X),train_T)
print "The accuracy of the test data set:\t\t\t" \
        , com.accuracy(clf.predict(test_X),test_T)
