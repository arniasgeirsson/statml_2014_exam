import modules.common as com
from sklearn import svm, grid_search
import numpy as np

# TODO, to normalize?
toNormalize = False
# Read the given train and test data sets
d, i_train, t_train, _ = com.parseData("data/SGTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)
_, i_test,  t_test,  _ = com.parseData("data/SGTest2014.dt", delimiter=",",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

# f assumes that there is atleast one x in X that has another label than the given x
def f(x,t,X,T):
    d = float("inf")
    for i in range(len(T)):
        if t != T[i]:
            d_ = np.linalg.norm(x-X[i])
            if d_ < d:
                d = d_
    return d

def jaakkolaSigma(X,T):
    a = np.zeros(len(T))
    for i in range(len(T)):
        a[i] = f(X[i],T[i],X,T)
    return np.median(a)

def jaakkolaGamma(X,T):
    sigma = jaakkolaSigma(X,T)
    return 0.5/(sigma**2)


#jaakkola = jaakkolaSigma(i_train,t_train)
# -> 1.81188376031
#jaakkola = jaakkolaGamma(i_train,t_train)
# -> 0.1523033091

# As the value is static given the same input values, and therefore unneeded to recompute with every run, I have precomputed it and hardcoded the value
jaakkolaSigma = 1.81188376031
jaakkola = 0.1523033091
print "Jaakkola sigma: ", jaakkolaSigma
print "Jaakkola gamma: ", jaakkola
#jaakkola = jaakkolaGamma(i_train,t_train)

# http://scikit-learn.org/stable/modules/grid_search.html
# http://scikit-learn.org/dev/auto_examples/grid_search_digits.html
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
# http://en.wikipedia.org/wiki/Hyperparameter_optimization

b = 10

Cs = [b**-2,b**-1,1,b,b**2,b**3]
Gs = [jaakkola*b**-3,jaakkola*b**-2,jaakkola*b**-1,jaakkola*b**0,jaakkola*b**1,jaakkola*b**2,jaakkola*b**3]

# Cs = [b]
# Gs = [jaakkola]

print "All C values used in gridsearch:\n\t", Cs
print "All gamma values used in gridsearch:\n\t", Gs

parameters = {'kernel':['rbf'], 'C':Cs, 'gamma':Gs}

svc = svm.SVC()
clf = grid_search.GridSearchCV(svc, parameters, cv = 5, n_jobs = 2)
clf.fit(i_train,t_train)

print "Best C value:\t\t", clf.best_params_['C']
print "Best gamma value:\t", clf.best_params_['gamma']

print "Mean square error of the training data set:\t" \
        , com.MSE(clf.predict(i_train),t_train)
print "Mean square error of the test data set:\t\t" \
        , com.MSE(clf.predict(i_test),t_test)

print "The accuracy of the training data set:\t\t" \
        , com.accuracy(clf.predict(i_train),t_train)
print "The accuracy of the test data set:\t\t\t" \
        , com.accuracy(clf.predict(i_test),t_test)


#print clf.get_params()
#print clf.grid_scores_
#print clf.best_score_
#print clf.best_estimator_
