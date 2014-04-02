import modules.common as com
import numpy as np
import sklearn.neighbors as knn
from sklearn.lda import LDA
from sklearn import grid_search, svm
from sklearn.qda import QDA
from sklearn import decomposition

# Linear classification methods
# k-nearest neighbors
# LDA -> or is it non-linear?
# Perceptron
# Quadratic classifier

# Non-linear classification methods
# Random forest -> or is it linear?
# SVM


# Notes:
# - 61 dimensions are a lot, use PCA to reduce the dimensions?
# - - is 61 that much when there are 25 classes
# - Considered there are so many features and classes, then the number of training and test data are not that rich, might affect the performance and creditability of the developed classifiers, how so?

# TODO, to normalize?
toNormalize = False
# Read the given train and test data sets
d, i_train, t_train, M = com.parseData("data/VSTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)
_, i_test,  t_test,  _ = com.parseData("data/VSTest2014.dt", delimiter=",",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])
pca = decomposition.PCA(n_components=50)
pca.fit(i_train,t_train)
i_train = pca.transform(i_train)
i_test = pca.transform(i_test)

# ---------- KNN ----------
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier

# TODO what other hyperparameters can be tempared with?
# http://scikit-learn.org/0.11/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py

parameters = { 'n_neighbors' : [1,5,10,20], 'weights':['uniform','distance']}
clf = knn.KNeighborsClassifier()
clf = grid_search.GridSearchCV(clf, parameters, cv = 5, n_jobs = 1)
clf.fit(i_train,t_train)

print "KNN: best hyperparameters: ", clf.best_params_
print "KNN: train MSE: ", com.MSE(clf.predict(i_train),t_train)
print "KNN: test MSE: ", com.MSE(clf.predict(i_test),t_test)
print "KNN: train accuracy: ", com.accuracy(clf.predict(i_train),t_train)
print "KNN: test accuracy: ", com.accuracy(clf.predict(i_test),t_test)


# ---------- LDA ----------
# http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html

# TODO the priors parameter?
# TODO play around with the n_components parameter
# Why no difference ?
parameters = { 'n_components' : np.arange(M)+1}
clf = LDA()
clf = grid_search.GridSearchCV(clf, parameters, cv = 5, n_jobs = 1)
clf.fit(i_train,t_train)

print "LDA: best hyperparameters: ", clf.best_params_
print "LDA: train MSE: ", com.MSE(clf.predict(i_train),t_train)
print "LDA: test MSE: ", com.MSE(clf.predict(i_test),t_test)
print "LDA: train accuracy: ", com.accuracy(clf.predict(i_train),t_train)
print "LDA: test accuracy: ", com.accuracy(clf.predict(i_test),t_test)

# ---------- QDA ----------
# http://scikit-learn.org/stable/modules/generated/sklearn.qda.QDA.html#sklearn.qda.QDA

# Does not work for some reason

# TODO the priors parameter?
# TODO play around with the reg_param parameter
# parameters = {}
# clf = QDA(reg_param=1.0)
# # clf = grid_search.GridSearchCV(clf, parameters, cv = 5, n_jobs = 2)
# clf.fit(i_train,t_train)

# print "QDA: train MSE: ", com.MSE(clf.predict(i_train),t_train)
# print "QDA: test MSE: ", com.MSE(clf.predict(i_test),t_test)
# print "QDA: train accuracy: ", com.accuracy(clf.predict(i_train),t_train)
# print "QDA: test accuracy: ", com.accuracy(clf.predict(i_test),t_test)

# ---------- SVM ----------

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
# jaakkolaSigma = 1.81188376031
jaakkola = 45.8805908327
# jaakkola = jaakkolaGamma(i_train,t_train)
# print "Jaakkola sigma: ", jaakkolaSigma
print "Jaakkola gamma: ", jaakkola

# http://scikit-learn.org/stable/modules/grid_search.html
# http://scikit-learn.org/dev/auto_examples/grid_search_digits.html
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
# http://en.wikipedia.org/wiki/Hyperparameter_optimization

b = 10

Cs = [b**-2,b**-1,1,b,b**2,b**3]
Gs = [jaakkola*b**-3,jaakkola*b**-2,jaakkola*b**-1,jaakkola*b**0,jaakkola*b**1,jaakkola*b**2,jaakkola*b**3]

Cs = [b]
Gs = [jaakkola]

print "All C values used in gridsearch:\n\t", Cs
print "All gamma values used in gridsearch:\n\t", Gs

parameters = {'kernel':['rbf'], 'C':Cs, 'gamma':Gs}

svc = svm.SVC()
clf = grid_search.GridSearchCV(svc, parameters, cv = 5, n_jobs = 1)
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



# ---------- RandomForest (RDF) ----------
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# http://scikit-learn.org/stable/modules/ensemble.html
# http://blog.yhathq.com/posts/random-forests-in-python.html

from sklearn.ensemble import RandomForestClassifier

parameters = { 'n_estimators' : [1,10,20,100,250,500,1000]}
clf = RandomForestClassifier()
clf = grid_search.GridSearchCV(clf, parameters, cv = 5, n_jobs = 2, verbose=1)
clf.fit(i_train,t_train)

print "RDF: best hyperparameters: ", clf.best_params_
print "RDF: train MSE: ", com.MSE(clf.predict(i_train),t_train)
print "RDF: test MSE: ", com.MSE(clf.predict(i_test),t_test)
print "RDF: train accuracy: ", com.accuracy(clf.predict(i_train),t_train)
print "RDF: test accuracy: ", com.accuracy(clf.predict(i_test),t_test)
