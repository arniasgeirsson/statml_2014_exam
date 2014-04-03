import modules.common as com
import modules.config as con
from sklearn.lda import LDA
from sklearn import grid_search

toNormalize = True
# Read the given train and test data sets
d, train_X, train_T, _ = com.parseData("data/VSTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)
_, test_X,  test_T,  _ = com.parseData("data/VSTest2014.dt", delimiter=",",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

print "----- LDA -----"

# Create and fit the LDA model
clf = LDA()
clf.fit(train_X,train_T)

# Calculate the MSE and accuracy of the LDA model
print "Mean square error of the training data set:\t",\
        com.MSE(clf.predict(train_X),train_T)
print "Mean square error of the test data set:\t\t",\
        com.MSE(clf.predict(test_X),test_T)
print "The accuracy of the training data set:\t\t",\
        com.accuracy(clf.predict(train_X),train_T)
print "The accuracy of the test data set:\t\t\t",\
        com.accuracy(clf.predict(test_X),test_T)


print "----- Random Forest -----"

from sklearn.ensemble import RandomForestClassifier

# Set the hyperparameters
parameters = { 'n_estimators':[10,100,250,500], 'max_features':['sqrt'],
                'max_depth':[None], 'criterion':['gini','entropy'],
                'min_samples_split':[2,4,8], 'min_samples_leaf':[1,10]}

if not con.recompute:
    parameters = {'n_estimators':[100],'max_features':['sqrt'],
                    'max_depth':[None], 'criterion':['entropy'],
                    'min_samples_split':[4], 'min_samples_leaf':[1]}


# Use gridsearch and cross validation to find the optimized hyperparameters
clf = RandomForestClassifier(n_jobs=con.n_jobs)
clf = grid_search.GridSearchCV(clf, parameters, cv = con.cv,
                                n_jobs = con.n_jobs, verbose=con.verbose)
clf.fit(train_X,train_T)

# Print all the possible hyperparameters and the best ones
print "RDF: all hyperparameters:\n\t", parameters
print "RDF: best hyperparameters:\n\t", clf.best_params_

# Calculate the MSE and accuracy of the random forest model
print "Mean square error of the training data set:\t",\
        com.MSE(clf.predict(train_X),train_T)
print "Mean square error of the test data set:\t\t",\
        com.MSE(clf.predict(test_X),test_T)
print "The accuracy of the training data set:\t\t",\
        com.accuracy(clf.predict(train_X),train_T)
print "The accuracy of the test data set:\t\t\t",\
        com.accuracy(clf.predict(test_X),test_T)
