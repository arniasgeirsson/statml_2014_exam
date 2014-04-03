import modules.common as com
import modules.config as con
from sklearn import grid_search
from sklearn.ensemble import RandomForestRegressor

toNormalize = True
# Read the given train and test data sets
d, train_X, train_T, M = com.parseData("data/SSFRTrain2014.dt",
                                        normalizeX=toNormalize)
_, test_X,  test_T,  _ = com.parseData("data/SSFRTest2014.dt",
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

parameters = {'n_estimators':[10,100,250], 'max_features':[int(M/3),M],
             'max_depth':[None], 'min_samples_split':[2,8],
             'min_samples_leaf':[3,5], 'criterion':['mse']}

if not con.recompute:
    parameters = {'n_estimators':[100], 'max_features':[int(M/3)],
             'max_depth':[None], 'min_samples_split':[2],
             'min_samples_leaf':[3], 'criterion':['mse']}

# Use gridsearch and cross validation to find the best hyperparameters
# and train the final model with those.
model = RandomForestRegressor(n_jobs=con.n_jobs)
model = grid_search.GridSearchCV(model, parameters, cv = con.cv,
                                    n_jobs = con.n_jobs, verbose=con.verbose)
model.fit(train_X,train_T)

# Print all the hyperparameters and the best ones
print "All hyperparameters:\n\t", parameters
print "Best hyperparameters:\n\t", model.best_params_

print "Mean square error of the training data set:" \
        , com.MSE(model.predict(train_X),train_T)
print "Mean square error of the test data set:\t\t" \
        , com.MSE(model.predict(test_X),test_T)
