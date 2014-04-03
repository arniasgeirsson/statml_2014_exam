import modules.common as com
import numpy as np
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import modules.config as con

# Make sure that the data is normalized
toNormalize = True
# Parse the input data
_, train_X, train_T, M = com.parseData("data/GTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)

# Create the same PCA as used in question 4
pca = decomposition.PCA(n_components=2)
pca.fit(train_X,train_T)
X = pca.transform(train_X)

k = 2
# Using n_jobs with anything but 1 does not work for me here on any of my
# machines
est = KMeans(n_clusters=k,precompute_distances=True,n_init=50,
                max_iter=100,n_jobs=1,verbose=con.verbose)
est.fit(train_X)

# Get the two 10-dimensional cluster center points
ccp10 = est.cluster_centers_
# Sort the two cluster points for convenience when printing
if ccp10[0,0] > ccp10[1,0]:
    ccp10 = np.array((ccp10[1], ccp10[0]))
print "The two 10-dimensional cluster center points:\n\t"\
        ,ccp10[0],"\n\t",ccp10[1]
# Project the cluster points to the two first principal components
ccp = pca.transform(ccp10)
# Sorth the two cluster points for convenience when printing
if ccp[0,0] > ccp[1,0]:
    ccp = np.array((ccp[1], ccp[0]))
print "The two 2-dimensional cluster center points:\n\t",ccp[0],"\n\t",ccp[1]

# Plot the same scatter point as in the question 4, although add the found
# centerpoints
scatterpoints = plt.figure('Scatterpoints & the two center points')
plt.set_cmap(cm.Spectral)
plt.scatter(X[:,0],X[:,1], marker='p', c=est.labels_)
plt.scatter(ccp[:,0],ccp[:,1], marker='v', c='r',s=60)

# Show and/or save the figure
if con.showfigs:
    scatterpoints.show()
if con.savefigs:
    scatterpoints.savefig(con.fig2_5_1_scPath)
