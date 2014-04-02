import modules.common as com
import numpy as np
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import modules.constants as con

# TODO, to normalize?
toNormalize = True
# Parse the input data
_, i_train, t_train, M = com.parseData("data/GTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)

pca = decomposition.PCA(n_components=2)
pca.fit(i_train,t_train)
X = pca.transform(i_train)

# TODO it must be cross validated somehow as the cluster center points are not always the same, do n times and average on the cluster center points?

# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
n = 2
est = KMeans(n_clusters=n,precompute_distances=True,max_iter=20000)
est.fit(i_train)

# Get the two 10-dimensional cluster center points
ccp10 = est.cluster_centers_
if ccp10[0,0] < ccp10[1,0]:
    ccp10 = np.array((ccp10[1], ccp10[0]))
print "The two 10-dimensional cluster center points:\n\t",ccp10[0],"\n\t",ccp10[1]
# Project the cluster points to the two first principal components
ccp = pca.transform(ccp10)
print "The two 2-dimensional cluster center points:\n\t",ccp[0],"\n\t",ccp[1]

# Average the cluster points
# avg = np.zeros((n,M))
# a = 100
# for i in range(a):
#     _est = KMeans(n_clusters=n,precompute_distances=True,max_iter=20000)
#     _est.fit(i_train)
#     cps = _est.cluster_centers_
#     if cps[0,0] < cps[1,0]:
#         cps = np.array((cps[1], cps[0]))
#     avg = avg + cps
# avg = avg / a

# print "average: ", ccp10 - avg
# print "avg mse: ", com.MSE(ccp10,avg)

scatterpoints = plt.figure('Scatterpoints & the two center points')
# http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
plt.set_cmap(cm.Spectral)
plt.scatter(X[:,0],X[:,1], marker=',', c=est.labels_)
plt.scatter(ccp[:,0],ccp[:,1], marker='o', c=np.arange(len(ccp)))

if con.showfigs:
    scatterpoints.show()
if con.savefigs:
    scatterpoints.savefig(con.fig2_5_1_scPath)
