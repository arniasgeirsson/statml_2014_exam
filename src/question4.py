import modules.common as com
from sklearn import decomposition
import matplotlib.pyplot as plt
import modules.constants as con
import numpy as np
import math

# PCA links
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
# http://en.wikipedia.org/wiki/Principal_component_analysis
# http://scipy-lectures.github.io/advanced/scikit-learn/
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# Eigen[values,vectors] links
# https://www.google.dk/search?q=principle+component+analysis+eigenspectrum&ie=utf-8&oe=utf-8&rls=org.mozilla:da:official&client=firefox-beta&channel=fflb&gws_rd=cr&ei=IAg0U4LIOcuc4wSaloDoCg#channel=fflb&q=eigenspectrum&rls=org.mozilla:da:official
# http://www.sedfitting.org/SED08/Paper_vs1.0_online/walcher_mssu9.html
# http://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html

# TODO, to normalize?
toNormalize = True
# Parse the input data
_, i_train, t_train, _ = com.parseData("data/GTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)
i1 = 6
i2 = 7
n2 = i_train # np.c_[i_train[:,i1],i_train[:,i2]]

# Create a PCA object and fit it with the input data
# TODO does whiten change anything?
pca = decomposition.PCA()
pca.fit(n2,t_train)
# Transform the input data to the generated PCA (TODO should not change the input data, as it is the same used to fit the PCA, or what?)
X = pca.transform(n2)

print "all variances: ", com.varFeature(i_train)
print "all X variances: ", com.varFeature(X)


# Show the eigenspectrum
eigenvalues = pca.explained_variance_
print "The eigenvalues:\n\t", eigenvalues
eigenspectrum = plt.figure('eigenspectrum')
plt.plot(range(1,len(eigenvalues)+1),eigenvalues)
if con.showfigs:
    eigenspectrum.show()
if con.savefigs:
    eigenspectrum.savefig(con.fig2_4_1_esPath)

# Show the importance of the number of principal components
featureMeaning = plt.figure('featureMeaning')
ratio = pca.explained_variance_ratio_
print "The eigenvalue importance:\n\t", ratio
print "The eigenvalue importance cummulative:\n\t" \
        , com.cum_sum(ratio)
plt.plot(range(1,len(ratio)+1),com.cum_sum(ratio))
if con.showfigs:
    featureMeaning.show()
if con.savefigs:
    featureMeaning.savefig(con.fig2_4_2_eiPath)

# Combined eigenspectrum and importance plot
comb, esAx = plt.subplots()
esAx.plot(range(1,len(eigenvalues)+1),eigenvalues, color='blue', marker='D')

imAx = esAx.twinx()
imAx.plot(range(1,len(ratio)+1),com.cum_sum(ratio), color='red', marker='o')

comb.show()

# Show a scatter plot of the data projected to the first two principal components
scatterpoints = plt.figure('scatterpoints')
plt.plot(n2[:,0],n2[:,1], 'xb')
v = com.var(n2)
ev, evv = np.linalg.eig(v) # (np.squeeze(np.asarray(smlc)))
# print ev
# print evv
m = com.mean(n2)
i = 0;
for point in evv:
    plt.arrow(m[0], m[1], point[0,0]*math.sqrt(ev[i])
                        , point[0,1]*math.sqrt(ev[i]), zorder = 2)
    i = i+1
plt.plot(m[0],m[1], "oy")


plt.plot(X[:,0],X[:,1], 'xr')
v = com.var(X)
ev, evv = np.linalg.eig(v) # (np.squeeze(np.asarray(smlc)))
# print ev
# print evv
m = com.mean(X)
i = 0;
for point in evv:
    plt.arrow(m[0], m[1], point[0,0]*math.sqrt(ev[i])
                        , point[0,1]*math.sqrt(ev[i]), zorder = 2)
    i = i+1

plt.plot(m[0],m[1], "og")
if con.showfigs:
    scatterpoints.show()
if con.savefigs:
    scatterpoints.savefig(con.fig2_4_3_scPath)


# from mpl_toolkits.mplot3d.axes3d import Axes3D
# scatter3d = plt.figure('3d scatter plot')
# ax = Axes3D(scatter3d)
# p = ax.plot_surface(X[:,0],X[:,1],X[:,2])



