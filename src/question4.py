import modules.common as com
import modules.config as con
import numpy as np
import math
from sklearn import decomposition
import matplotlib.pyplot as plt

# Make sure that the data is normalized
toNormalize = True
# Parse the input data
_, train_X, train_T, _ = com.parseData("data/GTrain2014.dt", delimiter=",",
                                        normalizeX=toNormalize)

# Create a PCA object and fit it with the input data
pca = decomposition.PCA()
pca.fit(train_X,train_T)
# Transform the input data to the generated PCA
X = pca.transform(train_X)

# Show the eigenspectrum
eigenvalues = pca.explained_variance_
print "The eigenvalues:\n\t", eigenvalues

# Show the importance of the number of principal components
ratio = pca.explained_variance_ratio_
c_ratio = com.cum_sum(ratio)
print "The eigenvalue importance:\n\t", ratio
print "The eigenvalue importance cummulative:\n\t", c_ratio

# Combined eigenspectrum and importance plot
comb, esAx = plt.subplots()
esAx.grid(True)
esAx.set_xlabel("eigenvector$_i$", fontsize=18)
esAx.set_title("Combined eigenspectrum and cum. variance ratio", fontsize=20)

# Set up the eigenspectrum plot
esAx.plot(range(1,len(eigenvalues)+1),eigenvalues, color='green',
                marker='D', label="eigenspectrum")
esAx.set_ylabel("eigenvalue", fontsize=18, color='green')

# Set up the explained variance ratio plot
imAx = esAx.twinx()
imAx.plot(range(1,len(ratio)+1),c_ratio, color='purple', marker='o')
imAx.set_ylabel("cummulative eigenvalue ratio (%)", fontsize=18,
                color='purple')

# Show and/or save the figure
if con.showfigs:
    comb.show()
if con.savefigs:
    comb.savefig(con.fig2_4_1_esPath)

# Show a scatter plot of the data projected to the first two principal
# components along with their respected eigenvector
scatterpoints, axes = plt.subplots()
axes.set_title('First 2 principal components scatter point and eigenvectors')

# The scatter plot
axes.scatter(X[:,0],X[:,1])

# The eigenvectors
ev, evv = np.linalg.eig(com.var(X))
m = com.mean(X)
i = 0;
for point in evv:
    evi = math.sqrt(ev[i])
    axes.arrow(m[0], m[1], point[0,0]*evi , point[0,1]*evi,
                zorder = 2, color='red')
    i = i+1

# The mean
axes.plot(m[0],m[1], "or")

# Show and/or save the figure
if con.showfigs:
    scatterpoints.show()
if con.savefigs:
    scatterpoints.savefig(con.fig2_4_2_scPath)
