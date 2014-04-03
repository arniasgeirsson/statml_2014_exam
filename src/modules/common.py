import numpy as np
import math

# Cummulatively sum a list of elements into a list with the sums
def cum_sum(xs):
    ln, y = len(xs), 0
    l = np.zeros(ln)
    for i in range(ln):
        y += xs[i]
        l[i] = y
    return l

# Extract all rows from X where the last coloum in X are in T
def extractClass(X,T):
    return np.array([x for x in X if x[-1] in T])

# Split a matrix into the first n-1 coloums and the last coloum
def splitdata(D):
	return D[:,0:len(D[0])-1], D[:,-1]

# Parse a data file and extract information from it
def parseData(filename, delimiter=" ",target_class=None,normalizeX=False,
				normY=None,n_components=None):
	# The whole data set
	data = np.loadtxt(filename,delimiter=delimiter)
	if target_class != None:
		data = extractClass(data,target_class)
	# The dataset without the target values
	inp, target = splitdata(data)
	if normalizeX:
		inp = normalize(inp,normY)
	if n_components != None and n_components > 0:
		inp = inp[:,:n_components]
	M = len(inp[0])
	return data,inp,target,M

# Normalize a given dataset x based on a given dataset y
def norm(x,y):
  	return (x - np.mean(y)) / np.std(y)

def normalize(x,y=None):
	if y is None:
		return norm(x,x)
	else:
		return norm(x,y)

# Calculate the mean-squared error
def MSE(x,t):
	return np.mean((t - x) ** 2)

# Calculate the root mean-squared error
def RMSE(x,t):
	return math.sqrt(MSE(x,t))

# Computes the mean of each axis of a input list of points
def mean(x):
    n = np.array(x).T
    return [sum(n[i])/len(n[i]) for i in range(0,len(n))]

# Computes the variance of the input x, where x is a list
# of points in a space
def var(x):
    m = mean(x)
    variance = np.matrix(np.zeros(shape=(len(x[0]),len(x[0]))))
    for point in x:
        tmp = (np.matrix(point) - np.matrix(m)).T
        variance = variance + tmp*tmp.T
    return variance / len(x)

# Creates a list of the mean of each colum-separated feature in a given dataset
def meanFeature(x):
	ms = np.zeros(len(x[0]))
	for i in range(len(x[0])):
		ms[i] = np.mean(x[:,i])
	return ms

# Creates a list of the variance of each colum-separated feature in a given dataset
def varFeature(x):
	ms = np.zeros(len(x[0]))
	for i in range(len(x[0])):
		ms[i] = np.var(x[:,i])
	return ms

# Calculates the accuracy of some prediction x with a given set of target values t
def accuracy(x,t):
	v = 0.0
	for i in range(len(x)):
		if x[i] == t[i]:
			v = v + 1.0
	return v / len(x)*100.0

# f assumes that there is atleast one x in X that has another label than the
# given x
def f(x,t,X,T):
    d = float("inf")
    for i in range(len(T)):
        if t != T[i]:
            d_ = np.linalg.norm(x-X[i])
            if d_ < d:
                d = d_
    return d

# Computes the jaakkola sigma based on some data set X and target values T
def jaakkolaSigma(X,T):
    a = np.zeros(len(T))
    for i in range(len(T)):
        a[i] = f(X[i],T[i],X,T)
    return np.median(a)

# Computes the jaakkola gamma based on some data set X and target values T
def jaakkolaGamma(X,T,sigma=None):
    if sigma is None:
        sigma = jaakkolaSigma(X,T)
    return 0.5/(sigma**2)
