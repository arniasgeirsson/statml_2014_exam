import numpy as np
import math
from sklearn import svm
import itertools

def cum_sum(xs):
    ln, y = len(xs), 0
    l = np.zeros(ln)
    for i in range(ln):
        y += xs[i]
        l[i] = y
    return l

def extractClass(X,T):
    return np.array([x for x in X if x[-1] in T])

def splitdata(D):
	return D[:,0:len(D[0])-1], D[:,-1]

def parseData(filename, delimiter=" ",target_class=None,normalizeX=False,
				normY=None):
	# The whole data set
	data = np.loadtxt(filename,delimiter=delimiter)
	if target_class != None:
		data = extractClass(data,target_class)
	# The dataset without the target values
	inp, target = splitdata(data)
	if normalizeX:
		inp = normalize(inp,normY)
	M = len(inp[0])
	return data,inp,target,M

# Assumes that x and z are numpy arrays
# gamma must be positive
def kernel(x,z,gamma):
	return math.exp(-gamma*np.linalg.norm(x-z)**2)

# Normalize a given dataset x based on a given dataset y
def norm(x,y):
  	return (x - np.mean(y)) / np.std(y)

def normalize(x,y=None):
	if y is None:
		return norm(x,x)
	else:
		return norm(x,y)

def MSE(x,t):
	return np.mean((t - x) ** 2)

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


def chunks(l,n):
    n = len(l)/n
    return [l[i:i+n] for i in range(0, len(l), n)]

def appends(ls,noti):
    l = []
    for i in range(len(ls)):
       if not i in noti:
           l.append(ls[i])
    return l

# Perform crossvalidation
def crossvalidateSVM(i,t,C,gamma,S = 5):
	avg = np.zeros(S)
	chnksI = np.array(chunks(i,S))
	chnksT = np.array(chunks(t,S))
	for i in range(S):
		chnkI =  np.array(chnksI[i])
		chnkT =  np.array(chnksT[i])
	  	restI = appends(chnksI,[i])
	  	restI = np.array(list(itertools.chain(*restI)))
	  	restT = appends(chnksT,[i])
	  	restT = np.array(list(itertools.chain(*restT)))

		clf = svm.SVC(C=C,gamma=gamma)
		clf.fit(restI,restT)
		avg[i] = accuracy(clf.predict(chnkI),chnkT)
	return sum(avg)/len(avg)

# Perform gridsearch on the given lists of C and gamma values
def gridSearchSVM(Cs, Gs, inp,target):
	return gridSearchPermutations(itertools.product(Cs,Gs),inp,target)

# Perform parameterized gridsearch
# Note the function is not yet been fully parameterized..
def gridSearchPermutations(grid,inp,target):
	bestAcc = -1
	best = np.zeros(2)
	for values in grid:
		# acc = crossvalidateSVM(inp,target,values[0],values[1])
		# Use our own cross validation implementation
		acc = crossvalidateSVM(inp,target,values[0],values[1])

		# Uncomment to use Sci-kit learns own cross validation function
		# clf = svm.SVC(kernel='rbf',C = x, gamma = y)
		# acc = cross_validation.cross_val_score(clf, inp, target, cv=5).mean()
		if acc > bestAcc:
			bestAcc = acc
			best = values
	return best, bestAcc

# Returns the number of bounded support vectors in a SVM
def countBoundedSV(clf):
	v = clf.dual_coef_
	return len(v[np.where(np.fabs(v) == clf.C)])

# Returns the number of free support vectors in a SVM
def countFreeSV(clf):
	v = clf.dual_coef_
	return len(v[np.where(np.fabs(v) < clf.C)])
