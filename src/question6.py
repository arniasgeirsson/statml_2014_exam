import modules.common as com
import numpy as np
import math

toNormalize = False
d, i_train, t_train, _ = com.parseData("data/SGTrain2014.dt", delimiter=",")
_, i_train0, t_train0, _ = com.parseData("data/SGTrain2014.dt", delimiter=",",target_class=[0],
                                        normalizeX=toNormalize)
_, i_train1, t_train1, _ = com.parseData("data/SGTrain2014.dt", delimiter=",",target_class=[1],
                                        normalizeX=toNormalize,
                                        normY=com.splitdata(d)[0])

# def g(L):
#     l = []
#     for n in L:
#         if not n in l:
#             l.append(n)
#     return np.array(l)

# tt = g(t_train)

def h0(x,X,d,m,d2):
    return np.linalg.norm(x-m)

def h1(x,X,d,m,d2):
    return math.sqrt(np.linalg.norm(x-m)**2)

def h2(x,X,d,m,d2):
    return math.sqrt(np.dot(x,x)-2.0*np.dot(x,m) + np.dot(m,m))

def h3(x,X,d,m,d2):
    a = np.dot(x,x)
    b = 2.0 * np.dot(x,d)
    c = np.dot(d,d)
    return math.sqrt(a-b+c)

def h4(x,X,d,m,d2):
    a = np.dot(x,x)
    ln = (1.0/len(X))
    b = 0.0
    for x_ in X:
        b = b + np.dot(x,x_)
    b = 2.0 * ln * b

    c = 0.0
    for x_ in X:
        c = c + np.dot(x_,d)
    c = ln * c
    return math.sqrt(a-b+c)

def h5(x,X,d,m,d2):
    a = np.dot(x,x)
    ln = (1.0/len(X))
    b = 0.0
    for x_ in X:
        b = b + (2.0*np.dot(x,x_) - np.dot(x_,d))
    b = ln * b

    return math.sqrt(a-b)

def h5_(x,X,d,m,d2):
    print "h5_"
    a = np.dot(x,x)
    ln = (1.0/len(X))
    b = 0.0
    for x_ in X:
        b_ = 0.0
        for x__ in X:
            b_ = b_ + np.dot(x_,x__)
        b_ = ln * b_
        b = b + (2.0*np.dot(x,x_) - b_)
    b = ln * b

    return math.sqrt(a-b)

def h6(x,X,d,m,d2):
    a = np.dot(x,x)
    ln = (1.0/len(X))
    b = 0.0
    for x_ in X:
        b = b + np.dot(x,x_)
    b = 2.0 * ln * b

    return math.sqrt(a-b+d2)

# The same as h6
def h6_(x,X,d,m,d2):
    a = np.dot(x,x)
    ln = (2.0/len(X))
    b = 0.0
    for x_ in X:
        b = b + np.dot(x,x_)
    b = ln * b

    return math.sqrt(a-b+d2)

def h7(x,X,d,m,d2):
    ln = (2.0/len(X))
    b = 0.0
    for x_ in X:
        b = b + np.dot(x,x_)
    b = ln * b

    return -b+d2

print "init start"
l = np.zeros(len(t_train))
ln0 = (1.0/len(i_train0))
ln1 = (1.0/len(i_train1))
d0 = 0.0
for x__ in i_train0:
    d0 = d0 + x__
d0 = ln0 * d0
d1 = 0.0
for x__ in i_train1:
    d1 = d1 + x__
d1 = ln1 * d1
m0 = com.mean(i_train0)
m1 = com.mean(i_train1)
d20 = 0.0
for x__ in i_train0:
    d20_ = 0.0
    for x___ in i_train0:
        d20_ = d20_ + np.dot(x__,x___)
    d20_ = ln0 * d20_
    d20 = d20 + d20_
d20 = ln0 * d20
d21 = 0.0
for x__ in i_train1:
    d21_ = 0.0
    for x___ in i_train1:
        d21_ = d21_ + np.dot(x__,x___)
    d21_ = d21_
    d21 = d21 + d21_
d21 = (ln1*ln1) * d21
print "init done"

def h(X=i_train,H=h0,p=1000):
    for i in range(len(t_train)):
        c0 = H(X[i],i_train0,d0,m0,d20)
        c1 = H(X[i],i_train1,d1,m1,d21)
        if (i % p == 0):
            print i
        if c0 < c1:
            l[i] = 0
        else:
            l[i] = 1
    return l

print "acc: ", com.accuracy(h(i_train,h0),t_train)
print "acc: ", com.accuracy(h(i_train,h1),t_train)
print "acc: ", com.accuracy(h(i_train,h2),t_train)
print "acc: ", com.accuracy(h(i_train,h3),t_train)
# print "acc: ", com.accuracy(h(i_train,h4),t_train)
# print "acc: ", com.accuracy(h(i_train,h5),t_train)
# print "acc: ", com.accuracy(h(i_train,h5_,p=1),t_train)
# print "acc: ", com.accuracy(h(i_train,h6),t_train)
# print "acc: ", com.accuracy(h(i_train,h6_),t_train)
# print "acc: ", com.accuracy(h(i_train,h7),t_train)
