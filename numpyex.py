#example for numpy

import numpy as np

x = np.array([[1,2,3],[4,5,6]], np.int32)
print '* type', type(x)
print '* shape', x.shape
print '* dtype', x.dtype
print '* print(x): \n', x
print '* print(x[1])', x[0]

print '\n* exersice of slicing'

b=x[:2, 1:4]
print '* print b\n', b

c=x[1,:]
print "* print c\n", c

print "* print b.T\n", b.T

y = np.empty_like(x)
v = np.array([1,0,1,1])
print "* print emptylike of x\n", y
print '* print v\n', v

for i in range(3):
        y[i,:] = x[i,:] + v
print '* result of y=x+v\n', y

vv = np.tile(v,(3,1))
print '* print vv\n', vv
