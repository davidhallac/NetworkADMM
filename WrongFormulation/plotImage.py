
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np

#x = [[[ 1.16409404, 0.91336569, 0.22815908],\
#[ 1.2308858, 0.28260795,  0.5286923 ]],[\
#[ 1.16409404,  0.91336569,  0.22815908],\
#[ 1.2308858,   0.28260795,  0.5286923 ]]]
x = [[[ 0.5, 0.5, 0.5],\
[ 0.1, 0.1, 0.1 ]],\
[[ 1.16409404,  0.91336569,  0.22815908],\
[ 1.2308858,   0.28260795,  0.5286923 ]]]


y = np.array(x)
yt = y.transpose()

L = y.min()
U = y.max()



print y
print (y-L)/(U-L)


plt.figure(1)
plt.imshow(y, interpolation='nearest')
#imshow(y, interpolation='bilinear')
#imshow(y, interpolation='bicubic')

#plt.grid(True)

plt.show()
