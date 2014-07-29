from cvxpy import *
from multiprocessing import Pool
import numpy as np
import scipy as sp
import snap

def f(x):
	xnew = Variable(3,1)
	a = [ 2.52106488, 2.28690449, 1.28005947]
	g = 0.5*square(norm(xnew - a))
	h = 0
	h = h + 0.5/2*square(norm(xnew - [ 2.16379838, 2.18759151, 2.14236057] + [ 0.32418347, 0.16236409, 0.00561542]))
	h = h + 0.5/2*square(norm(xnew - [ 1.99005998, 2.39417974, 2.79228709] + [ 0.49792187, -0.04422414, -0.6443111 ]))
	h = h + 0.5/2*square(norm(xnew - [ 2.46764519, 2.21118802, 2.09606314] + [ 0.02033666, 0.13876758, 0.05191285]))
	h = h + 0.5/2*square(norm(xnew - [ 2.30517462, 1.91537804, 2.11904844] + [ 0.18280723, 0.43457756, 0.02892755]))

	objective = Minimize(g+h)
	constraints = []
	p = Problem(objective, constraints)
	result = p.solve()

	print result
	return xnew.value


pool = Pool(processes = 5)
pool.map(f, range(100))

pool.close()
pool.join()