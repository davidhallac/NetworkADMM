from cvxpy import *
import itertools
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import rc
#from base_test import BaseTest
from cvxopt import matrix
from numpy import linalg as LA
import numpy as np
import scipy as sp
#import cvxpy as cp
import unittest
import math
import sys
from cStringIO import StringIO

#import resource; print 'Memory usage End: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def solveX(data):
	inputs = int(data[len(data)-1])
	lamb = data[len(data)-2]
	rho = data[len(data)-3]
	numtests = int(data[len(data)-4])
	c = data[len(data)-5]
	x = data[0:inputs]
	x_train = data[inputs: inputs + numtests*inputs]
	y_train = data[inputs+numtests*inputs: inputs+numtests*(inputs+1)]
	neighs = data[inputs+numtests*(inputs+1):len(data)-3]
	a = Variable(inputs,1)
	epsil = Variable(numtests,1)
	b = Variable(1,1)
	constraints = [epsil >= 0]
	g = 0.5*square(norm(a)) + c*norm(epsil,1)
	for i in range(numtests):
		temp = np.asmatrix(x_train[i*inputs:i*inputs+numtests])
		constraints = constraints + [y_train[i]*(temp*a + b) >= 1 - epsil[i]]
	f = 0
	for i in range(len(neighs)/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			f = f + rho/2*square(norm(a - z + u))
	objective = Minimize(5*g + 5*lamb*f)
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#result = p.solve(verbose=True)
		objective = Minimize(g+0.99*f)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		print "SCALING BUG"

	return a.value

def solveZ(data):
	#Old method: h = lamb*weight*norm(z1-z2)+rho/2*(square(norm(x1+u1-z1))+square(norm(x2+u2-z2)))
	inputs = int(data[len(data)-1])
	lamb = data[len(data)-2]
	rho = data[len(data)-3]
	x1 = data[0:inputs]
	x2 = data[inputs:2*inputs]
	u1 = data[2*inputs:3*inputs]
	u2 = data[3*inputs:4*inputs]
	weight = data[len(data)-4]
	a = x1 + u1
	b = x2 + u2
	theta = max(1 - lamb*weight/(rho*LA.norm(a - b)), 0.5)
	z1 = theta*a + (1-theta)*b
	z2 = theta*b + (1-theta)*a
	znew = np.matrix(np.concatenate([z1, z2])).reshape(2*inputs,1)
	return znew

def solveU(data):
	leng = len(data)-1
	u = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[len(data)-1]
	return u + (x - z)

def runADMM_Grid(m, edges, inputs, lamb, rho, numiters, x, u, z, S, ids, numtests, x_train, y_train, c):

	#Find max degree of graph
	maxdeg = 0;
	for i in range(m):
		maxdeg = max(maxdeg, len(S.getrow(i).nonzero()[1]))
	iters = 0

	#Stopping criteria
	(s, r, epri, edual) = (1,1,0,0)
	A = np.zeros((2*edges, m))
	for i in range(edges):
		A[2*i,ids[i,0]] = 1
		A[2*i+1, ids[i,1]] = 1
	sqn = math.sqrt(m*inputs)
	sqp = math.sqrt(2*inputs*edges)
	eabs = math.pow(10,-3) #CHANGE THESE TWO AS PARAMS
	erel = math.pow(10,-4)

	maxProcesses =  80
	pool = Pool(processes = min(max(m, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):
		#x update
		neighs = np.zeros(((2*inputs+1)*maxdeg,m))
		for i in range(m):
			counter = 0
			for j in range(edges):
			#Smaller number always first. This is ij with i < j
				if(ids[j,0] == i):
					neighs[counter*(2*inputs+1),i] = S.getrow(i).getcol(ids[j,1]).todense() #weight
					neighs[counter*(2*inputs+1)+1:counter*(2*inputs+1)+(inputs+1),i] = u[:,2*j] #u_ij 
					neighs[counter*(2*inputs+1)+(inputs+1):(counter+1)*(2*inputs+1),i] = z[:,2*j] #z_ij
					counter = counter + 1
				elif(ids[j,1] == i):
					neighs[counter*(2*inputs+1),i] = S.getrow(i).getcol(ids[j,0]).todense()
					neighs[counter*(2*inputs+1)+1:counter*(2*inputs+1)+(inputs+1),i] = u[:,2*j+1]
					neighs[counter*(2*inputs+1)+(inputs+1):(counter+1)*(2*inputs+1),i] = z[:,2*j+1]
					counter = counter + 1
		temp = np.concatenate((x,x_train,y_train,neighs,np.tile([c,numtests,rho,lamb,inputs], (m,1)).transpose()), axis=0)
		newx = pool.map(solveX, temp.transpose())
		#print "Solved X", iters
		x = np.array(newx).transpose()[0]

		#z update
		ztemp = z.reshape(2*inputs, edges, order='F')
		utemp = u.reshape(2*inputs, edges, order='F')
		xtemp = np.zeros((inputs, 2*edges))
		weights = np.zeros((1, edges))
		for j in range(edges):
			weights[0,j] = S.getrow(ids[j,0]).getcol(ids[j,1]).todense()
			xtemp[:,2*j] = np.array(x[:,ids[j,0]])
			xtemp[:,2*j+1] = x[:,ids[j,1]]
		xtemp = xtemp.reshape(2*inputs, edges, order='F')
		temp = np.concatenate((xtemp,utemp,ztemp,weights,np.tile([rho,lamb,inputs], (edges,1)).transpose()), axis=0)
		newz = pool.map(solveZ, temp.transpose())
		ztemp = np.array(newz).transpose()[0]
		ztemp = ztemp.reshape(inputs, 2*edges, order='F')
		s = LA.norm(rho*np.dot(A.transpose(),(ztemp - z).transpose())) #For dual residual
		z = ztemp

		#u update
		xtemp = np.zeros((inputs, 2*edges))
		for j in range(edges):
			xtemp[:,2*j] = np.array(x[:,ids[j,0]])
			xtemp[:,2*j+1] = x[:,ids[j,1]]
		temp = np.concatenate((u, xtemp, z, np.tile(rho, (1,2*edges))), axis=0)
		newu = pool.map(solveU, temp.transpose())
		u = np.array(newu).transpose()

		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s #updated at z-step

		#print r, epri, s, edual
		iters = iters + 1
	
	pool.close()
	pool.join()		
	x_actual = x#np.array(x)
	return (x_actual, u, z, x_actual, 0, 0)	

def main():
	
	size = 100
	m = size
	partitions = 5
	inputs = 10
	rho = 0.25
	maxedges = size*(size-1) #maximum possible edges

	sizepart = size/partitions
	(weights, ids) = (np.ones((maxedges,1)).flatten(), np.zeros((maxedges, 2)).astype(np.int64))
	counter = 0
	np.random.seed(2)
	for i in range(size):
		for j in range(i,size):
			if(((i/sizepart) == (j/sizepart)) and i != j):
				#Same partition
				if(np.random.random() >= 0.5):
					ids[counter,0] = round(i)
					ids[counter,1] = round(j)
					counter = counter+1
			elif (i != j):
				#Different partition
				if(np.random.random() >= 0.9):
					ids[counter,0] = round(i)
					ids[counter,1] = round(j)
					counter = counter+1
	S = sp.sparse.coo_matrix((weights,(ids[:,0].flatten(),ids[:,1].flatten())), shape=(size,size))
	S = S + S.transpose()
	#S[0,0] = 0 SHOULDNT MATTER, RIGHT??
	edges = counter #Actual number of edges

	#True a at each partition
	a_true = np.random.randn(inputs, partitions)
	#number of given values in the training set at each node
	numtests = 10
	testSetSize = 5
	v = np.random.randn(numtests,size)
	vtest = np.random.randn(testSetSize,size)

	x_train = np.random.randn(numtests*inputs, size)
	y_train = np.zeros((numtests,size))
	for i in range(size):
		a_part = a_true[:,i/sizepart]
		for j in range(numtests):
			y_train[j,i] = np.sign([np.dot(a_part.transpose(), x_train[j*inputs:j*inputs+numtests,i])+v[j,i]])

	x_test = np.random.randn(testSetSize*inputs, size)
	y_test = np.zeros((testSetSize, size))
	for i in range(size):
		a_part = a_true[:,i/sizepart]
		for j in range(testSetSize):
			y_test[j,i] = np.sign([np.dot(a_part.transpose(), x_test[j*inputs:j*inputs+numtests,i])+vtest[j,i]])

	(a_pred,x,u,z,counter) = (np.zeros((inputs,m)),np.zeros((inputs,m)),np.zeros((inputs,2*edges)),np.zeros((inputs,2*edges)),1)

	numiters = 25
	c = 0.79 #Between 0.785 and 0.793 NOT SURE WHAT THIS IS??
	thresh = 1.2
	lamb = 0.1#0.04
	updateVal = 1.5#1.05
	numtrials = math.log(thresh/lamb, updateVal) + 1 
	plots =	np.zeros((math.floor(numtrials)+1,2))
	#Solve for lambda = 0
	(x, u, z, xSol, pl1, pl2) = runADMM_Grid(m, edges, inputs, 0, 0.00001, numiters, x, u ,z, S, ids, numtests, x_train, y_train, c)
	a_pred = x
	#Test results on test set
	right = 0
	total = testSetSize*size
	for i in range(size):
		temp = a_pred[:,i]
		for j in range(testSetSize):
			pred = np.sign([np.dot(temp.transpose(), x_test[j*inputs:j*inputs+numtests,i])])
			if(pred == y_test[j,i]):
				right = right + 1
	print "Lambda = 0, ", right / float(total)
	plots[counter-1,:] = [0, right/float(total)]
	
	while(lamb <= thresh):
		print "Lambda = ", lamb
		(x, u, z, xSol, pl1, pl2) = runADMM_Grid(m, edges, inputs, lamb, rho, numiters, x, u ,z, S, ids, numtests, x_train, y_train, c)
		a_pred = x
		right = 0
		total = testSetSize*size
		for i in range(size):
			temp = a_pred[:,i]
			for j in range(testSetSize):
				pred = np.sign([np.dot(temp.transpose(), x_test[j*inputs:j*inputs+numtests,i])])
				if(pred == y_test[j,i]):
					right = right + 1
		plots[counter,:] = [lamb, right/float(total)]
		counter = counter + 1
		lamb = lamb*updateVal
		#lamb = lamb + 0.05
		print right / float(total)
	print "Finished"

	# plt.figure(0)
	# plt.plot(plots[:,0], plots[:,1], 'ro')
	# plt.xlabel('$\lambda$')
	# plt.ylabel('Prediction Accuracy')	
	# #plt.xscale('log')
	# plt.savefig('svm_reg_path',bbox_inches='tight')


	#plt.show()

if __name__ == '__main__':
	main()
