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


def solveX(data):
	inputs = int(data[len(data)-1])
	lamb = data[len(data)-2]
	rho = data[len(data)-3]
	x = data[0:inputs]
	a = data[inputs:2*inputs]
	neighs = data[2*inputs:len(data)-3]
	xnew = Variable(inputs,1)
	g = 0.5*square(norm(xnew - a))
	h = 0
	for i in range(len(neighs)/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*square(norm(xnew - z + u))
	objective = Minimize(g+h)
	constraints = []
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		for i in range(len(neighs)/(2*inputs+1)):
			weight = neighs[i*(2*inputs+1)]
			if(weight != 0):
				u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
				z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
				#print u, z
		#print neighs, a
		objective = Minimize(2*g+h)
		p = Problem(objective, constraints)
		result = p.solve()
		print "CVXPY BUG?", p	
	return xnew.value

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


	a = Variable(inputs,m)
	epsil = Variable(numtests,m)
	b = Variable(1,m)
	g = 0
	constraints = [epsil >= 0]
	for i in range(m):
		g = g + 0.5*square(norm(a[:,i]))+ c*norm(epsil[:,i],1)
		for j in range(numtests):
			temp = np.asmatrix(x_train[j*inputs:j*inputs+numtests,i])
			constraints = constraints + [y_train[j,i]*(temp*a[:,i] + b[i]) >= 1 - epsil[j,i]]
	f = 0
	for i in range(edges):
		f = f + S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*norm(a[:,ids[i,0]] - a[:,ids[i,1]])
	objective = Minimize(g + lamb*f)
	p = Problem(objective, constraints)
	result_actual = p.solve()


	x_actual = np.array(a.value)
	#print x_actual
	return (x_actual, u, z, x_actual, 0, 0)	

def main():
	
	size = 100
	m = size
	partitions = 5
	inputs = 10
	rho = 0.5
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

	(a_pred,u,z,counter) = (np.zeros((inputs,m)),np.zeros((inputs,2*edges)),np.zeros((inputs,2*edges)),1)

	numiters = 0
	c = 0.79 #Between 0.785 and 0.793 NOT SURE WHAT THIS IS??
	thresh = 1
	lamb = 0.5#0.04
	updateVal = 1.5#1.05
	numtrials = math.log(thresh/lamb, updateVal) + 1 
	plots =	np.zeros((math.floor(numtrials)+1,2))
	#Solve for lambda = 0
	(a_pred, u, z, xSol, pl1, pl2) = runADMM_Grid(m, edges, inputs, 0, 0.00001, numiters, a_pred, u ,z, S, ids, numtests, x_train, y_train, c)
	#Test results on test set
	right = 0
	total = testSetSize*size
	for i in range(size):
		temp = a_pred[:,i]
		for j in range(testSetSize):
			pred = np.sign([np.dot(temp.transpose(), x_test[j*inputs:j*inputs+numtests,i])])
			if(pred == y_test[j,i]):
				right = right + 1
	print right / float(total)
	plots[counter-1,:] = [0, right/float(total)]
	
	while(lamb <= thresh):
		(a_pred, u, z, xSol, pl1, pl2) = runADMM_Grid(m, edges, inputs, lamb, rho, numiters, a_pred, u ,z, S, ids, numtests, x_train, y_train, c)
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
		print "Lambda = ", lamb
		print right / float(total)
		lamb = lamb*updateVal
		#lamb = lamb + 0.05
	print "Finished"

	# #Plot noiseless
	# #a_noiseless = a - a_noise
	# #xplot = np.reshape(np.array(a_noiseless).transpose(), (size, size, inputs))
	# #plt.figure(counter)
	# #plt.imshow((xplot-L)/(U-L), interpolation='nearest')

	plt.figure(0)
	plt.plot(plots[:,0], plots[:,1], 'ro')
	plt.xlabel('$\lambda$')
	plt.ylabel('Prediction Accuracy')	
	#plt.xscale('log')
	plt.savefig('svm_reg_path',bbox_inches='tight')


	#plt.show()

if __name__ == '__main__':
	main()