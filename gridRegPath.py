from cvxpy import *
import itertools
#import parmap
from multiprocessing import Pool
import matplotlib.pyplot as plt
#from base_test import BaseTest
from cvxopt import matrix
from numpy import linalg as LA
import numpy as np
import scipy as sp
import cvxpy as cp
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
	xnew = cp.Variable(inputs,1)
	g = 0.5*cp.square(norm(xnew - a))
	h = 0
	for i in range(len(neighs)/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			y = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*cp.square(norm(xnew - z))
			for j in range(inputs): #This can be written better...
				h = h + y[j]*(xnew[j])
	objective = cp.Minimize(g+h)
	constraints = []
	p = cp.Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		objective = cp.Minimize(g+1.1*h)
		constraints = []
		p = cp.Problem(objective, constraints)
		result = p.solve()
		print "CVXPY BUG???", result
	return xnew.value


def solveZ(data):
	inputs = int(data[len(data)-1])
	lamb = data[len(data)-2]
	rho = data[len(data)-3]
	x1 = data[0:inputs]
	x2 = data[inputs:2*inputs]
	y1 = data[2*inputs:3*inputs]
	y2 = data[3*inputs:4*inputs]
	z = data[4*inputs:6*inputs]
	weight = data[len(data)-4]
	znew = cp.Variable(2*inputs,1)
	z1 = znew[0:inputs]
	z2 = znew[inputs:2*inputs]
	h = lamb*weight*norm(z1-z2)+rho/2*(cp.square(norm(x1-z1))+cp.square(norm(x2-z2)))
	for i in range(inputs):	
		h = h - y1[i]*z1[i] - y2[i]*z2[i]
	objective = cp.Minimize(h)
	constraints = []
	p = cp.Problem(objective, constraints)
	result = p.solve()
	return znew.value

def solveY(data):
	leng = len(data)-1
	y = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[len(data)-1]
	return y + rho*(x - z)


def runADMM_Grid(m, edges, inputs, outputs, lamb, rho, numiters, x, y, z, S, ids, a):
	#Find actual solution
	x_actual = cp.Variable(inputs,m)
	g = 0
	for i in range(m):
		g = g + 0.5*cp.square(norm(x_actual[:,i] - a[:,i]))
	f = 0
	for i in range(edges):
		f = f + S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*norm(x_actual[:,ids[i,0]] - x_actual[:,ids[i,1]])
	objective = cp.Minimize(g + lamb*f)
	constraints = []
	p = cp.Problem(objective, constraints)
	result_actual = p.solve()

	#Find max degree of graph
	maxdeg = 0;
	for i in range(m):
		maxdeg = max(maxdeg, len(S.getrow(i).nonzero()[1]))
	iters = 0

	#Stopping criteria
	s = 1
	r = 1
	epri = 0
	edual = 0
	A = np.zeros((2*edges, m))
	for i in range(edges):
		A[2*i,ids[i,0]] = 1
		A[2*i+1, ids[i,1]] = 1
	sqn = math.sqrt(m*inputs)
	sqp = math.sqrt(2*inputs*edges)
	eabs = math.pow(10,-3) #CHANGE THESE TWO AS PARAMS
	erel = math.pow(10,-3)



	pool = Pool(processes = max(m, edges))
	while(iters < numiters and (r > epri or s > edual or iters < 5)):
		#x update
		neighs = np.zeros(((2*inputs+1)*maxdeg,m))
		for i in range(m):
			counter = 0
			for j in range(edges):
			#Smaller number always first. This is ij with i < j
				if(ids[j,0] == i):
					neighs[counter*(2*inputs+1),i] = S.getrow(i).getcol(ids[j,1]).todense() #weight
					neighs[counter*(2*inputs+1)+1:counter*(2*inputs+1)+(inputs+1),i] = y[:,2*j] #y_ij 
					neighs[counter*(2*inputs+1)+(inputs+1):(counter+1)*(2*inputs+1),i] = z[:,2*j] #z_ij
					counter = counter + 1
				elif(ids[j,1] == i):
					neighs[counter*(2*inputs+1),i] = S.getrow(i).getcol(ids[j,0]).todense()
					neighs[counter*(2*inputs+1)+1:counter*(2*inputs+1)+(inputs+1),i] = y[:,2*j+1]
					neighs[counter*(2*inputs+1)+(inputs+1):(counter+1)*(2*inputs+1),i] = z[:,2*j+1]
					counter = counter + 1
		temp = np.concatenate((x,a,neighs,np.tile([rho,lamb,inputs], (m,1)).transpose()), axis=0)
		newx = pool.map(solveX, temp.transpose())
		x = np.array(newx).transpose()[0]
 
		#z update
		ztemp = z.reshape(2*inputs, edges, order='F')
		ytemp = y.reshape(2*inputs, edges, order='F')
		xtemp = np.zeros((inputs, 2*edges))
		weights = np.zeros((1, edges))
		for j in range(edges):
			weights[0,j] = S.getrow(ids[j,0]).getcol(ids[j,1]).todense()
			xtemp[:,2*j] = np.array(x[:,ids[j,0]])
			xtemp[:,2*j+1] = x[:,ids[j,1]]
		xtemp = xtemp.reshape(2*inputs, edges, order='F')
		temp = np.concatenate((xtemp,ytemp,ztemp,weights,np.tile([rho,lamb,inputs], (edges,1)).transpose()), axis=0)
		newz = pool.map(solveZ, temp.transpose())
		ztemp = np.array(newz).transpose()[0]
		ztemp = ztemp.reshape(inputs, 2*edges, order='F')
		s = LA.norm(rho*np.dot(A.transpose(),(ztemp - z).transpose())) #For dual residual
		z = ztemp
		#print z

		#y update
		xtemp = np.zeros((inputs, 2*edges))
		for j in range(edges):
			#print j, ids[j,0], x[:,ids[j,0]]
			xtemp[:,2*j] = np.array(x[:,ids[j,0]])
			xtemp[:,2*j+1] = x[:,ids[j,1]]
		temp = np.concatenate((y, xtemp, z, np.tile(rho, (1,2*edges))), axis=0)
		newy = pool.map(solveY, temp.transpose())
		y = np.array(newy).transpose()

		#Stopping criterion - p19 of ADMM paper. TODO: Not 100% accurate for now. Need to change
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),y.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s #updated at z-step

		print r, epri, s, edual
		obj2 = 0
		for i in range(m):
			obj2 = obj2 + 0.5*(LA.norm(x[:,i] - a[:,i]))*(LA.norm(x[:,i] - a[:,i]))
		obj1 = 0
		for i in range(edges):
			#obj1 = obj1 + lamb*S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*LA.norm(x[:,ids[i,0]] - x[:,ids[i,1]])
			obj1 = obj1 + S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*LA.norm(x[:,ids[i,0]] - x[:,ids[i,1]])
		print lamb*obj1 + obj2 - result_actual, result_actual
		iters = iters + 1

	pool.close()
	pool.join()
	#print LA.norm(x)
	#print LA.norm(x - x_actual.value)
	#USE ACTUAL SOLUTION
	#x = np.array(x_actual.value)
	obj2 = 0
	for i in range(m):
		obj2 = obj2 + 0.5*(LA.norm(x[:,i] - a[:,i]))*(LA.norm(x[:,i] - a[:,i]))
	obj1 = 0
	for i in range(edges):
		#obj1 = obj1 + lamb*S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*LA.norm(x[:,ids[i,0]] - x[:,ids[i,1]])
		obj1 = obj1 + S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*LA.norm(x[:,ids[i,0]] - x[:,ids[i,1]])
	print lamb*obj1 + obj2 - result_actual, result_actual
	return (x, y, z, x_actual, obj1, obj2)

def main():
	#Build 2x2 Grid
	size = 8
	edges = 2*size*(size-1)
	m = size*size
	inputs = 3 #RGB
	outputs = 1
	rho = 0.5
	#lamb = 4 #Not used anymore...

	#Uniform weights, build grid
	weights = np.ones((edges,1)).flatten()
	ids = np.zeros((edges, 2)).astype(np.int64)
	counter = 0
	for i in range(m):
		if(i < m - size):
			ids[counter,0] = round(i)
			ids[counter,1] = round(i+size)
			counter = counter + 1
		if((i+1) % size != 0):
			ids[counter,0] = round(i)
			ids[counter,1] = round(i+1)
			counter = counter + 1
	S = sp.sparse.coo_matrix((weights,(ids[:,0].flatten(),ids[:,1].flatten())), shape=(m,m))
	S = S + S.transpose() #+ sp.sparse.identity(m)

	#Random "targets" a
	np.random.seed(0)
	a = np.random.randn(inputs, m)
	#Add offets
	for i in range(m):
		if (i < (m/2)):
			if (i % size < (size/2)):
				#Quadrant 1
				a[:,i] = a[:,i] + [4,1,1]
			else:
				a[:,i] = a[:,i] + [1,4,1] #Quadrant 2
		else:
			if (i % size < (size/2)):
				a[:,i] = a[:,i] + [1,1,4] #Quadrant 3
			else:
				a[:,i] = a[:,i] + [2,2,2] #Quadrant 4

	
	#GenCon Solution - TODO make this distributed
	x = 0.0*np.ones((inputs,m))
	y = 0.0*np.ones((inputs,2*edges))
	z = 0.0*np.ones((inputs,2*edges))
	counter = 1 #For plotting
	x_con = cp.Variable(inputs,1)
	g = 0
	for i in range(m):
		g = g + 0.5*cp.square(norm(x_con - a[:,i]))
	objective = cp.Minimize(g)
	constraints = []
	p = cp.Problem(objective, constraints)
	result_actual = p.solve()
	
	#Convert GenCon to z
	z = np.asarray(np.tile(x_con.value, (1,2*edges)))
	
	#Get UACTUAL
	(u1, u2, u3, u4, pl1, pl2) = runADMM_Grid(m, edges, inputs, outputs, 0, rho, 25, x, y ,z, S, ids, a)
	U = u1.max()
	L = u1.min()

	numiters = 100
	numtrials = 1
	plots = np.zeros((numtrials,2))
	for lamb in np.linspace(4, 4, num=numtrials):
		(x, y, z, xSol, pl1, pl2) = runADMM_Grid(m, edges, inputs, outputs, lamb, rho, numiters, x, y ,z, S, ids, a)
		#x = np.array(xSol.value)
		xplot = np.reshape(x.transpose(), (size, size, inputs))
		plt.figure(counter)
		plt.imshow((xplot-L)/(U-L), interpolation='nearest')
		plots[counter-1,:] = [pl1, pl2]
		counter = counter + 1
	print "Finished"
	#print x.shape

	#Plot noiseless
	#plt.figure(counter)
	#a = np.random.randn(inputs, m)
	#Add offets
	#for i in range(m):
	#    if (i < (m/2)):
	#        if (i % size < (size/2)):
	#            a[:,i] = [4,1,1]
	#        else:
	#            a[:,i] = [1,4,1]
	#    else:
	#        if (i % size < (size/2)):
	#            a[:,i] = [1,1,4]
	#        else:
	#            a[:,i] = [2,2,2]
	#xplot = np.reshape(np.array(a).transpose(), (size, size, inputs))
	#plt.imshow((xplot-L)/(U-L), interpolation='nearest')

	plt.figure(counter)
	plt.plot(plots[:,0], plots[:,1], 'ro')
	#plt.grid(True)
	
	plt.show()


if __name__ == '__main__':
	main()