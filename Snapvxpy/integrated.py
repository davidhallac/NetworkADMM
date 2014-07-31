from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool

def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	x = data[0:inputs]
	a = data[inputs:2*inputs]
	neighs = data[2*inputs:data.size-3]
	xnew = Variable(inputs,1)
	g = 0.5*square(norm(xnew - a))
	h = 0
	for i in range(neighs.size/(2*inputs+1)):
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
		#result = p.solve(verbose=True)
		objective = Minimize(g+1.000001*h)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		print "SCALING BUG"
	return xnew.value

def solveZ(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	x1 = data[0:inputs]
	x2 = data[inputs:2*inputs]
	u1 = data[2*inputs:3*inputs]
	u2 = data[3*inputs:4*inputs]
	weight = data[data.size-4]
	a = x1 + u1
	b = x2 + u2
	theta = max(1 - lamb*weight/(rho*LA.norm(a - b)+0.000001), 0.5) #So no divide by zero error
	z1 = theta*a + (1-theta)*b
	z2 = theta*b + (1-theta)*a
	znew = np.matrix(np.concatenate([z1, z2])).reshape(2*inputs,1)
	return znew

def solveU(data):
	#TODO: replace data.size with len(data)
	leng = data.size-1
	u = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[data.size-1]
	return u + (x - z)

def runADMM(G1, sizeOptVar, lamb, rho, numiters, x, u, z, a, edgeWeights):

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	#Find max degree of graph; hash the nodes
	(maxdeg, counter) = (0, 0)
	node2mat = TIntIntH()
	for NI in G1.Nodes():
		maxdeg = max(maxdeg, NI.GetDeg())
		node2mat.AddDat(NI.GetId(), counter)
		counter = counter + 1

	#Stopping criteria
	eabs = math.pow(10,-2)
	erel = math.pow(10,-3)
	(r, s, epri, edual, counter) = (1,1,0,0,0)
	A = np.zeros((2*edges, nodes))
	for EI in G1.Edges():
		A[2*counter,node2mat.GetDat(EI.GetSrcNId())] = 1
		A[2*counter+1, node2mat.GetDat(EI.GetDstNId())] = 1
		counter = counter+1
	(sqn, sqp) = (math.sqrt(nodes*sizeOptVar), math.sqrt(2*sizeOptVar*edges))

	#Run ADMM
	iters = 0
	maxProcesses =  80
	pool = Pool(processes = min(max(nodes, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):

		#x-update
		(neighs, counter) = (np.zeros(((2*sizeOptVar+1)*maxdeg,nodes)), 0)
		for NI in G1.Nodes():
			counter2 = 0
			edgenum = 0
			for EI in G1.Edges():
				if(EI.GetSrcNId() == NI.GetId()):
					neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights[counter]
					neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum] #u_ij 
					neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum] #z_ij
					counter2 = counter2 + 1
				elif(EI.GetDstNId() == NI.GetId()):
					1+1
				edgenum = edgenum+1
		temp = np.concatenate((x,a,neighs,np.tile([rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
		newx = pool.map(solveX, temp.transpose())
		x = np.array(newx).transpose()[0]

		#z-update
		ztemp = z.reshape(2*sizeOptVar, edges, order='F')
		utemp = u.reshape(2*sizeOptVar, edges, order='F')
		xtemp = np.zeros((sizeOptVar, 2*edges))
		counter = 0
		for EI in G1.Edges():
			xtemp[:,2*counter] = np.array(x[:,node2mat.GetDat(EI.GetSrcNId())])
			xtemp[:,2*counter+1] = x[:,node2mat.GetDat(EI.GetDstNId())]
			counter = counter+1
		xtemp = xtemp.reshape(2*sizeOptVar, edges, order='F')
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(edgeWeights, (-1,edges)),np.tile([rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
		newz = pool.map(solveZ, temp.transpose())
		ztemp = np.array(newz).transpose()[0]
		ztemp = ztemp.reshape(sizeOptVar, 2*edges, order='F')
		s = LA.norm(rho*np.dot(A.transpose(),(ztemp - z).transpose())) #For dual residual
		z = ztemp

		#u-update
		(xtemp, counter) = (np.zeros((sizeOptVar, 2*edges)), 0)
		for EI in G1.Edges():
			xtemp[:,2*counter] = np.array(x[:,node2mat.GetDat(EI.GetSrcNId())])
			xtemp[:,2*counter+1] = x[:,node2mat.GetDat(EI.GetDstNId())]
			counter = counter + 1
		temp = np.concatenate((u, xtemp, z, np.tile(rho, (1,2*edges))), axis=0)
		newu = pool.map(solveU, temp.transpose())
		u = np.array(newu).transpose()


		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s

		print r, epri, s, edual
		iters = iters + 1

	pool.close()
	pool.join()


	return (x,u,z,0,0)

def main():

	#Generate graph, edge weights
	nodes = 25
	edges = 100
	G1 = GenRndGnm(PUNGraph, nodes, edges)
	edgeWeights = np.ones((edges,1)).flatten()

	#Generate data
	sizeData = 20
	a = np.random.randn(sizeData, nodes)

	#Initialize variables to 0
	sizeOptVar = 5
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*edges))
	z = np.zeros((sizeOptVar,2*edges))

	#Set parameters
	rho = 5
	numiters = 10
	thresh = 1
	lamb = 0.1
	updateVal = 0.05

	#Run regularization path
	while(lamb <= thresh):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, lamb, rho, numiters, x, u ,z, a, edgeWeights)
		print "Lambda = ", lamb
		lamb = lamb + updateVal



	
	counter = 0
	node2mat = TIntIntH()
	for NI in G1.Nodes():
		#print "node id %d with out-degree %d and in-degree %d" % (NI.GetId(), NI.GetDeg(), NI.GetDeg())
		node2mat.AddDat(NI.GetId(), counter)
		counter = counter + 1


	counter = 0
	for NI in G1.Nodes():
		xnew = Variable(sizeData,1)
		g = 0.5*square(norm(xnew - a[:,counter]))
		objective = Minimize(g)
		constraints = []
		p = Problem(objective, constraints)
		result = p.solve()
		#print xnew.value
		counter = counter + 1



	for EI in G1.Edges():
		1+1#print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())

	for NI in G1.Nodes():
		for Id in NI.GetOutEdges():
			1+1#print "edge (%d %d)" % (NI.GetId(), Id)




if __name__ == '__main__':
	main()
