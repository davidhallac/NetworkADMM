from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool

import csv


def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	sizeData = data[data.size-4]
	x = data[0:inputs]
	a = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-4]
	xnew = Variable(inputs,1)
	#Fill in objective function here! Params: Xnew (unknown), a (side data at node)
	g = 0.5*square(norm(xnew - a))

	h = 0
	print "Stop 6"
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*square(norm(xnew - z + u))
	print "Stop 7"
	objective = Minimize(5*g+5*h)
	constraints = []
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#Todo: CVXOPT scaling issue
		objective = Minimize(g+1.001*h)
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
	leng = data.size-1
	u = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[data.size-1]
	return u + (x - z)

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights):

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
				if (node2mat.GetDat(EI.GetSrcNId()) == NI.GetId()):
					#print "Found: ", NI.GetId(), "Connected to", EI.GetDstNId()
					neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
					neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum] #u_ij 
					neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum] #z_ij
					counter2 = counter2 + 1
				elif (node2mat.GetDat(EI.GetDstNId()) == NI.GetId()):
					#print "Found: ", NI.GetId(), "Connected to", EI.GetSrcNId()
					neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
					neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum+1] #u_ij 
					neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum+1] #z_ij
					counter2 = counter2 + 1
				edgenum = edgenum+1
			counter = counter + 1
		temp = np.concatenate((x,a,neighs,np.tile([sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
		newx = pool.map(solveX, temp.transpose())
		x = np.array(newx).transpose()[0]

		#z-update
		ztemp = z.reshape(2*sizeOptVar, edges, order='F')
		utemp = u.reshape(2*sizeOptVar, edges, order='F')
		xtemp = np.zeros((sizeOptVar, 2*edges))
		counter = 0
		weightsList = np.zeros((1, edges))
		for EI in G1.Edges():
			xtemp[:,2*counter] = np.array(x[:,node2mat.GetDat(EI.GetSrcNId())])
			xtemp[:,2*counter+1] = x[:,node2mat.GetDat(EI.GetDstNId())]
			weightsList[0,counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
			counter = counter+1
		xtemp = xtemp.reshape(2*sizeOptVar, edges, order='F')
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(weightsList, (-1,edges)),np.tile([rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
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

	#Set parameters
	rho = 0.1
	numiters = 25
	thresh = 0.15
	lamb = 0.0
	updateVal = 0.1
	#Graph Information
		#nodes = 54
		#edges = 100
	#Size of x
	sizeOptVar = 5
	#Size of side information at each node
	sizeData = 5


	#Generate graph, edge weights
	file = open("Sacramentorealestatetransactions.csv", "rU")
	file.readline() #ignore first line
	G1 = TUNGraph.New()
	locations = TIntFltPrH()
	counter = 0
	for line in file:
		G1.AddNode(counter)
		temp = TFltPr(float(line.split(",")[10]),float(line.split(",")[11]))
		locations.AddDat(counter, temp)
		counter = counter + 1

	#For each node, find closest neightbors and add edge, weight = 5/distance
	edgeWeights = TIntPrFltH()
	numNeighs = 10
	for NI in G1.Nodes():
		distances = TIntFltH()
		lat1 = locations.GetDat(NI.GetId()).GetVal1()
		lon1 = locations.GetDat(NI.GetId()).GetVal2()
		for NI2 in G1.Nodes():
			if(NI.GetId() != NI2.GetId()):
				lat2 = locations.GetDat(NI2.GetId()).GetVal1()
				lon2 = locations.GetDat(NI2.GetId()).GetVal2()
				dlon = math.radians(lon2 - lon1)
				dlat = math.radians(lat2 - lat1)
				a = math.pow(math.sin(dlat/2),2) + math.cos(lat1)*math.cos(lat2) * math.pow(math.sin(dlon/2),2)
				c = 2 * math.atan2( math.sqrt(a), math.sqrt(1-a) ) 
				dist = 3961 * c
				distances.AddDat(NI2.GetId(), dist)

		distances.Sort(False, True)
		it = distances.BegI()
		for j in range(numNeighs):
			if (not G1.IsEdge(NI.GetId(), it.GetKey())):
				G1.AddEdge(NI.GetId(), it.GetKey())
				#Add edge weight
				temp = TIntPr(min(NI.GetId(), it.GetKey()), max(NI.GetId(), it.GetKey()))
				edgeWeights.AddDat(temp, 1/(it.GetDat()+ 0.1))
			it.Next()		

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	print nodes, edges

	#Generate side information
	a = np.zeros((sizeData, nodes))
	file = open("Sacramentorealestatetransactions.csv", "rU")
	file.readline() #ignore first line
	counter = 0
	for line in file:
		a[0,counter] = float(line.split(",")[4])
		a[1,counter] = float(line.split(",")[5])
		a[2,counter] = float(line.split(",")[6])
		if(line.split(",")[7] == "Residential"):
			a[3,counter] = 1
		elif(line.split(",")[7] == "Condo"):
			a[3,counter] = 2
		elif(line.split(",")[7] == "Multi-Family"):
			a[3,counter] = 3
		a[4,counter] = float(line.split(",")[9])
		counter = counter + 1

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*G1.GetEdges()))
	z = np.zeros((sizeOptVar,2*G1.GetEdges()))

	#Run regularization path
	while(lamb <= thresh):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + lamb/10, numiters, x, u ,z, a, edgeWeights)
		print "Lambda = ", lamb
		lamb = lamb + updateVal







if __name__ == '__main__':
	main()
