from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool
#Plotting
import csv
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
import matplotlib.pyplot as plt
#Other function in this folder
from z_u_solvers import solveZ, solveU

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
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*square(norm(xnew - z + u))
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
	return xnew.value, g.value

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights):

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	maxNonConvexIters = 6*numiters

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

	#Non-convex case - keeping track of best point so far
	bestx = x
	bestu = u
	bestz = z
	bestObj = 0
	cvxObj = 10000000*np.ones((1, nodes))
	if(useConvex != 1):
		#Calculate objective
		for i in range(G1.GetNodes()):
			bestObj = bestObj + cvxObj[0,i] #TODO: Update this
		for EI in G1.Edges():
			weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
			edgeDiff = LA.norm(x[:,node2mat.GetDat(EI.GetSrcNId())] - x[:,node2mat.GetDat(EI.GetDstNId())])
			bestObj = bestObj + lamb*weight*math.log(1 + edgeDiff / epsilon)
		initObj = bestObj

	#Run ADMM
	iters = 0
	maxProcesses =  80
	pool = Pool(processes = min(max(nodes, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):

		#x-update
		neighs = np.zeros(((2*sizeOptVar+1)*maxdeg,nodes))
		edgenum = 0
		numSoFar = TIntIntH()
		for EI in G1.Edges():
			if (not numSoFar.IsKey(EI.GetSrcNId())):
				numSoFar.AddDat(EI.GetSrcNId(), 0)
			counter = node2mat.GetDat(EI.GetSrcNId())
			counter2 = numSoFar.GetDat(EI.GetSrcNId())
 			neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
 			neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum] 
 			neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum]
			numSoFar.AddDat(EI.GetSrcNId(), counter2+1)

			if (not numSoFar.IsKey(EI.GetDstNId())):
				numSoFar.AddDat(EI.GetDstNId(), 0)
			counter = node2mat.GetDat(EI.GetDstNId())
			counter2 = numSoFar.GetDat(EI.GetDstNId())
 			neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
 			neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum+1] 
 			neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum+1]
			numSoFar.AddDat(EI.GetDstNId(), counter2+1)
			edgenum = edgenum+1
		temp = np.concatenate((x,a,neighs,np.tile([c, numtests,sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
		values = pool.map(solveX, temp.transpose())
		newx = np.array(values)[:,0].tolist()
		newcvxObj = np.array(values)[:,1].tolist()
		x = np.array(newx).transpose()[0]
		cvxObj = np.reshape(np.array(newcvxObj), (-1, nodes))

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
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(weightsList, (-1,edges)),np.tile([epsilon, useConvex, rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
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

		#Update best objective (for non-convex)
		if(useConvex != 1):
			tempObj = 0
			#Calculate objective
			for i in range(G1.GetNodes()):
				tempObj = tempObj + cvxObj[0,i]
			initTemp = tempObj
			for EI in G1.Edges():
				weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
				edgeDiff = LA.norm(x[:,node2mat.GetDat(EI.GetSrcNId())] - x[:,node2mat.GetDat(EI.GetDstNId())])
				tempObj = tempObj + lamb*weight*math.log(1 + edgeDiff / epsilon)

			#Update best variables
			if(tempObj < bestObj or bestObj == -1):
				bestx = x
				bestu = u
				bestz = z
				bestObj = tempObj
				print "Iteration ", iters, "; Obj = ", tempObj, "; Initial = ", initTemp
			else:
			 	print "FAILED AT ITERATION ", iters, "; Obj = ", tempObj, "; Initial = ", initTemp

			if(iters == numiters - 1 and numiters < maxNonConvexIters):
				if(bestObj == initObj):
					numiters = numiters+1

		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s

		#print r, epri, s, edual
		iters = iters + 1

	pool.close()
	pool.join()

	if(useConvex == 1):
		return (x,u,z,0,0)
	else:
		return (bestx,bestu,bestz,0,0)




















def main():

	#Set parameters
	rho = 0.1
	numiters = 25
	thresh = -0.15
	lamb = 0.0
	updateVal = 0.1
	#Graph Information
	nodes = 54
	edges = 100
	#Size of x
	sizeOptVar = 5
	#Size of side information at each node
	sizeData = 5


	#Generate graph, edge weights
	file = open("Data/CalIt2.csv", "rU")
	#file.readline() #ignore first line
	G1 = TUNGraph.New()
	counter = 0
	#for line in file:
	while True:
		# temp = TFltPr(float(line.split(",")[10]),float(line.split(",")[11]))
		# locations.AddDat(counter, temp)
		# tempData = TFltV()
		# tempData.Add(float(line.split(",")[4]))
		# tempData.Add(float(line.split(",")[5]))
		# tempData.Add(float(line.split(",")[6]))
		# if(line.split(",")[7] == "Residential"):
		# 	tempData.Add(1)
		# elif(line.split(",")[7] == "Condo"):
		# 	tempData.Add(2)
		# elif(line.split(",")[7] == "Multi-Family"):
		# 	tempData.Add(3)
		# else:
		# 	tempData.Add(4)
		# tempData.Add(float(line.split(",")[12])*10) #12 for normalized; 9 for raw
		# dataset.AddDat(counter, tempData)
		line = file.readline() #7 --outflow
		if not line: 
			break
		outward = line.split(",")[3]
		line = file.readline()
		inward = line.split(",")[3]
		print outward, inward
		G1.AddNode(counter)


		counter = counter + 1

	file = open("location_sensors.txt", "r")
	G1 = TUNGraph.New()
	locations = TIntFltPrH()
	for line in file:
		G1.AddNode(int(line.split()[0]))
		temp = TFltPr(float(line.split()[1]),float(line.split()[2]))
		locations.AddDat(int(line.split()[0]), temp)

	#For each node, find closest neightbors and add edge, weight = 5/distance
	edgeWeights = TIntPrFltH()
	numNeighs = 3
	for NI in G1.Nodes():
		distances = TIntFltH()
		x1 = locations.GetDat(NI.GetId()).GetVal1()
		y1 = locations.GetDat(NI.GetId()).GetVal2()
		for NI2 in G1.Nodes():
			if(NI.GetId() != NI2.GetId()):
				x2 = locations.GetDat(NI2.GetId()).GetVal1()
				y2 = locations.GetDat(NI2.GetId()).GetVal2()
				dist = math.sqrt(math.pow(x1-x2,2) + math.pow(y1 - y2,2))
				distances.AddDat(NI2.GetId(), dist)
		distances.Sort(False, True)
		it = distances.BegI()
		for j in range(numNeighs):
			if (not G1.IsEdge(NI.GetId(), it.GetKey())):
				G1.AddEdge(NI.GetId(), it.GetKey())
				#Add edge weight
				temp = TIntPr(min(NI.GetId(), it.GetKey()), max(NI.GetId(), it.GetKey()))
				edgeWeights.AddDat(temp, 5/it.GetDat())
			it.Next()



	#Generate side information
	np.random.seed(2)
	a = np.random.randn(sizeData, nodes)

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*G1.GetEdges()))
	z = np.zeros((sizeOptVar,2*G1.GetEdges()))

	#Run regularization path
	while(lamb <= thresh):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb), numiters, x, u ,z, a, edgeWeights)
		print "Lambda = ", lamb
		lamb = lamb + updateVal







if __name__ == '__main__':
	main()
