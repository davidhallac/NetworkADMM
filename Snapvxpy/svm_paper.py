from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool
#Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Other function in this folder
from z_u_solvers import solveZ, solveU

def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	sizeData = data[data.size-4]
	numtests = int(data[data.size-5])
	c = data[data.size-6]
	x = data[0:inputs]
	rawData = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-6]
	xnew = Variable(inputs,1)

	#Fill in objective function here! Params: Xnew (unknown), a (side data at node)
	x_train = rawData[0:numtests*inputs]
	y_train = rawData[numtests*inputs: numtests*(inputs+1)]

	a = Variable(inputs,1)
	epsil = Variable(numtests,1)
	b = Variable(1,1)
	constraints = [epsil >= 0]
	g = 0.5*square(norm(a)) + c*norm(epsil,1)
	for i in range(numtests):
		temp = np.asmatrix(x_train[i*inputs:i*inputs+numtests])
		constraints = constraints + [y_train[i]*(temp*a + b) >= 1 - epsil[i]]
	f = 0
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			f = f + rho/2*square(norm(a - z + u))
	objective = Minimize(5*g + 5*f)
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#result = p.solve(verbose=True)
		objective = Minimize(g+0.99*f)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		print "Scaling bug"
	return a.value


# def solveZ(data):
# 	inputs = int(data[data.size-1])
# 	lamb = data[data.size-2]
# 	rho = data[data.size-3]
# 	useConvex = data[data.size-4]
# 	weight = data[data.size-5]
# 	x1 = data[0:inputs]
# 	x2 = data[inputs:2*inputs]
# 	u1 = data[2*inputs:3*inputs]
# 	u2 = data[3*inputs:4*inputs]
# 	a = x1 + u1
# 	b = x2 + u2
# 	(z1, z2) = (0,0)
# 	if(useConvex == 1):
# 		theta = max(1 - lamb*weight/(rho*LA.norm(a - b)+0.000001), 0.5) #So no divide by zero error
# 		z1 = theta*a + (1-theta)*b
# 		z2 = theta*b + (1-theta)*a
# 	else: #Non-convex version
# 		epsilon = 0.1
# 		d = LA.norm(a-b)
# 		c = lamb*weight

# 		if(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c >= 0):
# 			theta1 = (-rho*(d+epsilon) + math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d)
# 			theta1 = min(max(theta1,0),0.5)
# 			phi = math.log(1 + d*(1-2*theta1)/epsilon)
# 			objective1 = c*phi + rho*math.pow(d,2)*(math.pow(theta1,2))

# 			theta2 = (-rho*(d+epsilon) - math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d)
# 			theta2 = min(max(theta2,0),0.5)
# 			phi = math.log(1 + d*(1-2*theta2)/epsilon)
# 			objective2 = c*phi + rho*math.pow(d,2)*(math.pow(theta2,2))

# 			objective3 = rho/4*math.pow(LA.norm(a-b),2)

# 			if(min(objective1, objective2, objective3) == objective1):
# 				theta = min(max(theta1,0),0.5)
# 			elif(min(objective1, objective2, objective3) == objective2):
# 				theta = min(max(theta2,0),0.5)
# 			else:
# 				theta = 0.5
# 			z1 = (1-theta)*a + theta*b
# 			z2 = theta*a + (1-theta)*b
# 		else: #No real roots, use theta = 0.5
# 			(z1, z2) = (0.5*a + 0.5*b, 0.5*a + 0.5*b)

# 	znew = np.matrix(np.concatenate([z1, z2])).reshape(2*inputs,1)
# 	return znew

# def solveU(data):
# 	leng = data.size-1
# 	u = data[0:leng/3]
# 	x = data[leng/3:2*leng/3]
# 	z = data[(2*leng/3):leng]
# 	rho = data[data.size-1]
# 	return u + (x - z)

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights, numtests, useConvex, eabs, erel, c):

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
	bestObj = -1

	#Run ADMM
	iters = 0
	maxProcesses =  80
	pool = Pool(processes = min(max(nodes, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):
		
		tempObj = 0 #For non-convex

		#TODO: Update tempObj

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
		temp = np.concatenate((x,a,neighs,np.tile([c, numtests,sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
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
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(weightsList, (-1,edges)),np.tile([useConvex,rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
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
			#Calculate objective

			#Update best variables
			if(tempObj < bestObj or bestObj = -1)
				bestx = x
				bestu = u
				bestz = z
				bestObj = tempObj
				print "Updated best objective at iter ", iters, "TODO!!!"


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
	useConvex = 0 #1 = true, 0 = false
	rho = 0.0001
	numiters = 40
	thresh = 2
	lamb = 0.0
	updateVal = 0.02
	#Stopping criteria
	eabs = math.pow(10,-2)
	erel = math.pow(10,-3)
	#Graph Information
	nodes = 100
	#Size of x
	sizeOptVar = 10
	#C in SVM
	c = 0.79

	partitions = 5

	#Generate graph, edge weights
	np.random.seed(2)
	G1 = TUNGraph.New()
	for i in range(nodes):
		G1.AddNode(i)
	sizepart = nodes/partitions
	for NI in G1.Nodes():
		for NI2 in G1.Nodes():
			if(NI.GetId() != NI2.GetId()):				
				if ((NI.GetId()/sizepart) == (NI2.GetId()/sizepart)):
					#Same partition, edge w.p 0.5
					if(np.random.random() >= 0.5):
						G1.AddEdge(NI.GetId(), NI2.GetId())
				else:
					if(np.random.random() >= 0.9):
						G1.AddEdge(NI.GetId(), NI2.GetId())

	edges = G1.GetEdges()

	edgeWeights = TIntPrFltH()
	for EI in G1.Edges():
		temp = TIntPr(EI.GetSrcNId(), EI.GetDstNId())
		edgeWeights.AddDat(temp, 1)

	#Generate side information
	a_true = np.random.randn(sizeOptVar, partitions)
	numtests = 10 #Training set size
	testSetSize = 5
	v = np.random.randn(numtests,nodes)
	vtest = np.random.randn(testSetSize,nodes)

	trainingSet = np.random.randn(numtests*(sizeOptVar+1), nodes) #First all the x_train, then all the y_train below it
	for i in range(nodes):
		a_part = a_true[:,i/sizepart]
		for j in range(numtests):
			trainingSet[numtests*sizeOptVar+j,i] = np.sign([np.dot(a_part.transpose(), trainingSet[j*sizeOptVar:(j+1)*sizeOptVar,i])+v[j,i]])

	(x_test,y_test) = (np.random.randn(testSetSize*sizeOptVar, nodes), np.zeros((testSetSize, nodes)))
	for i in range(nodes):
		a_part = a_true[:,i/sizepart]
		for j in range(testSetSize):
			y_test[j,i] = np.sign([np.dot(a_part.transpose(), x_test[j*sizeOptVar:(j+1)*sizeOptVar,i])+vtest[j,i]])

	sizeData = trainingSet.shape[0]

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*edges))
	z = np.zeros((sizeOptVar,2*edges))

	#Run regularization path
	[plot1, plot2] = [TFltV(), TFltV()]
	while(lamb <= thresh):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb), numiters, x, u ,z, trainingSet, edgeWeights, numtests, useConvex, eabs, erel, c)
		print "Lambda = ", lamb

		#Get accuracy
		(right, total) = (0, testSetSize*nodes)
		a_pred = x
		for i in range(nodes):
			temp = a_pred[:,i]
			for j in range(testSetSize):
				pred = np.sign([np.dot(temp.transpose(), x_test[j*sizeOptVar:j*sizeOptVar+numtests,i])])
				if(pred == y_test[j,i]):
					right = right + 1
		accuracy = right / float(total)
		print accuracy

		plot1.Add(lamb)
		plot2.Add(accuracy)
		lamb = lamb + updateVal


	#Print/Save plot
	pl1 = np.array(plot1)
	pl2 = np.array(plot2)
	plt.plot(pl1, pl2)
	plt.savefig('image_svm',bbox_inches='tight')



if __name__ == '__main__':
	main()
