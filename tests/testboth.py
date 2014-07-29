# from cvxpy import *
from multiprocessing import Pool
import numpy as np
import scipy as sp
import snap

G1 = snap.TUNGraph.New()
G1.AddNode(1)
G1.AddNode(5)
G1.AddNode(32)
G1.AddEdge(1,5)
G1.AddEdge(5,1)
G1.AddEdge(5,32)

for NI in G1.Nodes():
	for Id in NI.GetOutEdges():
		print "edge (%d %d)" % (NI.GetId(), Id)


# xnew = Variable(3,1)
# a = [ 2.52106488, 2.28690449, 1.28005947]
# g = 0.5*square(norm(xnew - a))
# h = 0
# h = h + 0.5/2*square(norm(xnew - [ 2.16379838, 2.18759151, 2.14236057] + [ 0.32418347, 0.16236409, 0.00561542]))
# h = h + 0.5/2*square(norm(xnew - [ 1.99005998, 2.39417974, 2.79228709] + [ 0.49792187, -0.04422414, -0.6443111 ]))
# h = h + 0.5/2*square(norm(xnew - [ 2.46764519, 2.21118802, 2.09606314] + [ 0.02033666, 0.13876758, 0.05191285]))
# h = h + 0.5/2*square(norm(xnew - [ 2.30517462, 1.91537804, 2.11904844] + [ 0.18280723, 0.43457756, 0.02892755]))

# objective = Minimize(g+h)
# constraints = []
# p = Problem(objective, constraints)
# result = p.solve()

# print result


nodes = 10
G = snap.GenRndGnm(snap.PNEANet,nodes, nodes*(nodes-1)/4)

# define int, float and str attributes on nodes
G.AddIntAttrN("NValInt", 0)
G.AddFltAttrN("NValFlt", 0.0)
G.AddStrAttrN("NValStr", "0")

# define an int attribute on edges
G.AddIntAttrE("EValInt", 0)

# add attribute values, node ID for nodes, edge ID for edges

for NI in G.Nodes():
	nid = NI.GetId()
	val = nid
	G.AddIntAttrDatN(nid, val, "NValInt")
	G.AddFltAttrDatN(nid, float(val), "NValFlt")
	G.AddStrAttrDatN(nid, str(val), "NValStr")

	for nid1 in NI.GetOutEdges():
		eid = G.GetEId(nid,nid1)
		val = eid
		G.AddIntAttrDatE(eid, val, "EValInt")

# print out attribute values

for NI in G.Nodes():
	nid = NI.GetId()
	ival = G.GetIntAttrDatN(nid, "NValInt")
	fval = G.GetFltAttrDatN(nid, "NValFlt")
	sval = G.GetStrAttrDatN(nid, "NValStr")
	print "node %d, NValInt %d, NValFlt %.2f, NValStr %s" % (nid, ival, fval, sval)

	for nid1 in NI.GetOutEdges():
		eid = G.GetEId(nid, nid1)
		val = G.GetIntAttrDatE(eid, "EValInt")
		print "edge %d (%d,%d), EValInt %d" % (eid, nid, nid1, val)
		#print G.GetFltAttrDatN(nid, "NValFlt"), G.GetFltAttrDatN(nid1, "NValFlt")

snap.PlotInDegDistr(G, "Hello", "World")





