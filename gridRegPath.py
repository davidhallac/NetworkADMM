#import cvxpy.settings as s
#from cvxpy.atoms import *
#from cvxpy.expressions.constants import Constant, Parameter
#from cvxpy.expressions.variables import Variable
#from cvxpy.problems.objective import *
#from cvxpy.problems.problem import Problem
#import cvxpy.interface as intf
from cvxpy import *
import itertools
#import parmap
from multiprocessing import Pool
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
    y = data[inputs:2*inputs]
    z = data[2*inputs:3*inputs]
    a = data[3*inputs:4*inputs]
    neighs = data[4*inputs:len(data)-3]
    xnew = cp.Variable(inputs, 1)
    g = 0.5*cp.square(norm(xnew - a))
    h = 0
    for i in range(inputs): #This can be written better...
        h = h + y[i]*(xnew[i] - z[i])
    s = cp.square(norm(xnew - z))
    w = 0 #TODO fill in later
    for i in range(len(neighs)/(inputs+1)):
        w = w + neighs[i*(inputs+1)]*norm(xnew - neighs[i*(inputs+1)+1:i*(inputs+1)+6])
    objective = cp.Minimize(g + lamb/2*w + h + rho/2*s)
    constraints = []
    p = cp.Problem(objective, constraints)
    result = p.solve()
    return xnew.value

def solveZ(data):
    inputs = int(data[len(data)-1])
    lamb = data[len(data)-2]
    rho = data[len(data)-3]
    x = data[0:inputs]
    y = data[inputs:2*inputs]
    z = data[2*inputs:3*inputs]
    neighs = data[4*inputs:len(data)-3]
    znew = cp.Variable(inputs, 1)
    h = 0
    for i in range(inputs): #This can be written better...
        h = h + y[i]*(x[i] - znew[i])
    s = cp.square(norm(x - znew))
    w = 0 #TODO fill in later
    for i in range(len(neighs)/(inputs+1)):
        w = w + neighs[i*(inputs+1)]*norm(znew - neighs[i*(inputs+1)+1:i*(inputs+1)+6])
    objective = cp.Minimize(lamb/2*w + h + rho/2*s)
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

def runADMM_Random(m, edges, inputs, outputs, lamb, rho, numiters, x, y, z):
    np.random.seed(0)
    a = np.random.randn(inputs, m)

    #Get edges, build sparse matrix
    ids = np.random.randint(0,m, size=(edges,2))
    for i in ids:
        while (i[0] == i[1]):
            i[1] = np.random.randint(0,m)
    counter = 0
    for i in ids:
        counter = counter + 1
        for j in ids[counter:]:
            while ((i[0] == j[0] and i[1] == j[1]) or (i[1] == j[0] and i[0] == j[1])):
                j[0] = np.random.randint(0,m)
                j[1] = np.random.randint(0,m)
                while (j[0] == j[1]):
                    j[1] = np.random.randint(0,m)
                while True:
                    badentry = 0
                    for k in ids[0:counter]:
                        if ((j[0] == k[0] and j[1] == k[1]) or (j[0] == k[1] and j[1] == k[0])):
                            badentry = 1
                    if (badentry == 0):
                        break
                    j[0] = np.random.randint(0,m)
                    j[1] = np.random.randint(0,m)
                    while (j[0] == j[1]):
                        j[1] = np.random.randint(0,m)
    temp = np.random.rand(1,edges).flatten()
    S = sp.sparse.coo_matrix((temp,(ids[:,0].flatten(),ids[:,1].flatten())), shape=(m,m))
    S = S + S.transpose() #Bidirectional edges. Sparse matrix

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



    #Run ADMM
    #Find max degree of graph
    maxdeg = 0;
    for i in range(m):
        maxdeg = max(maxdeg, len(S.getrow(i).nonzero()[1])).value

    iters = 0
    pool = Pool(processes = m)
    while(iters < numiters):
        #x update
        neighs = np.zeros(((inputs+1)*maxdeg,m))
        for i in range(m):
            counter = 0
            for j in S.getrow(i).nonzero()[1]:
                neighs[counter*(inputs+1),i] = S.getrow(i).getcol(j).todense()
                neighs[counter*(inputs+1)+1:counter*(inputs+1)+6,i] = z[:,j]
                counter = counter+1
        temp = np.concatenate((x,y,z,a,neighs,np.tile([rho,lamb,inputs], (m,1)).transpose()), axis=0)
        newx = pool.map(solveX, temp.transpose())
        x = np.array(newx).transpose()[0]

        #z update
        for i in range(m):
            counter = 0
            for j in S.getrow(i).nonzero()[1]:
                neighs[counter*(inputs+1),i] = S.getrow(i).getcol(j).todense()
                neighs[counter*(inputs+1)+1:counter*(inputs+1)+6,i] = x[:,j]
                counter = counter+1
        temp = np.concatenate((x,y,z,a,neighs,np.tile([rho,lamb,inputs], (m,1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        z = np.array(newz).transpose()[0]

        #y update
        temp = np.concatenate((y, x, z, np.tile(rho, (1,m))), axis=0)
        newy = pool.map(solveY, temp.transpose())
        y = np.array(newy).transpose()

        iters = iters + 1
        print norm(x - x_actual.value).value


    print x
    print x-x_actual.value
    return (x, y, z)


def main():
    #Parameters
    m = 10 #number of nodes
    edges = 20 
    inputs = 5
    outputs = 1
    rho = 0.05
    lamb = 2
    numiters = 10
    #Initilialize variables
    x = 0.2*np.ones((inputs,m))
    y = 0.2*np.ones((inputs,m))
    z = 0.2*np.ones((inputs,m))
    
    (x, y, z) = runADMM_Random(m, edges, inputs, outputs, lamb, rho, numiters, x, y ,z)
    
    #lambs = [1,1.25,1.5,1.75,2,2.25]
    #for i in range(5):
    #    lamb = lambs[i]
    #    (x, y, z) = runADMM(m, edges, inputs, outputs, lamb, rho, numiters, x, y ,z)
    print "Finished"

if __name__ == '__main__':
    main()

