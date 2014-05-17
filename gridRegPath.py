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
        w = w + neighs[i*(inputs+1)]*norm(xnew - neighs[i*(inputs+1)+1:i*(inputs+1)+(inputs+1)])
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
        w = w + neighs[i*(inputs+1)]*norm(znew - neighs[i*(inputs+1)+1:i*(inputs+1)+(inputs+1)])
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


    #Run ADMM
    #Find max degree of graph
    maxdeg = 0;
    for i in range(m):
        maxdeg = max(maxdeg, len(S.getrow(i).nonzero()[1]))
    iters = 0
    pool = Pool(processes = m)
    while(iters < numiters):
        #x update
        neighs = np.zeros(((inputs+1)*maxdeg,m))
        for i in range(m):
            counter = 0
            for j in S.getrow(i).nonzero()[1]:
                neighs[counter*(inputs+1),i] = S.getrow(i).getcol(j).todense()
                neighs[counter*(inputs+1)+1:counter*(inputs+1)+(inputs+1),i] = z[:,j]
                counter = counter+1
        temp = np.concatenate((x,y,z,a,neighs,np.tile([rho,lamb,inputs], (m,1)).transpose()), axis=0)
        newx = pool.map(solveX, temp.transpose())
        x = np.array(newx).transpose()[0]

        #z update
        for i in range(m):
            counter = 0
            for j in S.getrow(i).nonzero()[1]:
                neighs[counter*(inputs+1),i] = S.getrow(i).getcol(j).todense()
                neighs[counter*(inputs+1)+1:counter*(inputs+1)+(inputs+1),i] = x[:,j]
                counter = counter+1
        temp = np.concatenate((x,y,z,a,neighs,np.tile([rho,lamb,inputs], (m,1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        z = np.array(newz).transpose()[0]

        #y update
        temp = np.concatenate((y, x, z, np.tile(rho, (1,m))), axis=0)
        newy = pool.map(solveY, temp.transpose())
        y = np.array(newy).transpose()

        iters = iters + 1

        #Print objective
        obj1 = 0
        for i in range(m):
            obj1 = obj1 + 0.5*(LA.norm(x[:,i] - a[:,i]))*(LA.norm(x[:,i] - a[:,i]))
        obj2 = 0
        for i in range(edges):
            obj2 = obj2 + S.getrow(ids[i,0]).getcol(ids[i,1]).todense()*LA.norm(x[:,ids[i,0]] - x[:,ids[i,1]])
        obj = obj1 + lamb*obj2
        print obj - result_actual
        #print norm(x - x_actual.value).value

    pool.close()
    pool.join()
    print x
    print x_actual.value
    return (x, y, z, x_actual)


def main():
    #Build 2x2 Grid
    size = 8
    edges = 2*size*(size-1)
    m = size*size
    inputs = 3 #RGB
    outputs = 1
    rho = 0.05
    lamb = 5
    numiters = 10

    #Uniform weights
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


    #Initilialize variables for GenCon

    #GenCon Solution - TODO make this distributed
    x = 0.2*np.ones((inputs,m))
    y = 0.0*np.ones((inputs,m))
    z = 0.2*np.ones((inputs,m))
    counter = 2 #For plotting
    x_con = cp.Variable(inputs,1)
    g = 0
    for i in range(m):
        g = g + 0.5*cp.square(norm(x_con - a[:,i]))
    objective = cp.Minimize(g)
    constraints = []
    p = cp.Problem(objective, constraints)
    result_actual = p.solve()
    z = np.asarray(np.tile(x_con.value, (1,m)))

    
    #Get UACTUAL
    (u1, u2, u3, u4) = runADMM_Grid(m, edges, inputs, outputs, 0, rho, 5, x, y ,z, S, ids, a)
    U = u1.max()
    L = u1.min()


    (x, y, z, xSol) = runADMM_Grid(m, edges, inputs, outputs, lamb, rho, numiters, x, y ,z, S, ids, a)
    xplot = np.reshape(x.transpose(), (size, size, inputs))
    plt.figure(1)
    plt.imshow((xplot-L)/(U-L), interpolation='nearest')
    
    numiters = 10
    numtrials = 5
    for lamb in np.linspace(4, 0, num=numtrials):
        (x, y, z, xSol) = runADMM_Grid(m, edges, inputs, outputs, lamb, rho, numiters, x, y ,z, S, ids, a)
        xplot = np.reshape(x.transpose(), (size, size, inputs))
        plt.figure(counter)
        plt.imshow((xplot-L)/(U-L), interpolation='nearest')
        counter = counter + 1
    print "Finished"

    #PLOTTING
    #plt.figure(2)
    #x2plot = np.reshape(np.array(xSol.value).transpose(), (size, size, inputs))
    #plt.imshow((x2plot-L)/(U-L), interpolation='nearest')

    #plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

