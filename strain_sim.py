#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:45:21 2026

@author: user
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import dune.geometry
import dune.grid
import dune.functions
import matplotlib.pyplot as plt

L = 40 + 39
v_h = 0.2 #Poisson ratio
beta = 50/200 * 573/400
x_1 = 18#17
x_2 = 62#64

    
def data_fitting(fitting_param=None, discrete_steps = 8000):
    # the fitting parameters are A, sigma, and l_param sequentially
    if fitting_param is None:
        #constants
        A = -0.5 #in unit (%)
        sigma = 0.1
        l_param = 1
    else:
        A = fitting_param[0] #in unit (%)
        sigma = fitting_param[1]
        l_param = fitting_param[2]
    f_L = lambda x: A * np.exp(-(x[0]-x_1)**2 / (2*sigma**2) )
    f_R = lambda x: A * np.exp(-(x[0]-x_2)**2 / (2*sigma**2))
    f = lambda x: (1-v_h)*(f_L(x) + f_R(x))/(1+beta)
    grid = dune.grid.structuredGrid([0], [L], [discrete_steps])
    basis = dune.functions.defaultGlobalBasis(grid, dune.functions.Lagrange(order=2))
    A,b = globalAssembler(basis, f, l_param)

    # compress the matrix
    Acsr = A.tocsr()
    
    
    # solve the sparse system efficiently (hint: use specific scipy solver)
    # Note: if the error "Matrix is exactly singular" occurs, check whether Dirichlet conditions are fulfilled!
    c = spsolve(Acsr, b)
    
    # d) plot the result!
    # create a dicrete grid function from the basis and coefficients of the solution
    cFunction = basis.asFunction(c)
    x = []
    y = []
    for element in grid.elements:
        #TODO
        local_center = element.geometry.center # Local coordinate of the cell center
        global_center = element.geometry.toGlobal(local_center)
        x.append(global_center)
        y.append(cFunction(element,local_center))
    y = np.array(y)
    x = np.array(x)
    # plt.plot(x, y)
    # plt.ylabel("epsilon_h (%)")
    # plt.xlabel("x(um)")
    # #plt.yticks(ticks=np.arange(0,0.3,0.1))
    # plt.show()
    return x, y

def l2Error(basis,f):
    
    # create a coefficient vector
    coeff = np.zeros(len(basis))
    
    # interpolate f in the basis
    # this sets the entries of coeff!
    basis.interpolate(coeff,f)
    
    # create a dicrete grid function
    fDiscrete = basis.asFunction(coeff)
    
    # choose a high enough quadrature order to integrate f and fDiscrete exactly
    quadOrder = 4 # TODO
    sum = 0
    # integrate over each element
    for element in grid.elements:
        #TODO
        quadRule = dune.geometry.quadratureRule(element.type, quadOrder)
        for pt in quadRule:
            integrationElement = element.geometry.integrationElement(pt.position)
            sum += integrationElement * pt.weight * (f( element.geometry.toGlobal(pt.position) ) - fDiscrete(element, pt.position))**2
        
    return np.sqrt(sum)
def isDirichlet(x):
    distance = []
    if x[0] < 0.5*L:
        distance.append(x[0])
    else:
        distance.append(L-x[0])
    if min(distance) < 1e-3:
        return True # TODO
    else:
        return False
def localAssembler(element,localBasis, f, l_param):
    
    n = len(localBasis)

    # dense local matrix and vector
    localA = np.zeros( (n,n) )
    localB = np.zeros( n )
    # now integrate over the grid element and compute the entries of localA and localB
    # see prog4_solution.py for hints how to construct the gradients of the local basis functions,
    # use again a quadrature rule to compute the integral
    quadOrder = 3
    quadRule = dune.geometry.quadratureRule(element.type, quadOrder)
    for pt in quadRule:
        jacobianInverseTransposed = element.geometry.jacobianInverseTransposed( pt.position )
        phiRefGradient = localBasis.evaluateJacobian( pt.position )
        phiGradient = [ np.dot(jacobianInverseTransposed, np.array(g)[0]) for g in phiRefGradient ]
        integrationElement = element.geometry.integrationElement(pt.position)
        phi = localBasis.evaluateFunction( pt.position )
        for k in range(n):
            for l in range(n):
                f_1 = l_param**2*np.dot(np.array(phiGradient[k]), np.array(phiGradient[l]))
                f_2 = phi[k][0] * phi[l][0]
                localA[k][l] += integrationElement * pt.weight * (f_1 + f_2)
    quadOrder = 2
    quadRUle = dune.geometry.quadratureRule(element.type, quadOrder)
    for pt in quadRule:
        phi = localBasis.evaluateFunction( pt.position )
        integrationElement = element.geometry.integrationElement(pt.position)
        for k in range(n):
            localB[k] += integrationElement * pt.weight * phi[k][0] * f(element.geometry.toGlobal(pt.position))
    return localA, localB


# c)
def globalAssembler(basis, f, l_param):
    
    N = len(basis)
    
    # create sparse matrix and dense vector (think about why we want a sparse structure)
    A = lil_matrix( (N,N) )
    b = np.zeros( N )
    print(A)
    # create a boolean array and interpolate it: then it will contain the Dirichlet information
    dirichletDOFs = np.zeros( N, dtype=bool )
    basis.interpolate(dirichletDOFs, isDirichlet ) # TODO use correct "isDirichlet" function
    
    # TODO interpolate g: g_h=\sum_i coefficients_i*basisfunction_i
    coeff = np.zeros(N)
    #basis.interpolate(coeff,g)
    # prepare a localView
    localView = basis.localView()
    
    # extract the grid from the basis
    grid = basis.gridView

    for element in grid.elements:
        
        # bind the local view to the current grid element:
        # then localViews knows everything about this element, as described in prog3
        localView.bind(element)
        
        # extract the local basis and compute the local matrix and vector
        localBasis = localView.tree().finiteElement().localBasis
        localA, localb = localAssembler(element,localBasis, f, l_param)
        n = localView.size()
        # use localView.index( ... ) for the local -> global index mapping!
        # use localView.size() for the total number of local basis functions
        for k in range(n):
            i = localView.index(k)[0]
            if dirichletDOFs[i]:
                A[i, i] = 1
                b[i] = coeff[i]
            else:
                b[i] += localb[k]
                for l in range(n):
                    j = localView.index(l)[0]
                    A[i, j] += localA[k][l]
        
    return A,b


if __name__ == "__main__":
    data_fitting()
