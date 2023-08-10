import numpy as np
from sympy import Add, poly, S
from roboticstoolbox.tools.trajectory import *

class ClippedTrajectory():
    def __init__(self, functions, T) -> None:
        self.functions = functions 
        self.T = T
        
    def __init__(self, start, goal, T) -> None:
        self.functions = [quintic_func(start[i], goal[i],T) for i in range(len(start))]
        self.T = T
    
    def __call__(self, n, t):
        '''Returns a tuple of three arrays (q qdot qdotdot)'''
        q_d = np.ndarray((n))
        qd_d = np.ndarray((n))
        qdd_d = np.ndarray((n))
        
        i = 0
        for f in self.functions:
            q,qd,qdd = f(min(t,self.T))
            q_d[i] = q; qd_d[i] = qd; qdd_d[i] = qdd
            i = i+1
        
        return q_d, qd_d, qdd_d

    def getTrajList(self):
        return self.functions
    
def sat(x, m=1):
    return min(max(-1, m*x), 1)

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])
    
def index2var(row, column, vars):
    return vars[row]*vars[column]

def coeff_dict(expr, *vars):
    collected = poly(expr, *vars).as_expr()
    i, d = collected.as_independent(*vars, as_Add=True)
    rv = dict(i.as_independent(*vars, as_Mul=True)[::-1] for i in Add.make_args(d))
    if i:
        assert 1 not in rv
        rv.update({S.One: i})
    return rv

def norm(v):
    sqnorm = float((v.T).dot(v))
    return np.sqrt(sqnorm)
def lambdaMin(matrix):
    eigenVal, _ = np.linalg.eig(matrix)
    return min(eigenVal)

def lambdaMax(matrix):
    eigenVal, _ = np.linalg.eig(matrix)
    return max(eigenVal)
