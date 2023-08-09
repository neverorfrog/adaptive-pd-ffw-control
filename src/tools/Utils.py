import numpy as np
from roboticstoolbox.tools.trajectory import *
from sympy import Add, poly, S

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