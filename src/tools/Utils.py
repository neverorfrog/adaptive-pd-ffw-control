import numpy as np
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