import time
import pprint
import numpy as np
import sympy
from roboticstoolbox.tools.trajectory import *

class ClippedTrajectory():
    def __init__(self, functions, T) -> None:
        self.functions = functions 
        self.T = T
        
    def __init__(self, start, goal, T) -> None:
        self.functions = [quintic_func(start[i], goal[i],T) for i in range(len(start))]
        self.T = T
        self.goal = goal
        self.start = start
    
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
    
class Profiler():
    logger = dict()
    meantimes = dict()
    occurences = dict()
    process_name = ""
    lastCheckpoint = 0

    def start(name):
        Profiler.lastCheckpoint = time.process_time()
        Profiler.process_name = name

    def stop():
        tmpCheckpoint = time.process_time()
        elapsed_time = tmpCheckpoint - Profiler.lastCheckpoint
        Profiler.logger[Profiler.process_name] = elapsed_time
        
        #Mean time stuff
        try:
            first_time = Profiler.meantimes[Profiler.process_name] is None
            current_mean = Profiler.meantimes[Profiler.process_name]
        except KeyError: #first time for the process
            Profiler.meantimes[Profiler.process_name] = 0
            Profiler.occurences[Profiler.process_name] = 0 
            
        current_mean = Profiler.meantimes[Profiler.process_name]
        Profiler.occurences[Profiler.process_name] += 1    
        i = Profiler.occurences[Profiler.process_name]
        Profiler.meantimes[Profiler.process_name] += (1/i)*(elapsed_time - current_mean)

    def print():
        pprint.pprint(f"{Profiler.logger:.3f}")
        
    def mean():
        print("MEANTIIIIIMES")
        pprint.pprint(Profiler.meantimes)
        
    
def sat(x, m=1):
    return min(max(-1, m*x), 1)

def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])
    
def index2var(row, column, vars):
    return vars[row]*vars[column]
  
def efficient_coeff_dict(expr, vars):
    poly = sympy.Poly(expr, vars)
    terms = poly.as_expr().as_ordered_terms()
    coeffs = poly.coeffs()

    d = dict()

    for i in range(len(terms)):
        mult = terms[i].args[0] if isinstance(terms[i].args[0], sympy.Pow) else terms[i].args[0]*terms[i].args[1]
        d[mult] = coeffs[i]
    
    return d

def norm(v):
    sqnorm = float((v.T).dot(v))
    return np.sqrt(sqnorm)

def lambdaMin(matrix):
    eigenVal, _ = np.linalg.eig(matrix)
    return min(eigenVal)

def lambdaMax(matrix):
    eigenVal, _ = np.linalg.eig(matrix)
    return max(eigenVal)


if __name__ == "__main__":
    d = dict()
    d["hi"] = 1
    print(d["hii"] is None)
