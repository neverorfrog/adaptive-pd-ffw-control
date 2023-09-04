import itertools
import time
import pprint
import numpy as np
import spatialmath.base.symbolic as sym
import sympy
from roboticstoolbox.tools.trajectory import *
from math import pi

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
        
        for i,f in enumerate(self.functions):
            q,qd,qdd = f(min(t,self.T))
            q_d[i] = q; qd_d[i] = qd; qdd_d[i] = qdd
        
        return q_d, qd_d, qdd_d

    def getTrajList(self):
        return self.functions

class ExcitingTrajectory():
    MAGIC_OVERFLOW = 7
    def __init__(self, params, T) -> None:
        assert(len(params[0]) == 4)
        self.functions = [self.fz(el) for el in params]
        self.T = T

    def fz(self, param):
        c1, a, c2, omega = param
        return lambda t: (
            c1*(1-np.exp(-a*pow(t,3))) + c2*(1-np.exp(-a*pow(t,3)))*np.sin(omega*t),
            3*a*c1*pow(t,2)*np.exp(-a*pow(t,3)) - c2*omega*np.cos(t*omega)*(np.exp(-a*pow(t,3)) - 1) + 3*a*c2*pow(t,2)*np.exp(-a*pow(t,3))*np.sin(t*omega),
            np.exp(-a*pow(t,3))*(c2*pow(omega,2)*np.sin(t*omega) - 9*pow(a,2)*c1*pow(t,4) + 6*a*c1*t - 9*pow(a,2)*c2*pow(t,4)*np.sin(t*omega) - c2*pow(omega,2)*np.exp(a*pow(t,3))*np.sin(t*omega) + 6*a*c2*t*np.sin(t*omega) + 6*a*c2*pow(t,2)*omega*np.cos(t*omega))
        )

    def __call__(self, n, t):

        q_d = np.ndarray((n))
        qd_d = np.ndarray((n))
        qdd_d = np.ndarray((n))

        #overflow inversion
        dec = math.floor(t / ExcitingTrajectory.MAGIC_OVERFLOW) & 1
        rem = t % ExcitingTrajectory.MAGIC_OVERFLOW
        if t > ExcitingTrajectory.MAGIC_OVERFLOW:
            t = ExcitingTrajectory.MAGIC_OVERFLOW-rem if dec else rem

        
        for i,f in enumerate(self.functions):
            q,qd,qdd = f(t)
            q_d[i] = q; qd_d[i] = qd; qdd_d[i] = qdd
        
        return q_d, qd_d, qdd_d

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
        pprint.pprint(Profiler.logger)
        
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


def checkGains(dynamicModel, robot, kp, kd):
    
    n = len(robot.links)

    M = sympy.Matrix(dynamicModel.evaluateMatrixPi(dynamicModel.M, robot.pi))
    g = sympy.Matrix(dynamicModel.evaluateMatrixPi(dynamicModel.g, robot.pi))      
    q = sym.symbol(f"q(1:{n+1})")  # link variables
    candidates = [0, pi/2, pi, 3/2*pi]
    tmp_cand = np.repeat([candidates],n,axis=0)
    km = 0
    kc1 = 0
    kc2 = 0
    kg = 0
    k1 = 0
    k2 = 0

    for i in range(n):
        diffM = M.diff(q[i])
        c_i = dynamicModel.getChristoffel(i)
        c_i = dynamicModel.getChristoffel(i)
        diffG = g.diff(q[i])
        diffc_i = c_i.diff(q[i])
        for configuration in itertools.product(*tmp_cand):
            configuration = list(configuration)
            #Evaluate differentiated matrices for every possible tuple of candidates
            evM = dynamicModel.evaluateMatrix(M, configuration, [0]*n, [0]*n, [0]*n)
            evdiffM = dynamicModel.evaluateMatrix(diffM, configuration, [0]*n, [0]*n, [0]*n)
            evC = dynamicModel.evaluateMatrix(c_i, configuration, [0]*n, [0]*n, [0]*n)
            evdiffC = dynamicModel.evaluateMatrix(diffc_i, configuration, [0]*n, [0]*n, [0]*n)
            evG = dynamicModel.evaluateMatrix(g, configuration, [0]*n, [0]*n, [0]*n)
            evdiffG = dynamicModel.evaluateMatrix(diffG, configuration, [0]*n, [0]*n, [0]*n)
            evdiffM = np.abs(evdiffM)
            evC = np.abs(evC)
            evdiffC = np.abs(evdiffC)
            evdiffG = np.abs(evdiffG)
            evG = norm(evG)
            #Max of evaluated matrices
            km = max(km, np.max(evdiffM))
            kc1 = max(kc1, np.max(evC))
            kc2 = max(kc2, np.max(evdiffC))
            kg = max(kg, np.max(evdiffG))
            k1 = max(k1, np.max(evG))
            k2 = max(k2, lambdaMax(np.array(evM,dtype=float)))
            
    
    km *= n**2
    kc1 *= n**2 
    kc2 *= n**3
    kg *= n 

    '''
    BEST = []
    maxr = 0
    for q1 in np.arange(-pi, pi, 0.5):
        for q2 in np.arange(-pi, pi, 0.5):

            for qq1 in np.arange(-pi, pi, 0.5):
                for qq2 in np.arange(-pi, pi, 0.5):

                    for qqq1 in np.arange(-pi, pi, 0.5):
                        for qqq2 in np.arange(-pi, pi, 0.5):
                            print(q1,q2,qq1,qq2,maxr)
                            if(np.linalg.norm(np.array([q1,q2]) - np.array([qq1,qq2]))*np.linalg.norm(np.array([qqq1,qqq2])) == 0):
                                continue
                            c = self.robot.inertia()
                            ratio = np.linalg.norm(np.matmul(c, np.array([qqq1,qqq2])))/ (np.linalg.norm(np.array([qq1,qq2]))*np.linalg.norm(np.array([qqq1,qqq2])))
                            if ratio > maxr:
                                maxr = ratio
                                BEST = [q1,q2,qq1,qq2,qqq1,qqq2]

    print(f"WITH ITERATIVE METHOD: {maxr} WITH CLOSED METHOD: {kc1} BEST: {BEST}")
    '''

    M = np.array(dynamicModel.inertia(robot.q), dtype=float) 
    qdd_bound = 1
    qd_bound = 1
    m = math.sqrt(n) * (kg + km*qdd_bound + kc2*(qd_bound**2))
    print(m)
    delta = 1 
    p = m + delta
    
    epsilon = 2*k1/kg + 2*k2/km + 2*kc1/kc2
            
    condition22 = p**2 * lambdaMax(M)
    condition33 = 2*(math.sqrt(n)*kc1*p*epsilon + p*lambdaMax(M) + kc1*math.sqrt(qd_bound))
    condition34 = 3*p + (p * (2*kc1*math.sqrt(qd_bound) + lambdaMax(kd) + 3)**2)/(2*lambdaMin(kd))
    
    print(f"condition22: {condition22}")
    print(f"condition33: {condition33}")
    print(f"condition34: {condition34}")

    # Condition 22 
    assert(lambdaMin(kp) > condition22)
    print("Condition 22 passed")

    # Condition 33
    assert(lambdaMin(kd) >  condition33)
    print("Condition 33 passed")

    # Condition 34
    assert(lambdaMin(kp) > condition34)
    print("Condition 34 passed")


if __name__ == "__main__":
    d = dict()
    d["hi"] = 1
    print(d["hii"] is None)
