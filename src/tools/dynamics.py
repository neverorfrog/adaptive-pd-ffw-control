import numpy as np
import spatialmath.base.symbolic as sym
import sympy
from tools.utils import skew, Profiler, efficient_coeff_dict, index2var
from tools.robots import *
import os
from sympy import trigsimp, expand, nsimplify, evalf
from math import pi
from operator import itemgetter

class EulerLagrange():

    def __init__(self, robot, path = None, planar = False):
        '''
        pi for dynamic parameters that we believe are ground truth
        '''
        self.robot = robot
        self.n = len(robot.links)
        self.pi = sym.symbol(f"pi_(1:{10*self.n+1})")
        self.path = path
        self.planar = planar
        
        if path and os.path.isdir(path):
            print("Dynamic model cache found. Loading model...")
            self.M = np.load(open(os.path.join(path,"M.npy"),"rb"), allow_pickle=True)
            self.S = np.load(open(os.path.join(path,"S.npy"),"rb"), allow_pickle=True)
            self.g = np.load(open(os.path.join(path,"g.npy"),"rb"), allow_pickle=True)
            self.rindices = np.load(open(os.path.join(path,"rindices.npy"),"rb"), allow_pickle=True)
            self.kindices = np.load(open(os.path.join(path,"kindices.npy"),"rb"), allow_pickle=True)
            self.p = np.load(open(os.path.join(path,"p.npy"),"rb"), allow_pickle=True)
            self.Y = sympy.Matrix(np.load(open(os.path.join(path,"Y.npy"),"rb"), allow_pickle=True))
            self.robot.pi = np.delete(self.robot.pi, self.rindices)
            self.robot.realpi = np.delete(self.robot.realpi, self.rindices) 
        else:
            self.init()
        
    def init(self):
        n = self.n
        
        # link variables
        q = sym.symbol(f"q(1:{n+1})") 
        q_d = sym.symbol(f"q_dot_(1:{n+1})")

        #Kinetic and Potential energy
        Profiler.start("Moving Frames")
        T,U = self.movingFrames(self.robot)
        Profiler.stop()
        
        #Inertia Matrix
        Profiler.start("Inertia Matrix")
        self.M = self.computeInertia(T, q_d)
        Profiler.stop()
        
        #Coriolis and centrifugal terms and gravity terms
        Profiler.start("Coriolis and centrifugal")
        self.S, self.g = self.computeCoriolisGravity(self.M, U, q, q_d)
        Profiler.stop()
        
        Profiler.start("Regressor Matrix")
        self.Y = self.computeMinimalParametrization()
        Profiler.stop()

        if(self.path):
            os.mkdir(self.path)
            np.save(open(os.path.join(self.path,"M.npy"), "wb"), self.M)
            np.save(open(os.path.join(self.path,"S.npy"), "wb"), self.S)
            np.save(open(os.path.join(self.path,"g.npy"), "wb"), self.g)
            np.save(open(os.path.join(self.path,"Y.npy"), "wb"), self.Y)
            np.save(open(os.path.join(self.path,"rindices.npy"), "wb"), self.rindices)
            np.save(open(os.path.join(self.path,"kindices.npy"), "wb"), self.kindices)
            np.save(open(os.path.join(self.path,"p.npy"), "wb"), self.p)
            
    
    def computeMinimalParametrization(self):
        n = self.n
        model = sympy.Matrix(self.getDynamics())
        Y = model.jacobian(self.pi)
        #minimum number of parameters for each link (in self.p) and eliminating the zero columns
        self.p = np.zeros((n,), dtype = int) #number of minimal parameters for each link
        self.kindices = [] #the ones of the columns to keep
        self.rindices = [] #the ones of the columns to remove
        for i in range(n*10):
            if np.array(Y[:,i] == sympy.zeros(n,1)): 
                self.rindices.append(i)
            else:
                linksection = int(i/10)
                self.p[linksection] += 1
                self.kindices.append(i)    
        Y = np.delete(Y, self.rindices, 1)
        self.robot.pi = np.delete(self.robot.pi, self.rindices)
        self.robot.realpi = np.delete(self.robot.realpi, self.rindices) 
        return Y


    def computeInertia(self, T, q_d):
        n = self.n
        T = sympy.Poly(T.evalf(),q_d).as_dict()
        M = np.full((n,n), sym.zero())
        for row in range(n):
            for column in range(row*(n + 1), n*(row + 1)):
                key = [0] * n
                key[row] += 1 
                key[column - row*n] += 1
                try: mij = T[tuple(key)]
                except KeyError: continue
                M[row, column - row*n] = mij
                M[column-row*n, row] += mij
        return M
    
    def computeCoriolisGravity(self, M, U, q, q_d):
        n = self.n
        S = np.full((n,n), sym.zero(), dtype = object)
        g = np.full((n,), sym.zero(), dtype = object)
        M = sympy.Matrix(M)
        for i in range(n):
            C_temp = M[:,i].jacobian(q)
            C = 0.5 * (C_temp + C_temp.T - M.diff(q[i]))
            S[i] = np.matmul(q_d, C)
            g[i] = -U.diff(q[i])
        return S, g
    
                
    def movingFrames(self, robot):
        n = self.n
        gv = np.array([0, 9.81, 0]) if self.planar else np.array([0, 0, 9.81])
        ri = np.full((3,), sym.zero(), dtype = object) #vector from RF i-1 to i wrt RF i-1
        Rinv = np.full((3,3), sym.zero(), dtype = object) #ith matrix representing rotation from Rf i to Rf i-1
        iwi = np.full((3,), sym.zero(), dtype = object) #angular velocity of link i wrt RF i
        ivi = np.full((3,), sym.zero(), dtype = object) #linear velocity of link i wrt RF i
        
        T = 0 #total kinetic energy of the robot
        U = 0 #total potential energy of the robot
        
        q = sym.symbol(f"q(1:{n+1})")  # link variables
        q_d = sym.symbol(f"q_dot_(1:{n+1})")
        
        for i in range(n):            
            offset = 10*i
            sigma = int(robot.links[i].isprismatic) #check for prismatic joints
            A = robot[i].A(q[i]) #homogeneus transformation from frame i to i+1            
            ri = (A.t)
            Ainv = A.inv()
            Rinv = (Ainv.R) #rotation from frame i+1 to i
                                                    
            #Kinetic Energy
            im1wi = iwi + (1-sigma) * q_d[i] * np.array([0,0,1]) #omega of link i wrt RF i-1 (3 x 1) 
            iwi = trigsimp( nsimplify(Rinv @ im1wi, tolerance = 1e-10, rational = True) )
            im1vi = ivi + sigma * q_d[i] * np.array([0,0,1]) + np.cross(im1wi, ri) #linear v of link i wrt RF i-1
            ivi = trigsimp( nsimplify(Rinv @ im1vi, tolerance = 1e-10, rational = True) )
            mirci = np.array(self.pi[offset+1:offset+4])
            I_link = np.array([[self.pi[offset+4], self.pi[offset+5], self.pi[offset+6]],
                               [self.pi[offset+5], self.pi[offset+7], self.pi[offset+8]],
                               [self.pi[offset+6], self.pi[offset+8], self.pi[offset+9]]])
            
            first = 0.5 * self.pi[offset+0] * nsimplify(np.matmul(ivi,ivi).evalf(), tolerance = 1e-10, rational = True)
            second = nsimplify( np.matmul(np.matmul(mirci, skew(ivi)), iwi).evalf() , tolerance = 1e-10, rational = True)
            third = 0.5 * np.matmul(np.matmul(iwi, I_link), iwi) 
            
            Ti = first + second  + third
            T = T + Ti
                   
            #Potential Energy
            rot0i = robot.A(i,q).R #transformation from RF 0 to RF i+1
            r0i = robot.A(i,q).t
            Ui = -self.pi[offset+0]*np.matmul(gv,r0i) - np.matmul(gv, np.matmul(rot0i,mirci))
            U = U + Ui
            
        return T,U 
    
    def getDynamics(self):
        qdd = sym.symbol(f"q_dot_dot_(1:{self.n+1})")
        qd_S = sym.symbol(f"q_dot_S_(1:{self.n+1})")
        return np.matmul(self.M,qdd) + np.matmul(self.S,qd_S) + self.g
    
    # Return the linear parametrization Y matrix such that Y*pi = tau
    def getY(self, simplify = False):
        return sym.simplify(self.Y) if simplify else self.Y  
    
    def evaluateMatrixPi(self, matrix):
        n = self.n
        sympi = sym.symbol(f"pi_(1:{10*n+1})")                     
        matrix = sympy.Matrix(matrix)
        d = dict()
            
        for i in range(len(self.robot.pi)):
            d[sympi[self.kindices[i]]] = self.robot.pi[i]
                          
        return matrix.xreplace(d).evalf()

    def gravity(self, q):
        n = self.n
        gravity = self.evaluateMatrixPi(self.g)
        gravity = self.evaluateMatrix(gravity, q, [0]*n,[0]*n,[0]*n)
        return gravity
    
    def coriolis(self, q, qd):
        n = self.n
        coriolis = self.evaluateMatrixPi(self.S)
        coriolis = self.evaluateMatrix(coriolis, q, qd, [0]*n,[0]*n)
        return coriolis
    
    def inertia(self, q):
        n = self.n
        inertia = self.evaluateMatrixPi(self.M)
        inertia = self.evaluateMatrix(inertia, q, [0]*n, [0]*n,[0]*n)
        return inertia

    def evaluateMatrix(self, mat, q, qd, qd_S, qdd):
        mat = sympy.Matrix(mat)
        q_sym = sym.symbol(f"q(1:{self.n+1})") 
        qd_sym = sym.symbol(f"q_dot_(1:{self.n+1})")
        qdd_sym = sym.symbol(f"q_dot_dot_(1:{self.n+1})")
        qd_S_sym = sym.symbol(f"q_dot_S_(1:{self.n+1})")

        d = dict()

        for i in range(self.n):
            d[q_sym[i]] = q[i]
            d[qd_sym[i]] = qd[i]
            d[qd_S_sym[i]] = qd_S[i]
            d[qdd_sym[i]] = qdd[i]
        
        return mat.xreplace(d).evalf()
    
    def evaluateY(self, q, qd, qd_S, qdd):
        return self.evaluateMatrix(self.Y, q, qd, qd_S, qdd)
    
    def getChristoffel(self,k):
        n = self.n
        M = sympy.Matrix(self.evaluateMatrixPi(self.M))
        q = sym.symbol(f"q(1:{n+1})") 
        c = sympy.Matrix(np.ndarray((n,n)))

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                first = M[k,j].diff(q[i])
                second = M[k,i].diff(q[j])
                third = M[i,j].diff(q[k])

                c[i,j] = first+second-third
        return 0.5 * c
    

if __name__ == "__main__":
    robot = ParametrizedRobot(Polar2R())
    model = EulerLagrange(robot, os.path.join("src/models",robot.name))
    
    q = [pi/2,pi/2]
    qd = [0.3,0]
    qdd = [-0.1,0]
    
    Profiler.start("EVALUATION")
    my_M = model.inertia(q)
    c_M = robot.inertia(q)   
    print(f"ERROR M: {my_M - c_M}")
    print(f"ERROR NORM: {(my_M-c_M).norm()}")
    
    my = model.coriolis(q,qd)
    c = robot.coriolis(q,qd)   
    print(f"ERROR S: {my - c}")
    print(f"ERROR NORM: {(my-c).norm()}")
    
    my = model.gravity(q)
    c = robot.gravload(q).reshape(-1,1)  
    print(f"MY g: {my}")
    print(f"CORKE g: {c}")
    print(f"ERROR NORM: {(my-c).norm()}")
                            
    print(f"MY u: {model.evaluateY(q,qd,qd,qdd) @ robot.realpi}")
    print(f"CORKE u: {robot.inertia(q) @ qdd + robot.coriolis(q, qd) @ qd + robot.gravload(q, gravity = [0,0,-9.81])}")
    Profiler.stop()
    Profiler.print()
