import numpy as np
import spatialmath.base.symbolic as sym
import sympy
from tools.utils import skew, Profiler, efficient_coeff_dict, index2var
from tools.robots import *
import os
import itertools

class EulerLagrange():

    def __init__(self, robot, path = None, planar = False):
        '''
        realrobot for kinematic parameters
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
            self.Y = sympy.Matrix(np.load(open(os.path.join(path,"Y.npy"),"rb"), allow_pickle=True))
        else:
            self.init()
        
    def init(self):
        self.Y = None
        self.dynamicModel = None
        n = self.n
        
        #kinematic parameters
        q = sym.symbol(f"q(1:{n+1})")  # link variables
        q_d = sym.symbol(f"q_dot_(1:{n+1})")
        
        #dynamic parameters
        self.pi = sym.symbol(f"pi_(1:{10*n+1})")

        #Kinetic and Potential energy
        Profiler.start("Moving Frames")
        T,U = self.movingFrames(self.robot)
        Profiler.stop()
        
        #Inertia Matrix
        Profiler.start("Inertia Matrix")
        self.M = self.computeInertia(T, q_d)
        Profiler.stop()
        
        # Profiler.start("Inertia Matrix")
        # coeffs = efficient_coeff_dict(T, q_d)
        # self.M = np.full((n,n), sym.zero(), dtype = object)
        # for row in range(n):
        #     self.M[row,:] = [coeffs[index2var(row,column,q_d)] for column in range(n)]
        # self.M = (np.ones((n,n))+np.diag([1]*n))*self.M
        # Profiler.stop()
        
        #Coriolis and centrifugal terms and gravity terms
        Profiler.start("Coriolis and centrifugal")
        self.S = np.full((n,n), sym.zero(), dtype = object)
        self.g = np.full((n,), sym.zero(), dtype = object)
        M = sympy.Matrix(self.M)
        for i in range(n):
            C_temp = M[:,i].jacobian(q)
            C = 0.5 * (C_temp + C_temp.T - M.diff(q[i]))
            self.S[i] = np.matmul(q_d, C)
            self.g[i] = -U.diff(q[i])
        Profiler.stop()
        
        Profiler.start("Regressor Matrix")
        symModel = sympy.Matrix(self.getDynamicModel())
        self.Y = symModel.jacobian(self.pi)
        Profiler.stop()

        if(self.path):
            os.mkdir(self.path)
            np.save(open(os.path.join(self.path,"M.npy"), "wb"), self.M)
            np.save(open(os.path.join(self.path,"S.npy"), "wb"), self.S)
            np.save(open(os.path.join(self.path,"g.npy"), "wb"), self.g)
            np.save(open(os.path.join(self.path,"Y.npy"), "wb"), self.Y)

    def computeInertia(self, T, q_d):
        n = self.n
        terms = sympy.Poly(T,q_d).coeffs()
        M = np.full((n,n), sym.zero())
        offset = 0
        for row in range(n): 
            offset += row 
            for column in range(row*(n + 1), n*(row + 1)):
                mij =  terms[column - offset]
                M[row, column - row*n] = mij
                M[column-row*n, row] += mij
        return M
    
    def setDynamics(self, realrobot, pi):
        self.realrobot = realrobot
        self.pi = pi
        g = self.g.reshape(-1,1)
                        
        self.inertia_generic = self.evaluateMatrixPi(sympy.Matrix(self.M).as_mutable(), realrobot, pi)
        self.coriolis_generic = self.evaluateMatrixPi(sympy.Matrix(self.S).as_mutable(), realrobot, pi)
        self.gravity_generic = self.evaluateMatrixPi(sympy.Matrix(g).as_mutable(), realrobot, pi)
            
            
    def movingFrames(self, robot):
        n = self.n
        g0 = sym.symbol("g")
        gv = np.array([0, -g0, 0]) if self.planar else np.array([0, 0, -g0])
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
            ri = A.t
            Ainv = A.inv()
            Rinv = Ainv.R #rotation from frame i+1 to i
                
            #Kinetic Energy
            im1wi = iwi + (1-sigma) * q_d[i] * np.array([0,0,1]) #omega of link i wrt RF i-1 (3 x 1) 
            iwi = Rinv @ im1wi
            im1vi = ivi + sigma * q_d[i] * np.array([0,0,1]) + np.cross(im1wi, ri) #linear v of link i wrt RF i-1
            ivi = Rinv @ im1vi; 
            mirci = np.array(self.pi[offset+1:offset+4])
            I_link = np.diag([self.pi[offset+4], self.pi[offset+7], self.pi[offset+9]])
            Ti = 0.5*self.pi[offset+0]*np.matmul(ivi.T, ivi) + np.matmul(np.matmul(mirci, skew(ivi)), iwi) + 0.5*np.matmul(np.matmul(iwi, I_link), iwi)
            T = T + Ti
            
            #Potential Energy
            rot0i = robot.A(i,q).R #transformation from RF 0 to RF i+1
            r0i = robot.A(i,q).t
            Ui = -self.pi[offset+0]*np.matmul(gv,r0i) - np.matmul(gv, np.matmul(rot0i,mirci))
            U = U + Ui
        
        return T,U
        
    
    def getDynamicModel(self):
        qdd = sym.symbol(f"q_dot_dot_(1:{self.n+1})")
        qd_S = sym.symbol(f"q_dot_S_(1:{self.n+1})")
        return np.matmul(self.M,qdd) + np.matmul(self.S,qd_S) + self.g
    
    # Return the linear parametrization Y matrix such that Y*pi = tau
    def getY(self, simplify = False):
        return sym.simplify(self.Y) if simplify else self.Y
    
    def evaluateMatrixPi(self, matrix, robot, pi):
        n = self.n
        a = sym.symbol(f"a(1:{n+1})") 
        g0 = sym.symbol("g")
        sympi = sym.symbol(f"pi_(1:{10*n+1})")
        
        d = dict()

        for i in range(self.n):
            shift = 10*i
            d[sympi[shift+0]] = pi[shift+0]
            d[sympi[shift+1]] = pi[shift+1]
            d[sympi[shift+2]] = pi[shift+2]
            d[sympi[shift+3]] = pi[shift+3]
            d[sympi[shift+4]] = pi[shift+4]
            d[sympi[shift+7]] = pi[shift+7]
            d[sympi[shift+9]] = pi[shift+9]
            
        return matrix.xreplace(d).evalf()
    
    def getChristoffel(self,k,robot,pi):
        n = self.n
        M = sympy.Matrix(self.getM(robot, pi))
        q = sym.symbol(f"q(1:{n+1})") 
        c = sympy.Matrix(np.ndarray((n,n)))

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                first = M[k,j].diff(q[i])
                second = M[k,i].diff(q[j])
                third = M[i,j].diff(q[k])

                c[i,j] = first+second-third
        return 0.5 * c

    def gravity(self, q):
        assert(self.realrobot is not None)
        n = len(self.realrobot.links)
        q_sym = sym.symbol(f"q(1:{n+1})")  # link variable
        gravity = self.gravity_generic
        for i in range(n):
            gravity = gravity.subs(q_sym[i], q[i])
        return gravity.evalf() 
    
    def coriolis(self, q):
        assert(self.realrobot is not None)
        n = len(self.realrobot.links)
        q_sym = sym.symbol(f"q(1:{n+1})")  # link variable
        coriolis = self.coriolis_generic
        for i in range(n):
            coriolis = coriolis.subs(q_sym[i], q[i])
        return coriolis.evalf()
    
    def inertia(self, q):
        assert(self.realrobot is not None)
        n = len(self.realrobot.links)
        q_sym = sym.symbol(f"q(1:{n+1})")  # link variable
        a = sym.symbol(f"a(1:{n+1})") 
        inertia = self.inertia_generic
        d = dict()

        for i in range(self.n):
            d[q_sym[i]] = q[i]
            d[a[i]] = self.realrobot.links[i].a
        
        return inertia.xreplace(d).evalf()    

    def evaluateMatrix(self, mat, q, qd, qd_S, qdd):
        q_sym = sym.symbol(f"q(1:{self.n+1})")  # link variables
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
    

if __name__ == "__main__":
    symrobot = SymbolicPlanarRobot(2)
    model = EulerLagrange(symrobot)
    