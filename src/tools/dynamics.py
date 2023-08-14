import numpy as np
import spatialmath.base.symbolic as sym
import sympy
from tools.utils import skew, index2var, coeff_dict, Profiler, efficient_coeff_dict
from tools.robots import *
class EulerLagrange():

    def __init__(self, robot):
        '''
        realrobot for kinematic parameters
        pi for dynamic parameters that we believe are ground truth
        '''
        self.profiler = Profiler()
        self.robot = robot
        self.n = len(robot.links); n = self.n
        self.Y = None
        self.dynamicModel = None
        
        #kinematic parameters
        q = sym.symbol(f"q(1:{n+1})")  # link variables
        q_d = sym.symbol(f"q_dot_(1:{n+1})")
        
        #dynamic parameters
        self.pi = sym.symbol(f"pi_(1:{10*n+1})")

        #Kinetic and Potential energy
        self.profiler.start("Moving Frames")
        T,U = self.movingFrames(robot)
        self.profiler.stop()
        
        #Inertia Matrix
        self.profiler.start("Inertia Matrix")

        coeffs = efficient_coeff_dict(T, q_d)
        
        self.M = np.full((n,n), sym.zero(), dtype = object)
        for row in range(n):
            self.M[row,:] = [coeffs[index2var(row,column,q_d)] for column in range(n)]
        self.M = (np.ones((n,n))+np.diag([1]*n))*self.M
        self.profiler.stop()

        #Coriolis and centrifugal terms and gravity terms
        self.profiler.start("Coriolis and centifugal")
        self.S = np.full((n,), sym.zero(), dtype = object)
        self.g = np.full((n,), sym.zero(), dtype = object)
        M = sympy.Matrix(self.M)
        for i in range(n):
            C_temp = M[:,i].jacobian(q)
            C = 0.5 * (C_temp + C_temp.T - M.diff(q[i]))
            self.S[i] = np.matmul(q_d, C)
            self.g[i] = -U.diff(q[i])
        self.profiler.stop()
        
        self.profiler.start("Regressor Matrix")
        symModel = sympy.Matrix(self.getDynamicModel())
        self.Y = symModel.jacobian(self.pi)
        self.profiler.stop()

        self.profiler.print()
        
        
    def setDynamics(self, realrobot, pi):
        self.realrobot = realrobot
        self.pi = pi
        self.inertia_generic = self.getM(realrobot, pi)
        self.coriolis_generic = self.getS(realrobot, pi)
        self.gravity_generic = self.getG(realrobot, pi)
            
            
    def movingFrames(self, robot):
        n = self.n
        g0 = sym.symbol("g")
        gv = np.array([0, -g0, 0]) 
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
    
    def getG(self, robot, pi):
        n = self.n
        gravity = sympy.Matrix(self.g)
        a = sym.symbol(f"a(1:{n+1})") 
        g0 = sym.symbol("g")
        sympi = sym.symbol(f"pi_(1:{10*n+1})")

        for i in range(n):
            shift = 10*i
            gravity = gravity.subs(sympi[shift+0], pi[shift+0])
            gravity = gravity.subs(sympi[shift+1], pi[shift+1])
            gravity = gravity.subs(sympi[shift+2], pi[shift+2])
            gravity = gravity.subs(sympi[shift+3], pi[shift+3])
            gravity = gravity.subs(a[i], robot.links[i].a)
        gravity = gravity.subs(g0, -9.81)
        return gravity
    
    def getS(self, robot, pi):
        n = self.n
        coriolis = sympy.Array(self.S)
        a = sym.symbol(f"a(1:{n+1})") 
        g0 = sym.symbol("g")
        sympi = sym.symbol(f"pi_(1:{10*n+1})")

        for i in range(n):
            shift = 10*i
            coriolis = coriolis.subs(sympi[shift+0], pi[shift+0])
            coriolis = coriolis.subs(sympi[shift+1], pi[shift+1])
            coriolis = coriolis.subs(sympi[shift+2], pi[shift+2])
            coriolis = coriolis.subs(sympi[shift+3], pi[shift+3])
            coriolis = coriolis.subs(sympi[shift+4], pi[shift+4])
            coriolis = coriolis.subs(sympi[shift+7], pi[shift+7])
            coriolis = coriolis.subs(sympi[shift+9], pi[shift+9])
            coriolis = coriolis.subs(a[i], robot.links[i].a)
        coriolis = coriolis.subs(g0, -9.81)
        return coriolis
        
    def getM(self, robot, pi):
        n = self.n
        inertia = sympy.Matrix(self.M)
        a = sym.symbol(f"a(1:{n+1})") 
        g0 = sym.symbol("g")
        sympi = sym.symbol(f"pi_(1:{10*n+1})")

        for i in range(n):
            shift = 10*i
            inertia = inertia.subs(sympi[shift+0], pi[shift+0])
            inertia = inertia.subs(sympi[shift+1], pi[shift+1])
            inertia = inertia.subs(sympi[shift+2], pi[shift+2])
            inertia = inertia.subs(sympi[shift+3], pi[shift+3])
            inertia = inertia.subs(sympi[shift+4], pi[shift+4])
            inertia = inertia.subs(sympi[shift+7], pi[shift+7])
            inertia = inertia.subs(sympi[shift+9], pi[shift+9])
            inertia = inertia.subs(a[i], robot.links[i].a)
        inertia = inertia.subs(g0, -9.81)
        return inertia
    
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
        inertia = self.inertia_generic
        for i in range(n):
            inertia = inertia.subs(q_sym[i], q[i])
        return inertia.evalf()      

    def evaluateMatrix(self, mat, q, qd, qd_S, qdd):
        actualMatrix = mat
        q_sym = sym.symbol(f"q(1:{self.n+1})")  # link variables
        qd_sym = sym.symbol(f"q_dot_(1:{self.n+1})")
        qdd_sym = sym.symbol(f"q_dot_dot_(1:{self.n+1})")
        qd_S_sym = sym.symbol(f"q_dot_S_(1:{self.n+1})")

        for i in range(self.n):
            actualMatrix = actualMatrix.subs(q_sym[i], q[i])
            actualMatrix = actualMatrix.subs(qd_sym[i], qd[i])
            actualMatrix = actualMatrix.subs(qd_S_sym[i], qd_S[i])
            actualMatrix = actualMatrix.subs(qdd_sym[i], qdd[i])
        
        return actualMatrix
    
    def evaluateY(self, q, qd, qd_S, qdd):
        actualY = self.Y
        q_sym = sym.symbol(f"q(1:{self.n+1})")  # link variables
        qd_sym = sym.symbol(f"q_dot_(1:{self.n+1})")
        qdd_sym = sym.symbol(f"q_dot_dot_(1:{self.n+1})")
        qd_S_sym = sym.symbol(f"q_dot_S_(1:{self.n+1})")

        for i in range(self.n):
            actualY = actualY.subs(q_sym[i], q[i])
            actualY = actualY.subs(qd_sym[i], qd[i])
            actualY = actualY.subs(qd_S_sym[i], qd_S[i])
            actualY = actualY.subs(qdd_sym[i], qdd[i])
        
        return actualY
    

if __name__ == "__main__":
    symrobot = SymbolicPlanarRobot(2)
    model = EulerLagrange(symrobot)
            
    