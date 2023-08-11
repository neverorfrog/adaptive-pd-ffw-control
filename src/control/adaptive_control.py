import numpy as np
from tools.robots import *
from tools.control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *
import itertools

class AdaptiveControl(TrajectoryControl):
    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0]):
        super().__init__(robot, env, gravity)
        self.dynamicModel = dynamicModel

        n = len(self.robot.links)
        a_sym = sym.symbol(f"a(1:{n+1})")
        g0_sym = sym.symbol("g")

        self.pi_sym = sym.symbol(f"pi_(1:{10*n+1})")
        self.pi = np.zeros(10*n)
        self.beliefPi = np.zeros(10*n)

        dynamicModel.getY()
        dynamicModel.Y = dynamicModel.Y.subs(g0_sym, gravity[1])
        for i in range(n):
            shift = 10*i
            dynamicModel.Y = dynamicModel.Y.subs(a_sym[i], self.robot.links[i].a)

            I_link = self.robot.links[i].I + self.robot.links[i].m*np.matmul(skew(self.robot.links[i].r).T, skew(self.robot.links[i].r))
            mr = self.robot.links[i].m*self.robot.links[i].r #+ np.random.normal(0,1,3)

            #Computation of actual dynamic parameters
            self.pi[shift+0] = self.robot.links[i].m 
            self.beliefPi[shift+0] = self.pi[shift+0] #+ np.random.normal(0,10,1)

            self.pi[shift+1] = mr[0]
            self.beliefPi[shift+1] = self.pi[shift+1]

            self.pi[shift+2] = mr[1]
            self.beliefPi[shift+2] = self.pi[shift+2]

            self.pi[shift+3] = mr[2]
            self.beliefPi[shift+3] = self.pi[shift+3]

            self.pi[shift+4] = I_link[0,0] 
            self.beliefPi[shift+4] = self.pi[shift+4] #+ np.random.normal(0,10,1)

            self.pi[shift+5] = 0
            self.beliefPi[shift+5] = self.pi[shift+5]

            self.pi[shift+6] = 0
            self.beliefPi[shift+6] = self.pi[shift+6]

            self.pi[shift+7] = I_link[1,1]
            self.beliefPi[shift+7] = self.pi[shift+7] #+ np.random.normal(0,10,1)

            self.pi[shift+8] = 0
            self.beliefPi[shift+8] = self.pi[shift+8]

            self.pi[shift+9] = I_link[2,2]
            self.beliefPi[shift+9] = self.pi[shift+9] #+ np.random.normal(0,10,1)
        

class Adaptive_Facile(AdaptiveControl):
    
    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0]):
        super().__init__(robot, env, dynamicModel, gravity)
        
    def feedback(self):
        
        n = len(self.robot.links)
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration and error
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        gainMatrix = np.linalg.inv(self.kd) @ self.kp    
        
        #Error     
        e = q_d - q
        ed = qd_d - qd
        qd_r = qd_d + gainMatrix @ e
        ed_r = qd_r - qd #modified velocity error
        qdd_r = qdd_d + gainMatrix @ ed
        arrived = self.check_termination(e,ed)
        
        #Update Law
        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_r, qdd_r)
        gainMatrix = np.eye(n*10) # TODO: make this a parameter
        deltaPi = (gainMatrix @ Y.T) @ ed_r
        self.beliefPi = self.beliefPi + deltaPi
        
        #Torque computation
        torque = self.kp @ e + self.kd @ ed + np.matmul(Y, self.beliefPi).astype(np.float64)
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)
                
        return torque, arrived
        

class Adaptive_FFW(AdaptiveControl):

    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0]):
        super().__init__(robot, env, dynamicModel, gravity)

    def checkGains(self, q_d, qd_d, qdd_d):

        n = len(self.robot.links)

        M = self.dynamicModel.getrealM(self.robot, self.beliefPi)
        M = sympy.Matrix(M)
        g = self.dynamicModel.getrealG(self.robot, self.beliefPi)
        g = sympy.Matrix(g)        

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
            c_i = self.dynamicModel.get_christoffel(i, self.robot, self.beliefPi)
            diffG = g.diff(q[i])
            diffc_i = c_i.diff(q[i])
            for configuration in itertools.product(*tmp_cand):
                configuration = list(configuration)
                #Evaluate differentiated matrices for every possible tuple of candidates
                evM = self.dynamicModel.evaluateMatrix(M, configuration, [0]*n, [0]*n, [0]*n)
                evdiffM = self.dynamicModel.evaluateMatrix(diffM, configuration, [0]*n, [0]*n, [0]*n)
                evC = self.dynamicModel.evaluateMatrix(c_i, configuration, [0]*n, [0]*n, [0]*n)
                evdiffC = self.dynamicModel.evaluateMatrix(diffc_i, configuration, [0]*n, [0]*n, [0]*n)
                evG = self.dynamicModel.evaluateMatrix(g, configuration, [0]*n, [0]*n, [0]*n)
                evdiffG = self.dynamicModel.evaluateMatrix(diffG, configuration, [0]*n, [0]*n, [0]*n)
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
        qdd_N_MW = math.sqrt(np.matmul(np.matmul(qdd_d.T, M), qdd_d))
        qd_N_MW_sq = np.matmul(np.matmul(qd_d.T, M), qd_d)

        m = math.sqrt(n) * (kg + km*qdd_N_MW + kc2*qd_N_MW_sq)
        delta = 1 # TODO: correct?
        p = m+delta

        epsilon = 2*k1/kg + 2*k2/kM + 2*kc1/kc2
        
        #print(p**2 * lambdaMax(M))
        print(p**2)
        #print(2*(math.sqrt(n)*kc1*p*epsilon + p*lambdaMax(M) + kc1*math.sqrt(qd_N_MW_sq)))
        #print(3*p + (p*(2*kc1*math.sqrt(qd_N_MW_sq) + 2*lambdaMax(self.kd) + 3)**2)/(2*lambdaMin(self.kd)))

        # Condition 22 
        #assert( lambdaMin(self.kp) > p**2 * lambdaMax(M) )

        # Condition 33
        #assert( lambdaMin(self.kd) > 2*(math.sqrt(n)*kc1*p*epsilon + p*lambdaMax(M) + kc1*math.sqrt(qd_N_MW_sq)) )

        # Condition 34
        #assert( lambdaMin(self.kp) > 3*p + (p*(2*kc1*math.sqrt(qd_N_MW_sq) + 2*lambdaMax(self.kd) + 3)**2)/(2*lambdaMin(self.kd)) )
        '''
        
    def feedback(self):
        '''Computes the torque necessary to follow the reference trajectory'''

        n = len(self.robot.links)

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        self.checkGains(q_d, qd_d, qdd_d)
        
        exit(0)

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)

        actualY = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)
        torque = self.kp @ e + self.kd @ ed + np.matmul(actualY, self.beliefPi).astype(np.float64)

        # Update rule
        gainMatrix = np.eye(n*10) # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix @ (actualY.T @ (sat_e+ed))
        #self.beliefPi = self.beliefPi + deltaPi

        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived
    
if __name__ == "__main__":
    
    #robot and environment creation
    n = 2
    robot = TwoLink()
    env = PyPlot()
    goal = [pi/2,pi/2]
    
    T = 3
    traj = ClippedTrajectory(robot.q, goal, T)

    symrobot = SymbolicPlanarRobot(n)
    model = EulerLagrange(symrobot)
        
    # loop = Adaptive_Facile(robot, env, model, [0,-9.81,0])
    loop = Adaptive_FFW(robot, env, model, [0,-9.81,0])

    loop.setR(reference = traj, goal = goal, threshold = 0.05)
    loop.setK(kp = [2e3,1e3], kd = [150,50])
    
    loop.simulate(dt = 0.01)
    loop.plot()