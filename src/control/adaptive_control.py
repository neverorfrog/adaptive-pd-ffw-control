import numpy as np
from tools.robots import *
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *
import itertools

class AdaptiveControl(TrajectoryControl):
    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0], plotting = True):
        super().__init__(robot, env, gravity, plotting)
        self.dynamicModel = dynamicModel

        n = len(self.robot.links)
        a_sym = sym.symbol(f"a(1:{n+1})")
        g0_sym = sym.symbol("g")

        self.pi_sym = sym.symbol(f"pi_(1:{10*n+1})")
        self.pi = np.zeros(10*n)
        self.beliefPi = np.zeros(10*n)

        Profiler.start("Y gravity substitution")
        dynamicModel.getY()
        gravIdx = np.nonzero(gravity)[0][0]
        d = dict()
        d[g0_sym] = gravity[gravIdx]

        Profiler.stop()
        piShift_stdev = 20
        piShift_mean = 0


        for i in range(n):
            Profiler.start(f"subs {i} stage")

            shift = 10*i
            d[a_sym[i]] = self.robot.links[i].a

            I_link = self.robot.links[i].I + self.robot.links[i].m*np.matmul(skew(self.robot.links[i].r).T, skew(self.robot.links[i].r))
            mr = self.robot.links[i].m*self.robot.links[i].r 

            #Computation of actual dynamic parameters
            self.pi[shift+0] = self.robot.links[i].m 
            self.beliefPi[shift+0] = max(0, self.pi[shift+0] + np.random.normal(piShift_mean,piShift_stdev,1))

            self.pi[shift+1] = mr[0]
            self.beliefPi[shift+1] = self.pi[shift+1] + np.random.normal(piShift_mean,piShift_stdev,1)

            self.pi[shift+2] = mr[1]
            self.beliefPi[shift+2] = self.pi[shift+2] + np.random.normal(piShift_mean,piShift_stdev,1)

            self.pi[shift+3] = mr[2]
            self.beliefPi[shift+3] = self.pi[shift+3] + np.random.normal(piShift_mean,piShift_stdev,1)

            self.pi[shift+4] = I_link[0,0] 
            self.beliefPi[shift+4] = max(0, self.pi[shift+4] + np.random.normal(piShift_mean,piShift_stdev,1))

            self.pi[shift+5] = I_link[0,1] 
            self.beliefPi[shift+5] = self.pi[shift+5] + np.random.normal(piShift_mean,piShift_stdev,1)

            self.pi[shift+6] = I_link[0,2] 
            self.beliefPi[shift+6] = self.pi[shift+6] + np.random.normal(piShift_mean,piShift_stdev,1)

            self.pi[shift+7] = I_link[1,1]
            self.beliefPi[shift+7] = max(0,self.pi[shift+7] + np.random.normal(piShift_mean,piShift_stdev,1))

            self.pi[shift+8] = I_link[1,2] 
            self.beliefPi[shift+8] = max(0,self.pi[shift+8] + np.random.normal(piShift_mean,piShift_stdev,1))

            self.pi[shift+9] = I_link[2,2]
            self.beliefPi[shift+9] = max(0,self.pi[shift+9] + np.random.normal(piShift_mean,piShift_stdev,1))
            Profiler.stop()

        dynamicModel.Y = dynamicModel.Y.xreplace(d)
        #dynamicModel.setDynamics(robot, self.beliefPi) # TODO: update at every cycle
        Profiler.print()
        

class Adaptive_Facile(AdaptiveControl):
    
    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0], plotting = True):
        super().__init__(robot, env, dynamicModel, gravity, plotting)
        
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
        gainMatrix = np.eye(n*10) * 0.01 # TODO: make this a parameter
        deltaPi = (gainMatrix @ Y.T) @ ed_r
        self.beliefPi = self.beliefPi + deltaPi
        
        #Torque computation        
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.beliefPi).astype(np.float64) # TODO CAMBIAAA
        
        bound = 700
        torque = np.clip(torque, -bound, bound)
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)
                
        print(f"u: {torque}")
        return torque, arrived
        

class Adaptive_FFW(AdaptiveControl):

    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0], plotting = True):
        super().__init__(robot, env, dynamicModel, gravity, plotting)

    def checkGains(self, q_d, qd_d, qdd_d):

        n = len(self.robot.links)

        M = self.dynamicModel.getM(self.robot, self.beliefPi)
        M = sympy.Matrix(M)
        g = self.dynamicModel.getG(self.robot, self.beliefPi) # TODO CAMBIAAA
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
            c_i = self.dynamicModel.getChristoffel(i, self.robot, self.beliefPi)
            c_i = self.dynamicModel.getChristoffel(i, self.robot, self.beliefPi)
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

        M = np.array(self.dynamicModel.inertia(self.robot.q), dtype=float) 
        qdd_bound = 1
        qd_bound = 1
        m = math.sqrt(n) * (kg + km*qdd_bound + kc2*(qd_bound**2))
        delta = 1 
        p = m + delta
        
        epsilon = 2*k1/kg + 2*k2/km + 2*kc1/kc2
                
        condition22 = p**2 * lambdaMax(M)
        condition33 = 2*(math.sqrt(n)*kc1*p*epsilon + p*lambdaMax(M) + kc1*math.sqrt(qd_bound))
        condition34 = 3*p + (p * (2*kc1*math.sqrt(qd_bound) + lambdaMax(self.kd) + 3)**2)/(2*lambdaMin(self.kd))
        
        print(f"condition22: {condition22}")
        print(f"condition33: {condition33}")
        print(f"condition34: {condition34}")
        

        # Condition 22 
        # assert(lambdaMin(self.kp) > condition22)
        # print("Condition 22 passed")

        # Condition 33
        # assert(lambdaMin(self.kd) >  condition33)
        # print("Condition 33 passed")

        # Condition 34
        # assert(lambdaMin(self.kp) > condition34)
        # print("Condition 34 passed")
        
             
    def feedback(self):
        '''Computes the torque necessary to follow the reference trajectory'''
        n = len(self.robot.links)

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)

        Profiler.start("feedback loop")
        actualY = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)

        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(actualY, self.beliefPi).astype(np.float64)

        bound = 700
        torque = np.clip(torque, -bound, bound)

        # Update rule
        gainMatrix = np.eye(n*10) * 0.1 # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix @ (actualY.T @ (sat_e+ed))
        self.beliefPi = self.beliefPi + deltaPi
        Profiler.stop()
        
        print("Feedback loop took {:.2f} s\nCurrent Error: {:.2f}".format(Profiler.logger["feedback loop"], np.linalg.norm(e)))
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)
        
        return torque, arrived
    
if __name__ == "__main__":
    
    #robot and environment creation
    robot = TwoLink()
    #robot = ThreeLink()
    #robot = models.DH.Puma560()

    env = PyPlot()
    goal = [pi/2,pi/2]

    T = 3
    traj = ClippedTrajectory(robot.q, goal, T)

    symrobot = SymbolicPlanarRobot(len(robot.links))
    model = EulerLagrange(symrobot, os.path.join("src/models",robot.name))
    Profiler.print()
        
    # loop = Adaptive_Facile(robot, env, model, [0,-9.81,0])
    loop = Adaptive_FFW(robot, env, model, [0,0,-9.81], plotting = True) #planar
    # loop = Adaptive_FFW(robot, env, model, [0,0,-9.81], plotting = True) #3D
    
    
    loop.setR(reference = traj, goal = goal, threshold = 0.05)
    loop.setK(kp = [200,80], kd = [50,20]) #3R
    # loop.setK(kp=) #Panda
    
    loop.simulate(dt = 0.01)
    loop.plot()