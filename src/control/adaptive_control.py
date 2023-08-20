import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from tools.robots import *
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *
import itertools
from sympy import trigsimp

class AdaptiveControl(TrajectoryControl):
    def __init__(self, robot = None, env = None, dynamicModel = None, plotting = True):
        super().__init__(robot, env, model, plotting)
        self.dynamicModel = dynamicModel
        

class Adaptive_FFW(AdaptiveControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True):
        super().__init__(robot, env, dynamicModel, plotting)

    def checkGains(self, q_d, qd_d, qdd_d):
        
        n = len(self.robot.links)

        M = sympy.Matrix(model.inertia_parametrized)
        g = sympy.Matrix(model.gravity_parametrized)        

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

        Profiler.start("Y evaluation")
        actualY = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)
        print(actualY.shape)
        Profiler.stop()

        Profiler.start("Feedback loop")
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(actualY, self.robot.pi).astype(np.float64)

        bound = 700
        torque = np.clip(torque, -bound, bound)

        # Update rule
        gainMatrix = np.eye(len(self.robot.pi)) * 0.1 # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix @ (actualY.T @ (sat_e + ed))
        self.robot.pi = self.robot.pi + deltaPi

        # print("Feedback loop took {:.2f} s\nCurrent Error: {:.2f}".format(Profiler.logger["feedback loop"], np.linalg.norm(e)))        
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)
        Profiler.stop()
        Profiler.print()           
                                
        return torque, arrived
    
if __name__ == "__main__":
    
    robot = ParametrizedRobot(TwoLink(), stddev = 5)
    # model = EulerLagrange(robot, os.path.join("src/models",robot.name), planar = True)
    model = EulerLagrange(robot, planar = True)
    Profiler.print()
            
    traj = ClippedTrajectory(robot.q, [pi/2,pi/2], 4)
    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = True)
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp= [200,80], kd = [60,40]) 
    loop.simulate(dt = 0.01)
    loop.plot()