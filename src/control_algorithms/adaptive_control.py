import numpy as np
from tools.robots import *
from tools.control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from trajectory_control import TrajectoryControl


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
            self.beliefPi[shift+0] = self.pi[shift+0] + np.random.normal(0,10,1)

            self.pi[shift+1] = mr[0]
            self.beliefPi[shift+1] = self.pi[shift+1]

            self.pi[shift+2] = mr[1]
            self.beliefPi[shift+2] = self.pi[shift+2]

            self.pi[shift+3] = mr[2]
            self.beliefPi[shift+3] = self.pi[shift+3]

            self.pi[shift+4] = I_link[0,0] 
            self.beliefPi[shift+4] = self.pi[shift+4] + np.random.normal(0,10,1)

            self.pi[shift+5] = 0
            self.beliefPi[shift+5] = self.pi[shift+5]

            self.pi[shift+6] = 0
            self.beliefPi[shift+6] = self.pi[shift+6]

            self.pi[shift+7] = I_link[1,1]
            self.beliefPi[shift+7] = self.pi[shift+7] + np.random.normal(0,10,1)

            self.pi[shift+8] = 0
            self.beliefPi[shift+8] = self.pi[shift+8]

            self.pi[shift+9] = I_link[2,2]
            self.beliefPi[shift+9] = self.pi[shift+9] + np.random.normal(0,10,1)
        

class Adaptive_Facile(AdaptiveControl):
    
    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0]):
        super().__init__(robot, env, dynamicModel, gravity)
        
    def feedback(self):
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        

class Adaptive_FFW(AdaptiveControl):

    def __init__(self, robot = None, env = None, dynamicModel = None, gravity = [0,0,0]):
        super().__init__(robot, env, dynamicModel, gravity)
        
         
    def feedback(self):
        '''Computes the torque necessary to follow the reference trajectory'''

        n = len(self.robot.links)

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        qdd = self.robot.qdd

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)

        print(e)

        actualY = self.dynamicModel.evaluateY(q_d, qd_d, qdd_d)
        torque = self.kp @ e + self.kd @ ed + np.matmul(actualY, self.pi).astype(np.float64)

        # Update rule
        gainMatrix = np.eye(n*10) # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)

        deltaPi = gainMatrix @ (actualY.T @ (sat_e+ed))

        self.beliefPi = self.beliefPi + deltaPi

        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque , arrived

    
if __name__ == "__main__":
    
    #robot and environment creation
    robot = ThreeLink()
    env = PyPlot()
    goal = [pi/2,pi/2,0]
    
    # T = 3
    # traj = ClippedTrajectory(robot.q, goal, T)

    # symrobot = SymbolicPlanarRobot(3)
    # model = EulerLagrange(3, robot = symrobot)
    
    # loop = Adaptive_FFW(robot, env, [0,-9.81,0])

    # loop.setR(reference = traj, goal = goal, threshold = 0.05)
    # loop.setK(kp = [200,100,100], kd = [100,60,60])
    
    # loop.simulate(dt = 0.01)
    # loop.plot()