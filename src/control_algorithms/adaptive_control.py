import numpy as np
from tools.robots import *
from tools.control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from trajectory_control import TrajectoryControl

class AdaptiveFacile(TrajectoryControl):
    def feedback(self):
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        

class Adaptive_FFW(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)       
        self.theta = np.array([5,3]) #aka a2 a3
                 
    def feedback(self):

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
                
        # Feedback action
        Y = "stub" #TODO
        torque = self.kp @ e + self.kd @ ed + Y @ self.theta 

        # Update rule
        gainMatrix = np.array([[5,0],[0,5]], dtype=np.float64) # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaTheta = gainMatrix @ (Y @ (sat_e+ed))
        self.theta = self.theta + deltaTheta
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived

    
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