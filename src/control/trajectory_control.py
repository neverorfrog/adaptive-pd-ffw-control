import numpy as np
from roboticstoolbox.models.DH import Stanford

from tools.robots import *
from tools.control import Control
from tools.dynamics import *
from tools.utils import *

from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
import spatialmath.base.symbolic as sym


class TrajectoryControl(Control):
    def __init__(self, robot=None, env=None, gravity=[0,0,0], plotting = True):
        super().__init__(robot, env, plotting)
        self.gravity = gravity
        self.u = [robot.gravload(self.robot.q, gravity)]
        
    def apply(self, torque) -> None:
        #numerical integration (runge kutta 2)
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = self.gravity)
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = self.gravity)
        
    def append(self,q_d,qd_d,qdd_d,torque):
        self.q.append(np.array(self.robot.q))
        self.qd.append(np.array(self.robot.qd))
        self.qdd.append(np.array(self.robot.qdd))
        self.q_d.append(np.array(q_d))
        self.qd_d.append(np.array(qd_d))
        self.qdd_d.append(np.array(qdd_d))
        self.u.append(torque)
        self.t.append(self.t[-1] + self.dt)
        
    def check_termination(self, e, ed):
        # Termination condition check
        e = np.abs(e)
        ed = np.abs(ed)

        position_ok = all(e < self.threshold) == True
        velocity_ok = all(ed < self.threshold) == True
        return self.t[-1] >= self.reference.T and position_ok and velocity_ok
        
        

class Feedforward(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd        
    
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        
        # Termination condition check
        e = q_d - q #position error
        ed = qd_d - qd #velocity error
        arrived = self.check_termination(e,ed)
        
        #Feedforward torque computation
        torque_ff = self.robot.inertia(q_d) @ qdd_d + self.robot.coriolis(q_d, qd_d) @ qd_d + self.robot.gravload(q_d, gravity = self.gravity)
        
        #Feedback action
        torque = torque_ff + self.kp @ e + self.kd @ ed
                        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque , arrived
    

class FBL(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)       
                 
    def feedback(self):
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        
        # Feedback action
        n = self.robot.coriolis(q, qd) @ qd + self.robot.gravload(q, gravity = self.gravity)
        torque = self.robot.inertia(q) @ (qdd_d +  self.kp @ e + self.kd @ ed ) + n

        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived   

if __name__ == "__main__":
    
    #robot and environment creation
    n = 2
    robot = TwoLink()
    env = PyPlot()
    goal = [pi/6,pi/6]
    
    T = 1
    traj = ClippedTrajectory(robot.q, goal, T)
    
    loop = Feedforward(robot, env, [0,-9.81,0])

    loop.setR(reference = traj, goal = goal, threshold = 0.05)
    loop.setK(kp = [200,80], kd = [50,20])
    
    loop.simulate(dt = 0.01)
    loop.plot()