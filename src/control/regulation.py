import numpy as np
from robots import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *

class Regulation(Control):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env)
        self.gravity = gravity
        
    
    def apply(self, torque) -> None:
        #numerical integration (runge kutta 2)
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2; 
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = self.gravity)
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = self.gravity)
         
    def feedback(self, kp, kd):
        '''Computes the torque necessary to reach the reference position'''
        e = self.goal - self.robot.q #position error
        ed = -self.robot.qd
        arrived = np.sum(np.abs(e)) < self.threshold and np.sum(np.abs(ed)) < self.threshold
        current_gravity = self.robot.gravload(q=self.robot.q, gravity=self.gravity)
        print(f" CURRENT GRAVITY TERM: {current_gravity}\n")
        torque = kp @ e + kd @ ed + current_gravity

        return torque , arrived
        
    
class Iterative(Control):

    def __init__(self, robot=None, env=None, gravity=[0,0,0], beliefRobot = None):
        super().__init__(robot, env)
        self.gravity = gravity
        self.gravityTerm_est = beliefRobot.gravload(q=self.robot.q, gravity=self.gravity)
        
    
    def apply(self, torque) -> None:
        #numerical integration (runge kutta 2)
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2; 
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = self.gravity)
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = self.gravity)
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        gamma = 2

        e = self.goal - self.robot.q #position error
        ed = -self.robot.qd
        arrived = np.sum(np.abs(e)) < self.threshold and np.abs(ed) < self.threshold
        torque = gamma * self.kp @ e + self.kd @ ed + self.gravityTerm_est

        #Gravity term update
        if(np.linalg.norm(self.robot.qd) < 0.1):
            self.gravityTerm_est = gamma * self.kp@e + self.gravityTerm_est
            realGravityTerm = self.robot.gravload(q=self.robot.q, gravity=self.gravity)
            print(f" CURRENT GRAVITY TERM PREDICTED: {self.gravityTerm_est}\nGROUND TRUTH: {realGravityTerm}")

        return torque , arrived
            
                
if __name__ == "__main__":
    
    #robot and environment creation
    robot = TwoLink()
    env = PyPlot()
    
    goal = [1.5,1]
    loop = Regulation(robot, env, [0,-9.81,0])
    loop.setR(goal = goal, threshold = 0.1)
    loop.setK(kp = [100,50], kd = [200,70])
    loop.simulate(dt = 0.01)
    loop.plot()
    



