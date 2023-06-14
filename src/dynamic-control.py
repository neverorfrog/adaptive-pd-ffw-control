import numpy as np
from tools.Models import TwoLink, UncertantTwoLink
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
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        e = self.goal - self.robot.q #position error
        ed = -self.robot.qd
        arrived = np.sum(np.abs(e)) < self.threshold and np.sum(np.abs(ed)) < self.threshold
        current_gravity = self.robot.gravload(q=self.robot.q, gravity=self.gravity)
        print(f" CURRENT GRAVITY TERM: {current_gravity}\n")
        torque = self.kp @ e + self.kd @ ed + current_gravity

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
    
class Feedforward(Control):

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
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        pos = self.robot.q
        vel = self.robot.qd
        acc = self.robot.qdd
        
        torque_ff = self.robot.rne(pos,vel,acc, self.gravity)
        
        print(self.i)
        # print(f" Reference: {self.reference.q[self.i]}\n Actual configuration: {pos} \n")
        e = self.reference.q[self.i] - pos #position error
        ed = self.reference.qd[self.i] - vel #velocity error
        # print(f" Position error: {e}\n Velocity error: {ed} \n")
        e_goal = self.goal - pos
        ed_goal = -vel
        arrived = np.sum(np.abs(e_goal)) < self.threshold and np.sum(np.abs(ed_goal)) < self.threshold
        # grav = self.robot.gravload(q=self.robot.q, gravity=self.gravity)
        torque = torque_ff + self.kp @ e + self.kd @ ed 

        return torque , arrived   
    
 
        
    
if __name__ == "__main__":
    
    #robot and environment creation
    brobot = UncertantTwoLink()
    robot = TwoLink()
    env = PyPlot()
    
    #t is the number of steps
    epochs = 200
    traj = jtraj(q0 = robot.q, qf = [1.5,1], t = epochs)
    loop = Feedforward(robot, env, [0,-9.81,0])
    loop.setR(reference = traj, goal = [1.5,1], threshold = 0.1)
    loop.setK(kp = [80,40], kd = [60,30])
    loop.simulate(dt = 1/epochs, epochs = epochs)
    loop.plot()
    



