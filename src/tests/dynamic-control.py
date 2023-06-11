import numpy as np
from tools.Models import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi

class PDReg(Control):

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
        '''Computes the torque necessary to reach the reference position (without gravity)'''
        gravityTerm = self.robot.gravload(q=self.robot.q, gravity=self.gravity)
        e = self.reference - self.robot.q #position error
        ed = -self.robot.qd
        arrived = np.sum(np.abs(e)) < self.threshold
        torque = self.kp @ e + self.kd @ ed + gravityTerm
        return torque , arrived 
        
    
if __name__ == "__main__":
    
    #robot and environment creation
    robot = TwoLink()
    robot.q = [0,0]
    robot.qd = [0,0]
    robot.qdd = [0,0]
    env = PyPlot()
    
    loop = PDReg(robot,env, [0,-9.81,0])
    loop.setR(reference = [pi/2,-pi/2])
    loop.setK(kp = [3,4], kd = [6,6])
    loop.simulate()
    loop.plot()
    



