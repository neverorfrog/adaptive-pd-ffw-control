import numpy as np
from tools.Models import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi

class PDReg(Control):
    
    def apply(self, torque) -> None:
        #numerical integration (runge kutta 2)
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2; 
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = [0,0,0])
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = [0,0,0])
         
         
    
    def feedback(self):
        '''Computes the torque necessary to reach the reference position (without gravity)'''
        e = self.reference - self.robot.q #position error
        ed = -self.robot.qd
        arrived = True if np.sum(np.abs(e)) < self.threshold else False
        torque = self.kp @ e + self.kd @ ed
        return torque , arrived 
        
    
if __name__ == "__main__":
    
    #robot and environment creation
    robot = TwoLink()
    robot.q = [0,0]
    robot.qd = [0,0]
    robot.qdd = [0,0]
    env = PyPlot()
    
    loop = PDReg(robot,env)
    loop.setR(reference = [pi/3,0])
    loop.setK(kp = [1,1], kd = [6,6])
    loop.simulate()
    loop.plot()
    



