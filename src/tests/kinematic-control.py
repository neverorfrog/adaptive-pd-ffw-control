import numpy as np
from tools.Models import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from typing import Tuple

class RRMC(Control): 
    '''Resolved rate motion control (no feedback action)''' 
        
    def apply(self, vd) -> None:
        J = self.robot.jacob0(self.robot.q)
        J_pinv = np.linalg.pinv(J)
        self.robot.qd =  J_pinv @ vd
        
    def feedback(self) -> Tuple[np.ndarray, bool]:
        return self.reference.reshape((self.robot.n)), False 

class PBS(Control):
    '''Position based servoing (without orientation for now)'''  
        
    def apply(self,vd) -> None:
        J = self.robot.jacob0(self.robot.q)
        J_pinv = np.linalg.pinv(J)
        self.robot.qd = J_pinv @ vd
    
    def feedback(self) -> Tuple[np.ndarray, bool]:
        '''Computes desired task velocity based on the task error'''
        currentConfig = self.robot.q
        currentPose = self.robot.fkine(currentConfig)
        currentPosition = currentPose.t[0:2]
        error = self.reference - currentPosition #position error
        arrived = True if np.sum(np.abs(error)) < self.threshold else False
        return self.kp @ error, arrived
             

if __name__ == "__main__":
    #robot and environment creation
    robot = TwoLink()
    robot.q = [0,0.5]
    env = PyPlot()
    
    #Control loop 1
    loop = RRMC(robot,env)
    vd = np.array([-0.2,-0.2])
    loop.setR(vd)
    loop.simulate(epochs = 50)
    loop.plot()
    
    #Control loop 2
    # loop = PBS(robot,env)
    # loop.setR([0,1], 0.01)
    # loop.setK(kp = np.ones(2))
    # loop.simulate()
    # loop.plot()
    
    
    
