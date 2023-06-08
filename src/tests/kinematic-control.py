import numpy as np
from tools.Models import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from typing import Tuple

class RRMC(Control): 
    '''Resolved rate motion control (no feedback action)''' 
        
    def feedforward(self, vd):
        J = self.robot.jacob0(self.robot.q)
        J_pinv = np.linalg.pinv(J)
        return J_pinv @ vd
        
    def simulate(self, vd, dt = 0.02, epochs = 200):
        #The feedforward is computed at every timestep
        vd = self.reference.reshape((self.robot.n)) #just to be sure
        self.q = np.zeros((epochs,self.robot.n))
        self.t = np.arange(stop = epochs, step = 1) * dt
        for i in range(epochs):
            self.robot.qd = self.feedforward(vd)
            self.q[i,:] = self.robot.q
            env.step(dt)


class PBS(Control):
    '''Position based servoing (without orientation for now)'''  
        
    def feedforward(self,vd) -> np.ndarray:
        J = self.robot.jacob0(self.robot.q)
        J_pinv = np.linalg.pinv(J)
        return J_pinv @ vd
    
    def feedback(self) -> Tuple[np.ndarray, bool]:
        '''Computes desired task velocity based on the task error'''
        currentPose = self.robot.fkine(self.robot.q)
        currentPosition = currentPose.t[0:2]
        error = self.reference - currentPosition #position error
        arrived = True if np.sum(np.abs(error)) < self.threshold else False
        return self.kp @ error, arrived
        
    def simulate(self, dt = 0.02): 
        self.q = [self.robot.q.tolist()]; 
        self.t = [0];
        arrived = False
        while not arrived:
            vd, arrived = self.feedback()
            self.robot.qd = self.feedforward(vd)
            env.step(dt)
            self.q.append(self.robot.q.tolist())
            self.t.append(self.t[-1] + dt)
             

if __name__ == "__main__":
    #robot and environment creation
    robot = TwoLink()
    robot.q = [0,0.5]
    env = PyPlot()
    
    #Control loop 1
    # loop = RRMC(robot,env)
    # vd = np.array([-0.2,-0.2])
    # loop.setR(vd)
    # loop.simulate(vd, epochs = 50)
    # loop.plot()
    
    #Control loop 2
    loop = PBS(robot,env)
    loop.setR([0,1], 0.01)
    loop.setK(kp = np.ones(2))
    loop.simulate()
    loop.plot()
    
    
    
