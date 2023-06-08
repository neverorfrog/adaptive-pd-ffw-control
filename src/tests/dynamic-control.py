import numpy as np
from tools.Models import TwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot

#TODO, to be finished (integration not working, torque values are skyrocketing)


class PDReg(Control):
    
    def feedforward(self, u):
        return self.robot.accel(self.robot.q,self.robot.qd,u, gravity = [0,0,0])
         
    
    def feedback(self):
        #position error
        e = self.reference - self.robot.q
        
        #derivative of position error
        ed = -self.robot.qd
        
        arrived = True if np.sum(np.abs(e)) < self.threshold else False
        return (self.kp @ e + self.kd @ ed) , arrived
    
    def simulate(self, dt = 0.02):
        self.q = [self.robot.q.tolist()]; 
        self.t = [0];
        arrived = False
        while not arrived:
            u, arrived = self.feedback()
            print(f"torque = {u}")
            qdd = self.feedforward(u)
            print(f"acceleration = {qdd}")
            
            #numerical integration
            qd = (qdd - self.robot.qdd)/dt
            self.robot.qdd = qdd
            self.robot.qd = qd
            
            env.step(dt)
            self.q.append(self.robot.q.tolist())
            self.t.append(self.t[-1] + dt)
        
        
        
    
    
if __name__ == "__main__":
    
    #robot and environment creation
    robot = TwoLink()
    robot.q = [0,0.5]
    robot.qd = [0,0]
    env = PyPlot()
    qref = [0,0]
    
    loop = PDReg(robot,env)
    loop.setR(qref, 0.001)
    loop.setK(kp = [0.06,0.05], kd = [0.02,0.04])
    loop.simulate()
    loop.plot()
    



