from typing import Tuple
from roboticstoolbox import *
import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from tools.Models import *
from math import pi
import matplotlib.pyplot as plt

class Control():
    
    def __init__(self, robot = None, env = None):
        #env 
        self.env = env
        env.launch(realtime=True)
        
        #robot 
        self.robot = robot
        self.robot.qd = np.zeros((robot.n)) 
        
        env.add(robot)
    
    def apply(self, signal) -> None:
        self.robot.qd = signal
    
    def feedback(self) -> Tuple[np.ndarray, bool]:
        return np.random.rand(self.robot.n)
    
    def setR(self, reference = None, goal = None, threshold = 0.001):
        if reference is not None:
            self.reference = reference
        self.threshold = threshold
        if goal is not None:
            self.goal = goal
    def setK(self, kp = None, kd = None):
        if kp is not None:
            self.kp = np.diag(kp)
        if kd is not None:
            self.kd = np.diag(kd)
    def setdt(self, dt):
        self.dt = dt
        
    def simulate(self, dt = None, epochs = None) -> None: 
        
        #Simulation time interval
        if dt is not None: 
            self.dt = dt 
        else:
            self.dt = 0.02
            
        if epochs is None:
            epochs = float('inf')
        
        self.q = [self.robot.q.tolist()]
        self.qd = [self.robot.qd.tolist()]
        self.t = [0]
        self.u = [np.ndarray((self.robot.n)).tolist()]
        arrived = False
        self.i = 0
        while True:
            signal, arrived = self.feedback()
            self.i += 1
            if arrived or self.i == epochs: break
            self.apply(signal)
            self.env.step(self.dt)
            self.q.append(self.robot.q.tolist())
            self.qd.append(self.robot.qd.tolist())
            self.t.append(self.t[-1] + self.dt)
            self.u.append(signal.tolist())
                
                
    def plot(self):        
        fig, axs = plt.subplots(self.robot.n, 3)  # a figure with a nx3 grid of Axes

        t = np.array(self.t) 
        q = np.array(self.q)
        qd = np.array(self.qd)
        u = np.array(self.u)
        
        fig.suptitle("Joint data in function of time")
        axs[0,0].set_title("q")
        axs[0,1].set_title("q_dot")
        axs[0,2].set_title("u")
        
        for i in range(self.robot.n):
            axs[i,0].plot(t, q[:,i], lw = 2, color = "red")
            axs[i,0].plot(t, self.reference.q[:,i], lw = 2, color = "blue")
            
            axs[i,1].plot(t, qd[:,i], lw = 2, color = "red")
            axs[i,1].plot(t, self.reference.qd[:,i], lw = 2, color = "blue")
            
            axs[i,2].plot(t, u[:,i], lw = 2)
        plt.show(block = True) 
          
        
if __name__ == "__main__":
    #robot creation
    robot = TwoLink()
    link1 = robot["link1"]
    print(link1)
    A1 = link1.A(pi/3)
    print(A1)
    
    env = PyPlot()
    loop = Control(robot,env)
    

