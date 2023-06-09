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
    
    def setR(self, reference, threshold = 0.001):
        self.reference = reference
        self.threshold = threshold
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
            self.q = [self.robot.q.tolist()]; 
            self.t = [0];
            self.u = [np.ndarray((self.robot.n)).tolist()]; 
            arrived = False
            while not arrived:
                signal, arrived = self.feedback()
                self.apply(signal)
                self.env.step(self.dt)
                self.q.append(self.robot.q.tolist())
                self.t.append(self.t[-1] + self.dt)
                self.u.append(signal.tolist())
        else:
            self.q = np.zeros((epochs,self.robot.n))
            self.t = np.arange(stop = epochs, step = 1) * dt
            arrived = False
            for i in range(epochs):
                signal, arrived = self.feedback()
                if arrived: break
                self.apply(signal)
                self.env.step(self.dt)
                self.q[i,:] = self.robot.q
                
                
    def plot(self):        
        _, axs = plt.subplots(self.robot.n, 2)  # a figure with a nx1 grid of Axes
        plt.xlabel("t"); plt.ylabel("q"); 
        t = np.array(self.t) 
        q = np.array(self.q)
        u = np.array(self.u)
        for i in range(self.robot.n):
            axs[i,0].plot(t, q[:,i], lw = 2)
            axs[i,1].plot(t, u[:,i], lw = 2)
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
    

