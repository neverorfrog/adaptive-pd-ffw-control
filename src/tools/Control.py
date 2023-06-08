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
    
    def feedforward(self):
        return np.random.rand(self.robot.n)
    
    def feedback(self):
        return None
    
    def setR(self, reference, threshold = 0.001):
        self.reference = reference
        self.threshold = threshold
    def setK(self, kp = None, kd = None):
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)
        
    def simulate(self, dt = 0.02):
        self.qd = self.feedforward()
        for _ in range(100):
            env.step()
    
    def plot(self):        
        _, axs = plt.subplots(self.robot.n, 1)  # a figure with a nx1 grid of Axes
        plt.xlabel("t"); plt.ylabel("q"); 
        t = np.array(self.t) 
        q = np.array(self.q)
        for i in range(self.robot.n):
            axs[i].plot(t, q[:,i], lw = 2)
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
    

