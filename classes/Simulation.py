from roboticstoolbox import *
import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from Models import *
from math import pi

class Simulation():
    
    def __init__(self, robot = None, env = None):
        self.robot = robot
        self.env = env
        env.launch(realtime=True)
        env.add(robot)
        robot.q = robot.qr
        
    def simulate(self):
        '''Simulation of arbitrary joint velocity'''
        self.robot.qd = np.array([0.8,1])
        for _ in range(500):
            self.env.step(0.05)
    

if __name__ == "__main__":
    #robot creation
    robot = TwoLink()
    link1 = robot["link1"]
    print(link1)
    A1 = link1.A(pi/3)
    print(A1)
    
    env = PyPlot()
    sim = Simulation(robot,env)
    sim.simulate()  

