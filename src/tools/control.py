from typing import Tuple
from roboticstoolbox import *
import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from tools.robots import *
from math import pi
import matplotlib.pyplot as plt

class Control():
    
    def __init__(self, robot = None, env = None, plotting = True):
        #env 
        self.env = env
        
        #robot 
        self.robot = robot
        self.robot.q = np.zeros((robot.n))
        self.robot.qd = np.zeros((robot.n))
        self.robot.qdd = np.zeros((robot.n))
        
        #realtime plotting
        if plotting:
            env.launch(realtime=True)
            env.add(robot)
        self.plotting = plotting
        
        #offline plotting 
        self.q = [np.array(self.robot.q)]
        self.qd = [np.array(self.robot.qd)]
        self.qdd = [np.array(self.robot.qdd)]
        self.t = [0]
        self.epi = list()
         
          
    def apply(self, signal) -> None:
        self.robot.qd = signal
    
    def feedback(self) -> Tuple[np.ndarray, bool]:
        return np.random.rand(self.robot.n)
    
    def setR(self, reference = None, threshold = 0.01):
        if reference is not None:
            self.reference = reference
            q_d, qd_d, qdd_d = reference(self.robot.n, 0)
            self.q_d = [q_d]
            self.qd_d = [qd_d]
            self.qdd_d = [qdd_d]
            if hasattr(reference, "goal"):
                self.goal = reference.goal

        self.threshold = threshold
        
            
    def setK(self, kp = None, kd = None):
        if kp is not None:
            self.kp = np.diag(kp)
        if kd is not None:
            self.kd = np.diag(kd)
            
    def setdt(self, dt):
        self.dt = dt
        
    def simulate(self, dt = None, T = None) -> None: 
        assert(self.reference is not None)
        #Simulation time interval
        if dt is not None: 
            self.dt = dt 
        else:
            self.dt = 0.02
            
        self.T = self.reference.T if T is None else T
        self.forced_termination = True if T is not None else False
        
        arrived = False
        while True:
            if arrived: break
            signal, arrived = self.feedback()
            self.apply(signal)
            if self.plotting:
                self.env.step(self.dt)
            else:
                for i in range(self.robot.n):
                    self.robot.q[i] += self.robot.qd[i] * (dt)
                    
    def append(self):
        self.q.append(np.array(self.robot.q))
        self.qd.append(np.array(self.robot.qd))
        self.qdd.append(np.array(self.robot.qdd))
        self.t.append(self.t[-1] + self.dt)
        
    def check_termination(self, e, ed):
        # Termination condition check
        e = np.abs(e)
        ed = np.abs(ed)

        position_ok = all(e < self.threshold) == True
        velocity_ok = all(ed < self.threshold) == True
        time_ok = self.t[-1] >= self.T
        if self.forced_termination: return time_ok
        return time_ok and position_ok and velocity_ok
                
                
    def plot(self):        
        fig, axs = plt.subplots(self.robot.n, 3, sharey="col", sharex = True)# a figure with a nx3 grid of Axes
        fig2, axs2 = plt.subplots(self.robot.n, 3, sharey="col", sharex = True)
        fig3, axs3 = plt.subplots(1, 1)

                
        q = np.array(self.q)
        qd = np.array(self.qd)
        qdd = np.array(self.qdd)
        q_d = np.array(self.q_d)
        qd_d = np.array(self.qd_d)
        qdd_d = np.array(self.qdd_d)
        e = np.subtract(q_d,q)
        ed = np.subtract(qd_d,qd)
        u = np.array(self.u)
        
        fig.suptitle("Joint data in function of time")
        axs[0,0].set_title("q")
        axs[0,1].set_title("q_dot")
        axs[0,2].set_title("q_dot_dot")                     
        axs2[0,0].set_title("u")
        axs2[0,1].set_title("error on q")
        axs2[0,2].set_title("error q_dot")
        
        for i in range(self.robot.n):
            axs[i,0].set_ylabel(f"joint {i+1}")
            axs[i,0].plot(self.t, q[:,i], lw = 2, color = "red")
            axs[i,1].plot(self.t, qd[:,i], lw = 2, color = "red")
            axs[i,2].plot(self.t, qdd[:,i], lw = 2, color = "red")
            
            axs[i,0].plot(self.t, q_d[:,i], lw = 2, color = "blue")
            axs[i,1].plot(self.t, qd_d[:,i], lw = 2, color = "blue")
            axs[i,2].plot(self.t, qdd_d[:,i], lw = 2, color = "blue")
            
            axs2[i,0].plot(self.t[:-1], u[:,i], lw = 2, color = "green")            
            axs2[i,1].plot(self.t, e[:,i], lw = 2, color = "black")
            axs2[i,2].plot(self.t, ed[:,i], lw = 2, color = "black")

        axs3.plot(self.t[1:], self.epi, lw=2, color="black")
            
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
    

