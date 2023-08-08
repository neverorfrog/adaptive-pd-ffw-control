import numpy as np
from tools.Models import *
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.Utils import *


class TrajectoryControl(Control):
    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env)
        self.gravity = gravity
        self.u = [robot.gravload(self.robot.q, gravity)]
        
    def apply(self, torque) -> None:
        #numerical integration (runge kutta 2)
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2; 
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = self.gravity)
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = self.gravity)
        
    def append(self,q_d,qd_d,qdd_d,torque):
        self.q.append(np.array(self.robot.q))
        self.qd.append(np.array(self.robot.qd))
        self.qdd.append(np.array(self.robot.qdd))
        self.q_d.append(np.array(q_d))
        self.qd_d.append(np.array(qd_d))
        self.qdd_d.append(np.array(qdd_d))
        self.u.append(torque)
        self.t.append(self.t[-1] + self.dt)
        
    def check_termination(self, e, ed):
        # Termination condition check
        e = np.abs(e)
        ed = np.abs(ed)

        position_ok = all(e < self.threshold) == True
        velocity_ok = all(ed < self.threshold) == True
        return self.t[-1] >= self.reference.T and position_ok and velocity_ok
        
        

class Feedforward(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd        
    
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        
        # Termination condition check
        e = q_d - q #position error
        ed = qd_d - qd #velocity error
        arrived = self.check_termination(e,ed)
        
        #Feedforward torque computation
        torque_ff = self.robot.inertia(q_d) @ qdd_d + self.robot.coriolis(q_d, qd_d) @ qd_d + self.robot.gravload(q_d, gravity = self.gravity)
        
        #Feedback action
        torque = torque_ff + self.kp @ e + self.kd @ ed
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque , arrived
    

class FBL(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)       
                 
    def feedback(self):
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        
        # Feedback action
        n = self.robot.coriolis(q, qd) @ qd + self.robot.gravload(q, gravity = self.gravity)
        torque = self.robot.inertia(q) @ (qdd_d +  self.kp @ e + self.kd @ ed ) + n

        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived   

class Adaptive_ffw(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)       
        self.theta = np.array([5,3]) #aka a2 a3

        self.a1 = robot.links[1].r[0] * robot.links[1].m 
        self.a4 = robot.links[1].m*self.robot.links[0].a + self.robot.links[0].r[0]*self.robot.links[0].m
                 
    def feedback(self):

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        
        #feedback action for known parameters
        y = np.array([[qdd_d[0], qdd_d[1]],[0, (qdd_d[0] + qdd_d[1])]])
        l1 = self.robot.links[0].a 
        m0 = np.array([[2*self.a1*l1*np.cos(q_d[1]), self.a1*l1*np.cos(q_d[1])],[self.a1*l1*np.cos(q_d[1]), 0]])
        c0 = np.array([-self.a1*l1*qd_d[1]*np.sin(q_d[1])*(2*qd_d[0]+qd_d[1]), self.a1*l1*np.sin(q_d[1])*(qd_d[0]**2)])
        g0 = np.array([self.a1*self.gravity[1]*np.cos(q_d[0]+q_d[1])+ self.a4*self.gravity[1]*np.cos(q_d[0]), self.a1*self.gravity[1]*np.cos(q_d[0]+q_d[1])])
        
        # Feedback action
        torque = self.kp @ e + self.kd @ ed + y @ self.theta + m0 @ qdd_d + c0*qd_d + g0 

        # Update rule
        gainMatrix = np.array([[5,0],[0,5]], dtype=np.float64) # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)

        deltaTheta = gainMatrix @ (y@(sat_e+ed))
        self.theta = self.theta + deltaTheta
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived
    

class Adaptive_ffw_10P(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)

        self.pi = []
        for link in robot.links:
            mr = link.m*link.r

            row = np.array([link.m], dtype=np.float64)
            row = np.concatenate([row, mr])
            row = np.concatenate([row, link.I[0]])
            row = np.concatenate([row, link.I[1][1:]])
            row = np.concatenate([row, link.I[2][2:]])

            self.pi.append(row)

        self.pi = np.array(self.pi)

        exit(0)
                 
    def feedback(self):

        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        #Error
        e = q_d - q
        ed = qd_d - qd
        arrived = self.check_termination(e,ed)
        
        torque = [1,1]
        
        # Trajectory logging
        self.append(q_d,qd_d,qdd_d,torque)

        return torque, arrived
    
if __name__ == "__main__":
    
    #robot and environment creation
    brobot = UncertantTwoLink()
    robot = TwoLink()
    env = PyPlot()
    goal = [pi/2,pi/2]
    
    T = 30
    traj = ClippedTrajectory(robot.q, goal, T)
    
    #loop = Feedforward(robot, env, [0,-9.81,0])
    #loop = Adaptive_ffw(robot, env, [0,-9.81,0])
    loop = Adaptive_ffw_10P(robot, env, [0,-9.81,0])

    loop.setR(reference = traj, goal = goal, threshold = 0.05)
    loop.setK(kp = [200,100], kd = [100,60])
    
    loop.simulate(dt = 0.01)
    loop.plot()