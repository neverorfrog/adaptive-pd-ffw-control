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
        
    def append(self,q,qd,qdd,q_d,qd_d,qdd_d,torque):
        self.q.append(np.array(q))
        self.qd.append(np.array(qd))
        self.qdd.append(qdd.tolist())
        self.q_d.append(np.array(q_d))
        self.qd_d.append(np.array(qd_d))
        self.qdd_d.append(np.array(qdd_d))
        self.u.append(torque)
        self.t.append(self.t[-1] + self.dt)
        


class Feedforward(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)
         
    def feedback(self):
        '''Computes the torque necessary to reach the reference position'''
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        qdd = self.robot.qdd
    
        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        
        # Termination condition check
        e_goal = self.goal - q
        ed_goal = -qd
        # print(f"Position error: {e_goal} \n Velocity error: {ed_goal}")
        arrived = self.t[-1] >= self.reference.T and np.sum(np.abs(e_goal)) < self.threshold and np.sum(np.abs(ed_goal)) < self.threshold
        
        #Feedforward torque computation
        torque_ff = self.robot.rne(q_d,qd_d,qdd_d, self.gravity)
        
        #Feedback action
        e = q_d - q #position error
        ed = qd_d - qd #velocity error
        # print(f" Position error: {e}\n Velocity error: {ed} \n")
        torque = torque_ff + self.kp @ e + self.kd @ ed
        
        self.append(q,qd,qdd,q_d,qd_d,qdd_d,torque)

        return torque , arrived
    

class FBL(TrajectoryControl):

    def __init__(self, robot=None, env=None, gravity=[0,0,0]):
        super().__init__(robot, env, gravity)       
                 
    def feedback(self):
        
        #Current configuration
        q = self.robot.q
        qd = self.robot.qd
        qdd = self.robot.qdd

        #Reference Configuration
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])

        # Feedback action
        e = q_d - q
        ed = qd_d - qd
        n = self.robot.coriolis(q, qd) @ qd + self.robot.gravload(q, gravity = self.gravity)
        torque = self.robot.inertia(q) @ (qdd_d +  self.kp @ e + self.kd @ ed ) + n

        # Termination condition check
        arrived = self.t[-1] >= self.reference.T and sum(abs(e)) < self.threshold and sum(abs(ed)) < self.threshold
        
        self.append(q,qd,qdd,q_d,qd_d,qdd_d,torque)

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
        qdd = self.robot.qdd

        #e = q_d - q
        #ed = qd_d - qd

        y = np.matrix([[qdd[0], qdd[1]],[0, (qdd[0] + qdd[1])]])

        l1 = self.robot.links[0].a

        m0 = np.matrix([[2*self.a1*l1*np.cos(q[1]), self.a1*l1*np.cos(q[1])],[self.a1*l1*np.cos(q[1]), 0]])

        c0 = np.array([-self.a1*l1*qd[1]*np.sin(q[1])*(2*qd[0]+qd[1]), self.a1*l1*np.sin(q[1])*(qd[0]**2)]).T

        g0 = np.array([self.a1*self.gravity[1]*np.cos(q[0]+q[1])+ self.a4*self.gravity[1]*np.cos(q[0]), self.a1*self.gravity[1]*np.cos(q[0]+q[1])])

        
        return [1,1], False
    
if __name__ == "__main__":
    
    #robot and environment creation
    brobot = UncertantTwoLink()
    robot = TwoLink()
    env = PyPlot()
    goal = [pi/2,0,0]
    
    T = 3
    traj_fun = [quintic_func(robot.q[i], goal[i],T) for i in range(robot.n)]

    traj = ClippedTrajectory(traj_fun, T)
    
    # loop = Feedforward(robot, env, [0,-9.81,0])
    loop = Adaptive_ffw(robot, env, [0,-9.81,0])
    
    loop.setR(reference = traj, goal = goal, threshold = 0.2)
    loop.setK(kp = [100,80,80], kd = [60,40,40])
    
    loop.simulate()
    loop.plot()