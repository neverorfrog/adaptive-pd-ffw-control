from typing import Any
import numpy as np
from tools.Models import TwoLink, UncertantTwoLink
from tools.Control import Control
from roboticstoolbox.backends.PyPlot import PyPlot
from math import pi
from roboticstoolbox import quintic_func
from tools.Utils import ClippedTrajectory

class FBL(Control):

    def __init__(self, robot=None, env=None, gravity=[0,0,0], beliefRobot = None):
        super().__init__(robot, env)
        self.gravity = gravity        
    
    def apply(self, torque) -> None:
        qd2 = self.robot.qd + (self.dt/2)*self.robot.qdd; 
        q2 = self.robot.q + self.dt/2*qd2; 
        qdd2 = self.robot.accel(q2,qd2, torque, gravity = self.gravity)
        self.robot.qd = self.robot.qd + self.dt*qdd2
        
        #next step
        self.robot.qdd = self.robot.accel(self.robot.q,self.robot.qd, torque, gravity = self.gravity)
                 
    def feedback(self):

        ref = self.reference(self.t[-1])

        q_d = list()
        qd_d = list()
        qdd_d = list()

        for e in ref:
            q_d.append(e[0])
            qd_d.append(e[1])
            qdd_d.append(e[2])

        e = q_d - self.robot.q 
        ed = qd_d - self.robot.qd

        n = self.robot.coriolis(self.robot.q, self.robot.qd) + self.robot.gravload(q=self.robot.q, gravity=self.gravity)

        torque = self.robot.inertia(self.robot.q) @ ( qdd_d +  self.kp @ e + self.kd @ ed ) + n

        condition = self.t[-1] >= self.reference.T and sum(abs(e)) < self.threshold and sum(abs(ed)) < self.threshold

        return torque , condition 
        
        
    
if __name__ == "__main__":

    #robot and environment creation
    brobot = UncertantTwoLink()

    T = 2

    traj_q1 = quintic_func(0, pi/2, T)
    traj_q2 = quintic_func(0, -pi/2, T)

    unifiedTraj = ClippedTrajectory([traj_q1, traj_q2], T)

    robot = TwoLink()
    robot.q = [0,0]
    robot.qd = [0,0]
    robot.qdd = [0,0]
    env = PyPlot()
    
    loop = FBL(robot,env, [0,-9.81,0], brobot)
    loop.setR(reference = unifiedTraj)
    loop.setK(kp = [16,16], kd = [4,4])
    loop.simulate()
    loop.plot()