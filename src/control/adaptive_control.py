import numpy as np
from math import sin, cos
from roboticstoolbox.backends.PyPlot import PyPlot
from roboticstoolbox.backends.swift import Swift
from tools.robots import *
from math import isnan
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *

class Adaptive_Facile(TrajectoryControl):
    
    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = 700):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = np.array(u_bound) if u_bound is not None else np.array([200]*len(robot.links))
        
        
    def feedback(self):
        #Feedback of the error
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        e = q_d - self.robot.q
        ed = qd_d - self.robot.qd
        arrived = self.check_termination(e,ed)
        
        #Feedforward of the torque 
        A = np.linalg.inv(self.kd) @ self.kp
        qd_r = qd_d + A @ e    
        qdd_r = qdd_d + A @ ed   
        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_r, qdd_r)    
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.pi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        epi = np.linalg.norm((self.robot.pi - self.robot.realpi).astype(np.float64))

        #Update rule
        ed_r = qd_r - self.robot.qd #modified velocity error
        gainMatrix = np.ones((len(self.robot.pi),)) * 0.15 # TODO: make this a parameter
        deltaPi = gainMatrix * (Y.T @ ed_r)
        self.robot.pi = self.robot.pi + deltaPi
        
        self.append(q_d,qd_d,qdd_d,torque)
        self.logParameters(epi)
                        
        return torque, arrived
    
    
class Adaptive_FFW(TrajectoryControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = None):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = np.array(u_bound) if u_bound is not None else np.array([200]*len(robot.links))
             
    def feedback(self):
        '''Computes the torque necessary to follow the reference trajectory'''
        #Feedback of the error        
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        e = q_d - self.robot.q
        ed = qd_d - self.robot.qd
        arrived = self.check_termination(e,ed)

        #Feedforward of the torque
        Profiler.start("Y evaluation")
        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)
        Profiler.stop()
        Profiler.start("Torque")
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.pi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        epi = np.linalg.norm((self.robot.pi - self.robot.realpi).astype(np.float64))
        Profiler.stop()

        #Update rule
        Profiler.start("Update Rule")
        gainMatrix = np.ones((len(self.robot.pi),)) * 0.07 # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix * (np.matmul(Y.T, sat_e+ed))
        self.robot.pi = self.robot.pi + deltaPi
        Profiler.stop()
        
        self.logParameters(epi)
        self.append(q_d,qd_d,qdd_d,torque)
        
        return torque, arrived

if __name__ == "__main__":
    
    for i in range(1):
        robot = ParametrizedRobot(Polar2R(), stddev = 0)
        # model = EulerLagrange(robot, path = os.path.join("src/models",robot.name), loadpi = False) 
        model = EulerLagrange(robot) 
        
        # traj = ClippedTrajectory(robot.q, [pi/2,pi/4], 6)
        # traj = ClippedTrajectory(robot.q, robot.q, 6)
        # traj = ClippedTrajectory([pi/3,pi/4], [-pi/2,-pi/6], 4)
        traj = ExcitingTrajectory([[0.5,0.5,0.5,1],[0.8,0.8,0.8,1.2]])
        
        loop = Adaptive_FFW(robot, PyPlot(), model, plotting = False)
        # loop = Adaptive_Facile(robot, PyPlot(), model, plotting = False)
        loop.setR(reference = traj, threshold = 0.05)
        loop.setK(kp = [200,100], kd = [60,30])
        loop.simulate(dt = 0.01, T = 6)
        # np.save(open(os.path.join(os.path.join("src/models",robot.name),"pi.npy"), "wb"), robot.pi)
        Profiler.mean()
    loop.plot()
    
    