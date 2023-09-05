import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from tools.robots import *
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *

class Adaptive_Facile(TrajectoryControl):
    
    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = 700):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = np.array(u_bound) if u_bound is not None else np.array([200]*len(robot.links))
        self.adaptiveGains = np.ones((len(self.robot.pi),)) * 0.05 #DEFAULT
        
        
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
        deltaPi = self.adaptiveGains * (Y.T @ ed_r)
        self.robot.pi = self.robot.pi + deltaPi
        
        self.append(q_d,qd_d,qdd_d,torque)
        self.logParameters(epi)
                        
        return torque, arrived
    
    
class Adaptive_FFW(TrajectoryControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = None):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = np.array(u_bound) if u_bound is not None else np.array([200]*len(robot.links))
        self.adaptiveGains = np.ones((len(self.robot.pi),)) * 0.05 #DEFAULT

             
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
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = self.adaptiveGains * (np.matmul(Y.T, sat_e+ed))
        self.robot.pi = self.robot.pi + deltaPi
        Profiler.stop()
        
        self.logParameters(epi)
        self.append(q_d,qd_d,qdd_d,torque)
        
        return torque, arrived
    


    
    