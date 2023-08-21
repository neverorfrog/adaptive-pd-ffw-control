import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from roboticstoolbox.backends.swift import Swift
from tools.robots import *
from math import pi
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from control.trajectory_control import TrajectoryControl
from tools.dynamics import *

class Adaptive_Facile(TrajectoryControl):
    
    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = 700):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = u_bound
        
    def feedback(self):
        #Feedback of the error
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        e = q_d - self.robot.q
        ed = qd_d - self.robot.qd
        arrived = self.check_termination(e,ed)
        
        #Feedforward of the torque 
        gainMatrix = np.linalg.inv(self.kd) @ self.kp
        qd_r = qd_d + gainMatrix @ e    
        qdd_r = qdd_d + gainMatrix @ ed   
        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_r, qdd_r)    
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.pi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        self.append(q_d,qd_d,qdd_d,torque)
                
        #Update rule
        ed_r = qd_r - self.robot.qd #modified velocity error
        deltaPi = (gainMatrix @ Y).T @ ed_r
        self.robot.pi = self.robot.pi + deltaPi
                        
        return torque, arrived
    
    

class Adaptive_FFW(TrajectoryControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = None):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = u_bound if isinstance(u_bound, np.ndarray) else np.array([200]*len(robot.links))
        
             
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
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.realpi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        self.append(q_d,qd_d,qdd_d,torque)
        Profiler.stop()

        #Update rule
        Profiler.start("Update Rule")
        gainMatrix = np.eye(len(self.robot.pi)) * 0.1 # TODO: make this a parameter
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = gainMatrix @ (Y.T @ (sat_e + ed))
        self.robot.pi = self.robot.pi + deltaPi
        Profiler.stop()
        Profiler.print()
        
        print(f"CONFIG: {self.robot.q}")        
        print(f"ERROR P: {e}")  
        print(f"ERROR V: {ed}")
                           
        return torque, arrived
    
if __name__ == "__main__":
    
    robot = ParametrizedRobot(Polar2R(), stddev = 5)
    model = EulerLagrange(robot, os.path.join("src/models",robot.name)) 
    # model = EulerLagrange(robot)
    
    
    traj = ClippedTrajectory(robot.q, [pi/2,0], 4)
    
    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = True, u_bound = [100,100])
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp= [100,2000], kd = [10,40]) 
        
    # checkGains(model, robot, loop.kp, loop.kd)
    
    loop.simulate(dt = 0.01)
    loop.plot()
    
    