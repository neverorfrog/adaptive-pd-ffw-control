import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
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

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = 700):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = u_bound
        
             
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
                                
        return torque, arrived
    
if __name__ == "__main__":
    
    robot = ParametrizedRobot(TwoLink(), stddev = 5)
    model = EulerLagrange(robot, os.path.join("src/models",robot.name))
    # model = EulerLagrange(robot)
    Profiler.print()
            
    traj = ClippedTrajectory(robot.q, [pi/2,pi/3], 4)
    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = True)
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp= [200,80], kd = [60,40]) 
    loop.simulate(dt = 0.01)
    loop.plot()
    Profiler.mean()
    
    