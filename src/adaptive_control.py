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
        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + np.matmul(Y, self.robot.realpi).astype(np.float64)
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        self.append(q_d,qd_d,qdd_d,torque)
                
        #Update rule
        ed_r = qd_r - self.robot.qd #modified velocity error
        deltaPi = (gainMatrix @ Y).T @ ed_r
        self.robot.pi = self.robot.pi + deltaPi
                        
        return torque, arrived
    
    

class Adaptive_FFW(TrajectoryControl):

    def __init__(self, robot = None, env = None, dynamicModel: EulerLagrange = None, plotting = True, u_bound = None, gainMatrixMultiplier = 0.1):
        super().__init__(robot, env, dynamicModel, plotting)
        self.dynamicModel = dynamicModel
        self.u_bound = np.array(u_bound) if u_bound is not None else np.array([200]*len(robot.links))
        assert(gainMatrixMultiplier < 1.)

        self.gainMatrix = np.diag(self.robot.pi)
        self.gainMatrix *= gainMatrixMultiplier

        
    def feedback(self):
        '''Computes the torque necessary to follow the reference trajectory'''
        #Feedback of the error
        q_d, qd_d, qdd_d = self.reference(self.robot.n, self.t[-1])
        e = q_d - self.robot.q
        ed = qd_d - self.robot.qd
        #arrived = self.check_termination(e,ed)

        #Feedforward of the torque
        Profiler.start("Y evaluation")
        Y = self.dynamicModel.evaluateY(q_d, qd_d, qd_d, qdd_d)
        Profiler.stop()
        Profiler.start("Torque")
        #feedforward = np.matmul(self.robot.inertia(q_d), qdd_d) + np.matmul(self.robot.coriolis(q_d,qd_d), qd_d) +self.robot.gravload(q_d, gravity=self.gravity)        
        feedforward = np.matmul(Y, self.robot.pi).astype(np.float64)

        #print(np.linalg.norm(feedforward-myfeedforward))

        torque = np.matmul(self.kp,e) + np.matmul(self.kd,ed) + feedforward
        torque = np.clip(torque, -self.u_bound, self.u_bound)
        self.append(q_d,qd_d,qdd_d,torque)
        epi = np.linalg.norm((self.robot.pi - self.robot.realpi).astype(np.float64))
        self.logParameters(epi)
        Profiler.stop()

        #Update rule
        Profiler.start("Update Rule")
        
        #gainMatrix = np.diag([0.1,0.1,0.1,0.000001,0.000001,0.000001])
        sat_e = np.array([sat(el) for el in e], dtype=np.float64)
        deltaPi = self.gainMatrix @ (Y.T @ (sat_e + ed))
        self.robot.pi = self.robot.pi + deltaPi
        Profiler.stop()
    
        #gtM = self.robot.inertia(self.robot.q)
        #predM = self.dynamicModel.inertia(q,True)

        #print(np.linalg.norm(gtM-predM))

        
        #tp = np.matmul(self.kp,e)
        #td = np.matmul(self.kd,ed)

        #print(f"ERROR P: {e}    NORM: {np.linalg.norm(e)}")  
        #print(f"ERROR V: {ed}    NORM: {np.linalg.norm(ed)}")
        #print(f"TORQUE DAL PROPORTIONAL: {tp[3]}   NORM: {np.linalg.norm(tp)}")
        #print(f"TORQUE DAL DERIVATIVE: {td[3]}   NORM: {np.linalg.norm(td)}")
        #print(f"TORQUE DAL MODELLO: {np.linalg.norm(np.matmul(Y, self.robot.pi).astype(np.float64))}")
        print(f"TIME {self.t[-1]}")
        
        #return np.matmul(Y, self.robot.realpi).astype(np.float64), self.t[-1] > 5 or isnan(torque[0])
        return torque, self.t[-1] > 6.5 or isnan(torque[0])
    
if __name__ == "__main__":
    
    robot = ParametrizedRobot(UR3(), max_parameters_error = 1., seed=42)
    model = EulerLagrange(robot ,os.path.join("src/models",robot.name)) 
    
    traj = ExcitingTrajectory([[1.11,2.31,1.02,1.21],[2.1,1.11,1.032,1.25],[1.12,1.21,1.05,1.12],[1.05,1.06,1.033,1.234],[1.44,1.31,1.05,1.36],[1.1,1.22,2.019,1.19]], 6.5)
    #traj = ClippedTrajectory(robot.q, [pi/2, -pi/8, pi/4, -pi/4, pi/2, pi/2.5], 5)

    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = False, u_bound = [2e2,2e2,2e2,2e2,2e2,2e2], gainMatrixMultiplier=0.000001)
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp=[150,80,60,5,5,2], kd = [10,7,5,0.5,0.01,0.01])
        
    #checkGains(model, robot, loop.kp, loop.kd)
    
    loop.simulate(dt = 0.001)
    loop.plot()
    Profiler.mean()
    
    