import numpy as np
from roboticstoolbox.backends.PyPlot import PyPlot
from tools.robots import *
from roboticstoolbox.tools.trajectory import *
from tools.utils import *
from tools.dynamics import *
from control.adaptive_control import *

def SCARA_loop(robot, loadpi = False):
    model = EulerLagrange(robot, path = os.path.join("src/models",robot.name), loadpi = loadpi) 
    # traj = ClippedTrajectory([pi/8,0,-0.1,0], [pi/2,pi/3,0.1,-pi/6], 6)
    traj = SinusoidalTrajectory([pi/3,pi/4,0.2,pi/4])
    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = False)
    loop.adaptiveGains = np.array([5e-2]*model.p[0] + [5e-2]*model.p[1] + [5e-4]*model.p[2] + [5e-7]*model.p[3])
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp = [10,10,20,1], kd = [5,5,10,0.1])
    return loop

def Polar_loop(robot, loadpi = False):
    model = EulerLagrange(robot, path = os.path.join("src/models",robot.name), loadpi = loadpi) 
    traj = SinusoidalTrajectory([pi/3,pi/4])
    loop = Adaptive_FFW(robot, PyPlot(), model, plotting = False)
    loop.adaptiveGains = np.array([5e-2]+[5e-7]*10) 
    loop.setR(reference = traj, threshold = 0.05)
    loop.setK(kp = [20,10], kd = [6,3]) 
    return loop

if __name__ == "__main__":
    for i in range(1):
        robot = ParametrizedRobot(SCARA(), dev_factor = 5)
        model = EulerLagrange(robot)
        Profiler.print() 
        loop = Polar_loop(robot)
        # loop.simulate(dt = 0.001, T = 10)
        # np.save(open(os.path.join(os.path.join("src/models",robot.name),"pi.npy"), "wb"), robot.pi)
        Profiler.mean()
    loop.plot()