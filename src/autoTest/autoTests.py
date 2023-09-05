from tools.robots import *
from tools.utils import *
from tools.dynamics import *

from math import pi
from roboticstoolbox.backends.PyPlot import PyPlot
from controlTest import TestAdaptive_FFW

'''
TEMPLATE FOR A TEST:


    tot+=1
    description = "Setup Environment planar 2R"
    try:
        # FILL HERE WITH CODE
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

'''

OK = "\033[92m[PASSED]\033[0m"
NO = "\033[91m[FAILED]\033[0m"


if __name__ == "__main__":
    tot = 0
    passed = 0

    tot+=1
    description = "Setup Environment planar 2R"

    try:
        goal = [pi/2,pi/2]
        T = 3
        robot = ParametrizedRobot(TwoLink(), 10)
        traj = ClippedTrajectory(robot.q, goal, T)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    
    tot+=1
    description = "Computing the planar dynamic model"

    try:
        model = EulerLagrange(robot, planar = True)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Planar simulation loop"

    try:
        loopPassed = False
        loop = TestAdaptive_FFW(robot, PyPlot(), model, plotting = True)

        loop.setR(reference = traj, threshold = 0.05)
        loop.setK(kp = [200,80], kd = [50,20])

        loop.simulate(dt = 0.01)
        loop.plot()
        
        passed+=1
        loopPassed = True
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Planar inertia Matrix correctness"

    try:
        assert(loop.m_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)
    
    tot+=1
    description = "Planar S Matrix correctness"

    try:
        assert(loop.s_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Planar gravity Matrix correctness"

    try:
        assert(loop.g_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Planar Feed Forward correctness"

    try:
        assert(loop.ffw_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Planar Convergency"

    try:
        assert(loop.convergency & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Setup Environment Polar 2R"

    try:
        goal = [pi/2,pi/4]
        T = 3
        robot = ParametrizedRobot(Polar2R(), 10)
        traj = ClippedTrajectory(robot.q, goal, T)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    
    tot+=1
    description = "Computing the Polar dynamic model"

    try:
        model = EulerLagrange(robot, planar = False)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Polar simulation loop"

    try:
        loopPassed = False
        loop = TestAdaptive_FFW(robot, PyPlot(), model, plotting = True)

        loop.setR(reference = traj, threshold = 0.05)
        loop.setK(kp = [200,80], kd = [50,20])

        loop.simulate(dt = 0.01)
        loop.plot()
        
        passed+=1
        loopPassed = True
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Polar inertia Matrix correctness"

    try:
        assert(loop.m_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)
    
    tot+=1
    description = "Polar S Matrix correctness"

    try:
        assert(loop.s_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Polar gravity Matrix correctness"

    try:
        assert(loop.g_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Polar Feed Forward correctness"

    try:
        assert(loop.ffw_test & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)

    tot+=1
    description = "Polar Convergency"

    try:
        assert(loop.convergency & loopPassed)
        passed+=1
        print(f"[TEST] {description}... {OK}")
    except Exception as e: 
        print(f"[TEST] {description}... {NO}")
        print(e)
    
    print(f"[TEST] {passed}/{tot} tests passed")
