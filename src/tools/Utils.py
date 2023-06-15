
class ClippedTrajectory():
    def __init__(self, functions, T) -> None:
        self.functions = functions 
        self.T = T
    
    #returns a tuple (q qdot qdotdot) for every joint
    def __call__(self, t):
        return [f(min(t,self.T)) for f in self.functions ]

    def getTrajList(self):
        return self.functions