import os
import numpy as np
import sympy as sp
from multiprocessing import Process

directory = "src/models"
robot = "Puma 560"



def simply(path):
    print(f"simplify of {path} started")
    bin = np.load(open(path,"rb"), allow_pickle=True)
    bin = sp.simplify(bin)
    np.save(open(path, "wb"), bin)
    print(f"file {path} simplified")

processes = []
directory = os.path.join(directory,robot)
for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    p = Process(target=simply, args=(path,))
    p.start()
    processes.append(p)

for el in processes:
    el.join()