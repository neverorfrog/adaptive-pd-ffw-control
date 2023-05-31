import bdsim
sim = bdsim.BDSim(animation=True)
bd = sim.blockdiagram()

# define the blocks
demand = bd.STEP(T=1, name='demand')
sum = bd.SUM('+-')
gain = bd.GAIN(10)
plant = bd.LTI_SISO(0.5, [2, 1], name='plant')
scope = bd.SCOPE(styles=['k', 'r--'])

# connect the blocks
bd.connect(demand, sum[0], scope[1])
bd.connect(sum, gain)
bd.connect(gain, plant)
bd.connect(plant, sum[1], scope[0])

bd.compile()  # check the diagram
sim.report(bd)  # , format="latex")
sim.report(bd, "lists")  # list all blocks and wires
sim.report(bd, "schedule")

bd.compile()          # check the diagram
bd.report_summary()   # list the system
out = sim.run(bd, 5)   # simulate for 5s
print(out)