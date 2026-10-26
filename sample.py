from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.utils.plotting import plot_timeseries


plant = SymbolicDoublePendulum(mass=[0.6, 0.5],
                               length=[0.3, 0.2])

sim = Simulator(plant=plant)

controller = PointPIDController(torque_limit=[5.0, 5.0])
controller.set_parameters(Kp=10, Ki=1, Kd=1)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=[2.8, 1.2, 0., 0.],
                                   tf=5., dt=0.01, controller=controller)
plot_timeseries(T, X, U)