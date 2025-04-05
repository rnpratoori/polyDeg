# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, io, log, default_real_type, plot
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from pathlib import Path

import pyvista

# Define simulation parameters
s = 1.0e0
R = 10
T = 300
V = 1.0
epsilon_ = 1
lambda_ = epsilon_**2
# lambda_ = 1.0 * s
M = 1.0 / s
chi12 = 3.0
chi13 = 3.0
chi23 = 3.0
N1 = 1.0
N2 = 1.0
N3 = 1.0
beta = 0.001
dt = 1.0e-07
T = 1.0e-05 # End time
num_steps = T / dt # Number of time steps

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [100, 100], mesh.CellType.triangle)

# Create FunctionSpace
P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
ME = fem.functionspace(domain, mixed_element([P1, P1, P1, P1]))

# Define variational problem
# Define trial and test functions
u = fem.Function(ME)
# Previous solution
u_0 = fem.Function(ME)
u__1 = fem.Function(ME)
q1, v1, q2, v2 = ufl.TestFunctions(ME)
# Split mixed functions
c1, mu1, c2, mu2 = ufl.split(u)
c1_0, mu1_0, c2_0, mu2_0 = ufl.split(u_0)
c1__1, mu1__1, c2__1, mu2__1 = ufl.split(u__1)

# Initial condition
# Define the initial conditions using lambda functions
def c1_ic(x):
    values = 0.3 + 0.02 * (0.5 - rng_1.random(x.shape[1]))
    return np.clip(values, 0.01, 0.95)

def c2_ic(x):
    values = 0.3 + 0.02 * (0.5 - rng_2.random(x.shape[1]))
    return np.clip(values, 0.01, 0.95)

u.x.array[:] = 0.0
# dist_mu, dist_sigma = 0.0, 0.01
rng_1 = np.random.default_rng()
rng_2 = np.random.default_rng()
u.sub(0).interpolate(c1_ic)
u.sub(2).interpolate(c2_ic)
assert np.all(u.sub(0).collapse().x.array + u.sub(2).collapse().x.array < 0.99)
u.x.scatter_forward()

# def visualize_mixed(mixed_function: fem.Function, scale=1.0):
#     c1 = mixed_function.sub(0).collapse()
#     c2 = mixed_function.sub(2).collapse()

#     c1_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(c1.function_space))
#     c1_grid.point_data["c1"] = c1.x.array
#     plotter_c1 = pyvista.Plotter()
#     plotter_c1.add_mesh(c1_grid, show_edges=False)
#     plotter_c1.view_xy()
#     plotter_c1.show()

#     p_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(c2.function_space))
#     p_grid.point_data["c2"] = c2.x.array
#     plotter_p = pyvista.Plotter()
#     plotter_p.add_mesh(p_grid, show_edges=False)
#     plotter_p.view_xy()
#     plotter_p.show()

# visualize_mixed(u_0)

# Mark variables and apply constraints
c1 = ufl.variable(c1)
c2 = ufl.variable(c2)

# Define the free-energy with penalty term
f = s * (R * T / V) * (c1 * ufl.ln(c1) / N1 + c2 * ufl.ln(c2) / N2 + (1 - c1 - c2) * ufl.ln(1 - c1 - c2) / N3
                    + chi12 * c1 * c2 + chi13 * c1 * (1 - c1 - c2) + chi23 * c2 * (1 - c1 - c2)
                    + beta * ((1 / c1) + (1 / c2) + (1 / (1 - c1 - c2))))

# Now differentiate with respect to the variables
dfdc1 = ufl.diff(f, c1)
dfdc2 = ufl.diff(f, c2)

# Define residuals
F0 = ufl.inner(c1, q1) * ufl.dx - ufl.inner(c1_0, q1) * ufl.dx + (dt/2) * M * ufl.inner(ufl.grad(mu1 + mu1_0), ufl.grad(q1)) * ufl.dx
F1 = ufl.inner(mu1, v1) * ufl.dx - ufl.inner(dfdc1, v1) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c1), ufl.grad(v1)) * ufl.dx
F2 = ufl.inner(c2, q2) * ufl.dx - ufl.inner(c2_0, q2) * ufl.dx + (dt/2) * M * ufl.inner(ufl.grad(mu2 + mu2_0), ufl.grad(q2)) * ufl.dx
F3 = ufl.inner(mu2, v2) * ufl.dx - ufl.inner(dfdc2, v2) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c2), ufl.grad(v2)) * ufl.dx
# F0 = 3 * ufl.inner(c1, q1) * ufl.dx - 4 * ufl.inner(c1_0, q1) * ufl.dx + ufl.inner(c1__1, q1) * ufl.dx + (2*  dt) * M * ufl.inner(ufl.grad(mu1), ufl.grad(q1)) * ufl.dx
# F1 = ufl.inner(mu1, v1) * ufl.dx - ufl.inner(dfdc1, v1) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c1), ufl.grad(v1)) * ufl.dx
# F2 = 3 * ufl.inner(c2, q2) * ufl.dx - 4 * ufl.inner(c2_0, q2) * ufl.dx + ufl.inner(c2__1, q2) * ufl.dx + (2 * dt) * M * ufl.inner(ufl.grad(mu2 + mu2_0), ufl.grad(q2)) * ufl.dx
# F3 = ufl.inner(mu2, v2) * ufl.dx - ufl.inner(dfdc2, v2) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c2), ufl.grad(v2)) * ufl.dx
F = F0 + F1 + F2 + F3

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(domain.comm, problem)
# Create Newton Solver and set nonlinear solver tolerances
solver = NewtonSolver(domain.comm, problem)
solver.convergence_criterion = "incremental"  # (or "residual" depending on your preference)
solver.rtol = 1e-10       # relative tolerance
solver.atol = 1e-10       # absolute tolerance
solver.max_it = 30       # maximum Newton iterations
# Set up PETSc linear solver options
ksp = solver.krylov_solver
opts = PETSc.Options()  # Access PETSc options
option_prefix = ksp.getOptionsPrefix()
# Option set for ASM preconditioning with LU on subproblems and overlap 2:
opts[f"{option_prefix}pc_type"] = "asm"
opts[f"{option_prefix}sub_ksp_type"] = "preonly"
opts[f"{option_prefix}sub_pc_type"] = "lu"
opts[f"{option_prefix}pc_asm_overlap"] = 2
ksp.setFromOptions()

# Post-process
# Save solution to file
t = 0.0
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "ch_3p_fh.bp"

# Create named functions for output
c1_out = u.sub(0).collapse()
c2_out = u.sub(2).collapse()
c1_out.name = "c1"
c2_out.name = "c2"

log.set_log_level(log.LogLevel.INFO)

with io.VTXWriter(domain.comm, filename, [c1_out, c2_out], engine="BP4") as vtx:
    # Project initial condition
    vtx.write(t)
    
    # Time-stepping
    u_0.x.array[:] = u.x.array
    u__1.x.array[:] = u_0.x.array
    while t < T:
        t += dt
        r = solver.solve(u)
        print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        u__1.x.array[:] = u_0.x.array
        u_0.x.array[:] = u.x.array
        
        # Update output functions properly
        c1_out.x.array[:] = u.sub(0).collapse().x.array
        c2_out.x.array[:] = u.sub(2).collapse().x.array
        vtx.write(t)