# Import Libraries
from mpi4py import MPI
from dolfinx import mesh, fem, io, log, default_real_type
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from pathlib import Path
from script import create_periodic_mesh

# Define simulation parameters
A = 3000
B = 3
epsilon_ = 1
lambda_ = epsilon_**2
M = 1.0
chi = 2.0
N1 = 1.0
N2 = 1.0
beta = 0.001
dt = 1.0e-07
T = 1.0e-06 # End time
num_steps = T / dt # Number of time steps

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=mesh.CellType.triangle)

# Create FunctionSpace
P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
ME = fem.functionspace(domain, mixed_element([P1, P1]))

# Define variational problem
# Define trial and test functions
u = fem.Function(ME)
# Previous solution
u0 = fem.Function(ME)
q, v = ufl.TestFunctions(ME)
# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Initial condition
u.x.array[:] = 0.0
rng = np.random.default_rng(2)
u.sub(0).interpolate(lambda x: 0.63 + 0.1 * (0.5 - rng.random(x.shape[1])))
u.x.scatter_forward()

# Compute df/dc
# c = ufl.variable(ufl.max_value(1e-6, c))
c = ufl.variable(c)
f = A * (c * ufl.ln(c) / N1 + (1 - c) * ufl.ln(1 - c) / N2 + B * c * (1 - c) + beta * ((1 / c) + (1 / (1 - c))))
dfdc = ufl.diff(f, c)

# Define residuals
F0 = ufl.inner(c, q) * ufl.dx - ufl.inner(c0, q) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu + mu0), ufl.grad(q)) * ufl.dx
F1 = ufl.inner(mu, v) * ufl.dx - ufl.inner(dfdc, v) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
F = F0 + F1

# Create NonlinearProblem
problem = NonlinearProblem(F, u)

# Create Newton Solver
# log.set_log_level(log.LogLevel.INFO)
solver = NewtonSolver(domain.comm, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer MUMPS, then superlu_dist, then default
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
if sys.hasExternalPackage("mumps") and not use_superlu:
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    # opts[f"{option_prefix}pc_factor_mat_ordering_type"] = "mumps_par"
    # opts[f"{option_prefix}mat_mumps_use_parallel_factorization"] = True
elif sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()

# # Create DG space for visualization
# V_out = fem.functionspace(domain, ("DG", 1))

# # Create DG functions for each component
# c_out = fem.Function(V_out)
# mu_out = fem.Function(V_out)

# Post-process
# Save solution to file
t = 0.0
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "out_ch_2p_fh.bp"

# Write using DG projection
with io.VTXWriter(domain.comm, filename, [u.sub(0)], engine="BP4") as vtx:
    # Project initial condition
    vtx.write(t)
    
    # Time-stepping
    u0.x.array[:] = u.x.array
    while t < T:
        t += dt
        r = solver.solve(u)
        print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        u0.x.array[:] = u.x.array
        vtx.write(t)