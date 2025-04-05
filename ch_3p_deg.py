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

# Define simulation parameters
R = 8.314
T = 300
V = 1.0
epsilon_ = 0.02
lambda_ = epsilon_**2
M = 1.0
chi12 = 2.0
chi13 = 2.0
chi23 = 2.0
N1 = 5.0
N2 = 5.0
N3 = 1.0
dt = 5.0e-05
T = 1.0e-03 # End time
num_steps = T / dt # Number of time steps

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([2, 1])], [10, 5], mesh.CellType.quadrilateral)

# Create FunctionSpace
P1 = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
ME = fem.functionspace(domain, mixed_element([P1, P1, P1, P1]))

# Define variational problem
# Define trial and test functions
u = fem.Function(ME)
# Previous solution
u_0 = fem.Function(ME)
q1, v1, q2, v2 = ufl.TestFunctions(ME)
# Split mixed functions
c1, mu1, c2, mu2 = ufl.split(u)
c1_0, mu1_0, c2_0, mu2_0 = ufl.split(u_0)

# Initial condition
u.x.array[:] = 0.0
rng_1 = np.random.default_rng(42)
u.sub(0).interpolate(lambda x: 0.3 + 0.02 * (0.5 - rng_1.random(x.shape[1])))
# rng_2 = np.random.default_rng(2)
u.sub(2).interpolate(lambda x: 1 - (0.3 + 0.02 * (0.5 - rng_1.random(x.shape[1]))))
u.x.scatter_forward()

# Mark variables and apply constraints
c1_var = ufl.variable(c1)
c2_var = ufl.variable(c2)

# Apply constraints with max_value
c1_safe = ufl.max_value(1e-6, c1_var)
c2_safe = ufl.max_value(1e-6, c2_var)
c3_safe = ufl.max_value(1e-6, 1 - c1_var - c2_var)  # Constraint for third component

# Define the free-energy using safe variables
f = (R * T / V) * (
    c1_safe * ufl.ln(c1_safe) / N1 + 
    c2_safe * ufl.ln(c2_safe) / N2 + 
    c3_safe * ufl.ln(c3_safe) / N3 + 
    chi12 * c1_var * c2_var + 
    chi13 * c1_var * c3_safe + 
    chi23 * c2_var * c3_safe
)

# Now differentiate with respect to the variables
dfdc1 = ufl.diff(f, c1_var)
dfdc2 = ufl.diff(f, c2_var)

# Define residuals
F0 = ufl.inner(c1, q1) * ufl.dx - ufl.inner(c1_0, q1) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu1 + mu1_0), ufl.grad(q1)) * ufl.dx
F1 = ufl.inner(mu1, v1) * ufl.dx - ufl.inner(dfdc1, v1) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c1), ufl.grad(v1)) * ufl.dx
F2 = ufl.inner(c2, q2) * ufl.dx - ufl.inner(c2_0, q2) * ufl.dx + (dt/2) * ufl.inner(ufl.grad(mu2 + mu2_0), ufl.grad(q2)) * ufl.dx
F3 = ufl.inner(mu2, v2) * ufl.dx - ufl.inner(dfdc2, v2) * ufl.dx - lambda_ * ufl.inner(ufl.grad(c2), ufl.grad(v2)) * ufl.dx
F = F0 + F1 + F2 + F3

# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer superlu_dist, then MUMPS, then default
if sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
elif sys.hasExternalPackage("mumps"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Post-process
# Save solution to file
t = 0.0
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "ch_3p_deg.bp"

# Create named functions for output
c1_out = u.sub(0)
c2_out = u.sub(2)
c1_out.name = "c1"
c2_out.name = "c2"

with io.VTXWriter(domain.comm, filename, [c1_out, c2_out], engine="BP4") as vtx:
    # Project initial condition
    vtx.write(t)
    
    # Time-stepping
    u_0.x.array[:] = u.x.array
    while t < T:
        t += dt
        r = solver.solve(u)
        print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        u_0.x.array[:] = u.x.array
        vtx.write(t)