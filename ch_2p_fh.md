# Cahn-Hilliard Equation using Flory-Huggins

This is the implementation for Cahn-Hilliard Equation for two phase separation in 2D using FEniCSx.
The boundary-value problem is described by:
$$\frac{\partial c}{\partial t} - \nabla \cdot M \left( \nabla \left( \frac{\partial f}{\partial c} - {\epsilon^2} \Delta c \right) \right) = 0 \quad \in \Omega$$

We use split formulation to rephrase the fourth-order equation into two coupled second-order equations:
$$
\begin{aligned}
\frac{\partial c}{\partial t} - \nabla \cdot M \nabla \mu &= 0 \quad \in \Omega\\
\mu - \frac{\partial f}{\partial c} + {\epsilon^2} \Delta c &= 0 \quad \in \Omega
\end{aligned}
$$

## Variational formulation

### Weak formulation

The variational form of the equations are:
$$
\begin{aligned}
\int_\Omega \frac{\partial c}{\partial t} q dx + \int_\Omega M \nabla \mu \cdot \nabla q dx = 0 \\
\int_\Omega \mu v dx - \int_\Omega \frac{\partial f}{\partial c} v dx - \int_\Omega {\epsilon^2} \nabla c \cdot \nabla v dx = 0
\end{aligned}
$$

### Crank-Nicholson time stepper

The sampling of the first PDE at time $t_{n+1}$ is given by:
$$
\begin{aligned}
\int_\Omega \frac{c_{n+1} - c_n}{dt} q dx + \frac{1}{2} \int_\Omega M \nabla \left( \mu_{n+1} + \mu_n \right) \cdot \nabla q dx = 0 \\
\int_\Omega \mu_{n+1} v dx - \int_\Omega \frac{d f_{n+1}}{dc} v dx - \int_\Omega {\epsilon^2} \nabla c_{n+1} \cdot \nabla v dx = 0
\end{aligned}
$$

## Problem definition

- Domain: $\Omega = [0,1] \times [0,1]$
- Local energy: $f = A \left[ \frac{\phi_1}{N_1} \ln \phi_1 + \frac{\phi_2}{N_2} \ln \phi_2 + B \phi_1 \phi_2 \right]$
- Gradient energy coefficent: $\lambda = {\epsilon^2} = 1 \times 10^{-2}$
- Mobility coefficient: $M = 1$
- Flory-Huggins parameter: $\chi_{1,2} = 3$
- Length of polymer chains: $N_1 = N_2 = 1$