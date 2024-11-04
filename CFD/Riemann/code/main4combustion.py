import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import FluidReact as FR
import Exact_Riemann_Combustion as ERC

# case1 strong detonation
# rhoL = 1.57861
# uL = 2799.82
# pL = 7.70752e6
# qL = 0.0
# gammaL = 1.27
# rhoR = 0.601
# uR = 0.0
# pR = 1.0e5
# qR = 242000/0.018
# gammaR = 1.27

# case2 C-J detonation
rhoL = 0.974514
uL = 1586.05
pL = 4.04266e6
qL = 0.0
gammaL = 1.27

rhoR = 0.601
uR = 0.0
pR = 1.0e5
qR = 242000/0.018
gammaR = 1.27

t_end = 0.0005
N = 1000
x_c = 4.0
x_L = 0.0
x_R = 8.0

N = 1200
tol = 1e-12
maxit = 1000
WL = FR.FluidReact(rhoL,uL,pL,gammaL,qL)
WR = FR.FluidReact(rhoR,uR,pR,gammaR,qR)
ERC_solver = ERC.ExactRiemannSolverDetonation(WL,WR,tol,maxit,False)
[rho_cj,u_cj,p_cj] = ERC_solver.CJ_data()
[p_star,u_star] = ERC_solver.solver()
# 完整解
Complete_solver = ERC.RiemannCompleteSolution(p_star,u_star,WL,WR,t_end,x_L,x_R,x_c,N,False)
[x,rho,u,p] = Complete_solver.complete_solution()

plt.plot(x, rho, label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L, x_R])
plt.ylabel(r"$\rho$")
plt.show()
plt.clf()
plt.plot(x, u, label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L, x_R])
plt.ylabel(r"$u$")
plt.show()
plt.clf()
plt.plot(x, p, label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L, x_R])
plt.ylabel(r"$p$")
plt.show()
plt.clf()