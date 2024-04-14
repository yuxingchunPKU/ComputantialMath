import Fluid as Fd
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
tol = 1e-12
maxit = 1000
rho_L = 1.0
u_L = 0.0
p_L = 1.0
gamma_L = 1.4

# right data
rho_R = 0.125
u_R = 0.0
p_R = 0.1
gamma_R = 1.667
p_inf_L = 0.0
p_inf_R = 0.0
t_final = 0.2

'''
计算区域
'''
x_L = -0.5
x_R = 0.5
N=400
WL = Fd.Fluid(rho_L,u_L,p_L,gamma_L,p_inf_L)
WR = Fd.Fluid(rho_R,u_R,p_R,gamma_R,p_inf_R)
ERS_solver = ERS.ExactRiemannSolver(WL,WR,tol,maxit)
[p_star,u_star]=ERS_solver.solver()
Complete_solver = ERCS.Riemann_Complete_solve(p_star,u_star,WL,WR,t_final,x_L,x_R,N)
[x,rho,u,p] = Complete_solver.solve()

plt.plot(x,rho,label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L,x_R])
plt.ylabel(r"$\rho$")
plt.savefig("rho.png",dpi=300)
plt.clf()
plt.plot(x,u,label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L,x_R])
plt.ylabel(r"$u$")
plt.savefig("u.png",dpi=300)
plt.clf()
plt.plot(x,p,label='Exact')
plt.grid()
plt.legend()
plt.xlim([x_L,x_R])
plt.ylabel(r"$p$")
plt.savefig("p.png",dpi=300)
plt.clf()
#plt.show()


