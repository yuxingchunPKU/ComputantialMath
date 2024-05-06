'''
理想气体的Euler格式
'''
import matplotlib.pyplot as plt
import numpy as np
import Fluid1d as Fd
import NumericalFlux as NF
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.append("../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
'''
定义 单元和边界单元
需要单元和边界单元的对应关系
边界单元和单元的对应关系
在特殊的边界单元上做单元的标记 以方便做边界条件
在单元上放置守恒量 在边界单元上放置通量
计算通量
计算时间步长
更新守恒量
'''

'''
多介质单元的新增步

'''

# 初值函数
def init_fluid_data(U_data, Ele_c):
    for i in range(len(U_data)):
        if Ele_c[i] < 0.5:
            rho = 1
            u = 0
            P = 1.0
            U_temp = np.zeros(U_data[i].DIM + 2)
            U_temp[0] = rho
            U_temp[1] = rho * u
            U_temp[2] = P / (U_data[i].gamma - 1) + 0.5 * rho * u * u
            U_data[i].set_U(U_temp)
        else:
            rho = 0.125
            u = 0
            P = 0.1
            U_temp = np.zeros(U_data[i].DIM + 2)
            U_temp[0] = rho
            U_temp[1] = rho * u
            U_temp[2] = P / (U_data[i].gamma - 1) + 0.5 * rho * u * u
            U_data[i].set_U(U_temp)

DIM = 1
CFL = 0.2
U_init = 1e-8 * np.ones(DIM + 2, dtype=float)
gamma = 1.4
x_min = 0
x_max = 1
Nx = 400
max_t = 0.15
point = np.linspace(0, 1, Nx + 1)
n_edge = np.shape(point)[0]
n_ele = n_edge - 1
ele_vol = (x_max - x_min) / n_ele
edge_mark = np.zeros(n_edge, dtype=int)
edge_mark[0] = 1
edge_mark[-1] = 1

ele2edge = np.array([np.arange(0, n_edge - 1), np.arange(1, n_edge)])
ele2edge = ele2edge.transpose()
ele2point = ele2edge
# edge2cell = np.array([n_edge,2])
edge2cell = np.array([np.arange(-1, n_ele), np.arange(0, n_ele + 1)])
edge2cell[-1, -1] = -1
edge2cell = edge2cell.transpose()
# 将向量组合在一起 可能不再是array了
ele_centroid = 0.5 * (point[0:-1] + point[1:])

# 对数据进行初始化
U_data = []
for i in range(n_ele):
    U_data.append(Fd.Fluid_1d(U_init))

# 守恒量 初始值
init_fluid_data(U_data, ele_centroid)

t = 0
while (t < max_t):
    # 计算时间步长
    max_speed = 0
    for i in range(n_ele):
        max_speed = max(max_speed, U_data[i].max_speed())

    dt = CFL * ele_vol / max_speed

    # 计算数值通量
    Flux_vec = np.zeros([DIM + 2, n_edge])
    # lax-Friedriches
    for i in range(n_edge):
        if edge2cell[i, 0] == -1:
            Ur = U_data[edge2cell[i, 1]]
            Ul = Ur
        elif edge2cell[i, 1] == -1:
            Ul = U_data[edge2cell[i, 0]]
            Ur = Ul
        else:
            Ul = U_data[edge2cell[i, 0]]
            Ur = U_data[edge2cell[i, 1]]

        NF1 = NF.NumericalFlux(Ul,Ur)
        Flux_vec[:, i] = NF1.HLLC()
    # 更新守恒量
    for i in range(n_ele):
        U_data[i].set_U(U_data[i].U - (dt / ele_vol) * (Flux_vec[:, ele2edge[i, 1]] - Flux_vec[:, ele2edge[i, 0]]))

    t = t + dt


Rho = np.zeros(n_ele)
V = np.zeros(n_ele)
P = np.zeros(n_ele)

for i in range(n_ele):
    Rho[i] = U_data[i].U[0]
    V[i] = U_data[i].U[1] / U_data[i].U[0]
    P[i] = U_data[i].p

rho_L = 1.0
u_L = 0.0
p_L = 1.0
gamma_L = 1.4
p_inf_L = 0.0
# right data
rho_R = 0.125
u_R = 0.0
p_R = 0.1
gamma_R = 1.4
p_inf_R = 0.0

x_L = -0.5
x_R = 0.5

N=200
tol = 1e-12
maxit = 1000
WL = Fdexact.Fluid(rho_L,u_L,p_L,gamma_L,p_inf_L)
WR = Fdexact.Fluid(rho_R,u_R,p_R,gamma_R,p_inf_R)
ERS_solver = ERS.ExactRiemannSolver(WL,WR,tol,maxit)
[p_star,u_star]=ERS_solver.solver()
Complete_solver = ERCS.Riemann_Complete_solve(p_star,u_star,WL,WR,t,x_L,x_R,N)
[x,rho,u,p] = Complete_solver.solve()

plt.figure()
plt.plot(x+0.5,rho,label='Exact')
plt.plot(ele_centroid, Rho,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$\rho$")
plt.savefig("rho.png",dpi=300)
plt.clf()
plt.plot(x+0.5,u,label='Exact')
plt.plot(ele_centroid, V,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$u$")
plt.savefig("u.png",dpi=300)
plt.clf()
plt.plot(x+0.5,p,label='Exact')
plt.plot(ele_centroid, P,label='Numerical')
plt.grid()
plt.legend()
plt.ylabel(r"$p$")
plt.savefig("p.png",dpi=300)
plt.clf()