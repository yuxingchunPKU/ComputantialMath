import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from fealpy.mesh import TriangleMesh
import Fluid2d as Fd
import NumericalFlux2d as NF
import sys
sys.path.append("../Riemann")
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS

from fealpy.mesh import MeshFactory as MF
from scipy.interpolate import griddata
box=[0,1,0,0.1]
mesh = MF.boxmesh2d(box=box,nx=200,ny=21,meshtype='tri')

# 从低维到高维几何体数量
n_node = mesh.number_of_nodes()
n_edge = mesh.number_of_edges()
n_cell = mesh.number_of_cells()

#边与单元的关系
cell2pnt =mesh.ds.cell_to_node()
cell2edge = mesh.ds.cell_to_edge()
edge2cell = mesh.ds.edge_to_cell()
isBDedge = mesh.ds.boundary_edge_flag() #是否是边界点的向量
edge_length = mesh.entity_measure('edge')
edge_norm = mesh.edge_unit_normal() #单位外法向
edge_center = mesh.entity_barycenter('edge')
tri_vol = mesh.entity_measure('cell')
tri_center = mesh.entity_barycenter('cell')
circumference_tri=edge_length[cell2edge[:,0]]+ edge_length[cell2edge[:,1]]+ edge_length[cell2edge[:,2]]
DIM = 2
CFL = 0.2
U_init = 1e-8 * np.ones(DIM + 2, dtype=float)
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
max_t = 0.15

'''
边界条件
'''
# 标记单元的边界
bd_label=np.zeros(n_edge)
for i in range(n_edge):
    if isBDedge[i]:
        #判断xy边界条件需要 xy 方向的两个坐标
        if abs(edge_center[i,1]-0.0)<1e-8 and \
           abs(edge_center[i,0]-0.0)>1e-8 and \
           abs(edge_center[i,0]-1.0)>1e-8:
            bd_label[i]=2
        elif abs(edge_center[i,1]-0.1)<1e-8 and \
           abs(edge_center[i,0]-0.0)>1e-8 and \
           abs(edge_center[i,0]-1.0)>1e-8:
            bd_label[i]=2
        else:
            bd_label[i]=1
'''
初始值
'''
U_data = np.zeros([DIM+2,n_cell],dtype=float)
for i in range(n_cell):
    if tri_center[i,0]<0.5:
        U_data[0,i] = rho_L
        U_data[1,i] = rho_L*u_L
        U_data[2,i] = 0
        U_data[3,i] = p_L/(gamma_L-1) + 0.5*rho_L*u_L*u_L
    else:
        U_data[0, i] = rho_R
        U_data[1, i] = rho_R * u_R
        U_data[2, i] = 0
        U_data[3, i] = p_R / (gamma_R - 1) + 0.5 * rho_R * u_R * u_R

t= 0
is_stop = False
while (not  is_stop):
    max_speed =0
    dU = np.zeros([DIM+2,n_cell],dtype=float)
    #计算各个边界的通量
    for i in range(n_edge):
        '''
        判断是否是边界
        '''
        Uin = np.zeros([DIM+2],dtype=float)
        Uout = np.zeros([DIM+2], dtype=float)
        if (isBDedge[i]):
            in_cell_idx = edge2cell[i,0]
            Uin[:]=U_data[:,in_cell_idx]
            # 透射边界条件
            if bd_label[i]==1:
                Uout[:] = Uin[:]
            elif bd_label[i]==2:
                Uout[:] = Uin[:]
                # 边界条件取错了
                Uout[2] = -Uout[2]
        else:
            in_cell_idx = edge2cell[i,0]
            out_cell_idx = edge2cell[i,1]
            Uin[:] = U_data[:, in_cell_idx]
            Uout[:] = U_data[:, out_cell_idx]

        Uin2d = Fd.Fluid_2d(Uin)
        Uout2d = Fd.Fluid_2d(Uout)
        #计算最大特征速度
        max_speed = max(max_speed,Uin2d.max_speed())
        max_speed = max(max_speed,Uout2d.max_speed())
        NF2 = NF.NumericalFlux(Uin2d,Uout2d,edge_norm[i,:])
        # 计算通量
        Flux= NF2.LLF()
        Flux = Flux*edge_length[i]
        # 更新
        dU[:,edge2cell[i, 0]] += Flux
        if (not isBDedge[i]):
            dU[:,edge2cell[i,1]] -= Flux

    # 计算时间步长
    dt = (CFL/max_speed)*np.min(tri_vol/circumference_tri)
    # dt = CFL*np.min(tri_vol)/(max_speed*np.max(edge_length))
    if (t+dt>=max_t and t<max_t):
        dt = max_t-t
        is_stop= True
    t = t + dt
    # 更新守恒量
    U_data=U_data - dt*(dU/tri_vol)
    print("dt="+str(dt)+",   "+"t="+str(t))

x_L = -0.5
x_R = 0.5
N=1000
tol = 1e-12
maxit = 1000
WL = Fdexact.Fluid(rho_L,u_L,p_L,gamma_L,p_inf_L)
WR = Fdexact.Fluid(rho_R,u_R,p_R,gamma_R,p_inf_R)
ERS_solver = ERS.ExactRiemannSolver(WL,WR,tol,maxit)
[p_star,u_star]=ERS_solver.solver()
Complete_solver = ERCS.Riemann_Complete_solve(p_star,u_star,WL,WR,t,x_L,x_R,N)
[x,rho,u,p] = Complete_solver.solve()

InterX=np.linspace(0,1,1000)
InterY=0.05*np.ones(1000)

Rho = np.zeros(n_cell)
V = np.zeros(n_cell)
P = np.zeros(n_cell)

for i in range(n_cell):
    U = Fd.Fluid_2d(U_data[:,i])
    Rho[i] =U.rho
    V[i] = np.linalg.norm(U.u)
    P[i] = U.p

InterRho = griddata(tri_center,U_data[0,:],(InterX,InterY),method='nearest')
InterU = griddata(tri_center,V,(InterX,InterY),method='nearest')
InterP = griddata(tri_center,P,(InterX,InterY),method='nearest')

plt.plot(InterX,InterRho,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,rho,label='Exact')
plt.show()

plt.clf()
plt.plot(InterX,InterU ,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,u,label='Exact')
plt.show()

plt.clf()
plt.plot(InterX,InterP,color='blue', marker='o',linewidth=0,markerfacecolor='none')
plt.plot(x+0.5,p,label='Exact')
plt.show()
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# plt.show()

