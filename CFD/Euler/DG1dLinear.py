import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.append("../Riemann")


def bais(x,x_i,dx):
    N = len(x)
    M_b = 3
    B = np.zeros([M_b,N],dtype=float)
    B[0,:] = 1*np.ones(N,dtype=float)
    B[1,:] = (x-x_i)/(0.5*dx)
    B[2,:] = ((x-x_i)/(0.5*dx))**2 -1/3
    return B

def bais_x(x,x_i,dx):
    N = len(x)
    M_b = 3
    B = np.zeros([M_b,N],dtype=float)
    B[1,:] = np.ones(N,dtype=float)/(0.5*dx)
    B[2,:] = 2*(x-x_i)/(0.5*dx)*(np.ones(N,dtype=float)*1/(0.5*dx))
    return B

def init_fluid_data(x):
    return np.sin(2*np.pi*x)
    # return 1.0*np.ones(np.shape(x))

def lax_friedrichs_flux(ul, ur, a):
    return 0.5 * (a * (ul + ur) - np.abs(a) * (ur - ul))


M_b=3
a=1.0
DIM = 1
CFL = 0.1
x_min = 0
x_max = 1
Nx = 40
max_t = 0.2
point = np.linspace(x_min, x_max, Nx + 1)
n_edge = np.shape(point)[0]
n_ele = n_edge - 1
ele_vol = (x_max - x_min) / n_ele
edge_mark = np.ones(n_edge, dtype=int)
edge_mark[0] = -1
edge_mark[-1] = 0

ele2edge = np.array([np.arange(0, n_edge - 1), np.arange(1, n_edge)])
ele2edge = ele2edge.transpose()
ele2point = ele2edge
edge2cell = np.array([np.arange(-1, n_ele), np.arange(0, n_ele + 1)])
edge2cell[-1, -1] = -1
edge2cell = edge2cell.transpose()
ele_centroid = 0.5 * (point[0:-1] + point[1:])

# 对数据进行初始化
U_data = np.zeros([M_b,Nx],dtype=float)

'''
需要对初始值解方程
'''
W = 3
x_int_ref, wi = roots_legendre(W)
wi *= 0.5
for i in range(Nx):
    x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
    B = bais(x_int, ele_centroid[i], ele_vol)
    M = (B*wi)@np.transpose(B)*ele_vol
    U = ele_vol*np.dot(B*init_fluid_data(x_int),wi)
    U_data[:,i] = np.linalg.inv(M)@U

t = 0
Minv=np.array([[1,0],[0,3]])

is_stop = False
while (not is_stop):
    U_data_cp = U_data
    max_speed = a
    def step_forward(U_data):
        dU = np.zeros([M_b,Nx],dtype=float)
        edge_bias_r = bais(point[0:-1],ele_centroid,ele_vol)
        edge_bias_l = bais(point[1:],ele_centroid,ele_vol)
        Ul = np.zeros(n_edge,dtype=float)
        Ur = np.zeros(n_edge,dtype=float)
        Ul[1:] = np.sum(edge_bias_l * U_data,axis=0)
        Ur[0:-1] = np.sum(edge_bias_r * U_data,axis=0)
        Ul[0] = Ul[-1]
        Ur[-1] = Ur[0]
        Fu_edge = lax_friedrichs_flux(Ul,Ur,a)
        for i in range(n_ele):
            # 左右端点的坐标
            point_lr = [ele_centroid[i]-0.5*ele_vol,ele_centroid[i]+0.5*ele_vol]
            bais_lr = bais(point_lr,ele_centroid[i],ele_vol)
            V1 = Fu_edge[i+1]*bais_lr[:,1]-Fu_edge[i]*bais_lr[:,0]
            '''
            取积分点
            计算基函数在积分点上的值
            组合成函数值
            计算导数的梯度值
            '''
            x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
            bais_val = bais(x_int,ele_centroid[i],ele_vol)
            bais_val_x = bais_x(x_int,ele_centroid[i],ele_vol)
            u = U_data[:,i]@bais_val
            Fu = a*u #计算通量
            V2= -ele_vol*(bais_val_x*wi)@Fu
            '''
            计算矩阵 M
            '''
            M=(bais_val * wi) @ np.transpose(bais_val) * ele_vol
            dU[:,i] = np.linalg.inv(M)@(V1+V2)

        return dU
    dt = CFL * ele_vol / max_speed
    if (t + dt >= max_t and t < max_t):
        dt = max_t - t
        is_stop = True
    t = t + dt
    # 第一个时间步
    U_data = U_data - dt * step_forward(U_data)
    # 第二个时间步
    U_data = 0.75*U_data_cp + 0.25*(U_data-dt*step_forward(U_data))
    # 第三个时间步
    U_data = (1/3)*U_data_cp + (2/3)*(U_data-dt*step_forward(U_data))

Bc = bais(ele_centroid, ele_centroid, ele_vol)
U_out = np.sum(Bc*U_data,axis=0)

U_exact = np.sin(2*np.pi*(ele_centroid-a*max_t))
plt.plot(ele_centroid,U_out,'*-',label='DG P2')
plt.plot(ele_centroid,U_exact,label='exact')
plt.legend()
plt.show()


