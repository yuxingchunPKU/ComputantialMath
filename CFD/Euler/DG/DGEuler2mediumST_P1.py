'''
实现一个P2元的两相流模拟
函数是光滑的三角函数
自然边界条件 激波管问题
'''
#!基函数的名字打错了
import math
import numpy as np
from scipy.special import roots_legendre
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../../Riemann")
import Fluid1d as Fd1d
import NumericalFlux as NF
import Fluid as Fdexact
import Exact_Riemann_solver as ERS
import Exact_Riemann_complete_solve as ERCS
matplotlib.use('TkAgg')

# minmod 函数
def minmod(a,b,c,M=10,dx=1):
    bc_min = np.minimum(b*np.sign(a),c*np.sign(a))
    return a*(np.abs(a) <= M*dx**2)+(np.sign(a)*np.maximum(0,np.minimum(np.abs(a),bc_min)))*(np.abs(a) > M*dx**2)

def bais(x,x_i,dx):
    N_x = len(x)
    B = np.zeros([M_b, N_x], dtype=float)
    B[0, :] = 1 * np.ones(N_x, dtype=float)
    B[1, :] = (x - x_i) / (0.5 * dx)
    return B
def bais_single(x,x_i,dx):
    B = np.zeros(M_b, dtype=float)
    B[0] = 1
    B[1] = (x - x_i) / (0.5 * dx)
    return B

def bais_x(x,x_i,dx):
    N_x = len(x)
    B = np.zeros([M_b, N_x], dtype=float)
    B[1, :] = np.ones(N_x, dtype=float) / (0.5 * dx)
    return B

def bais_x_single(x,x_i,dx):
    B = np.zeros(M_b, dtype=float)
    B[1] = 1 / (0.5 * dx)
    return B
# 两相的初值
def init_fluid_data(U_data,Ele_c):
    for i in range(n_ele):
        # 初始时刻界面的位置在 0.5
        if Ele_c[i] < 0.5:
            rho = 1.0
            u = 0
            P = 1.0
            U_data[0,:,i,0] = rho
            U_data[1,:,i,0] = rho * u
            U_data[2,:,i,0] = P / (gamma0 - 1) + 0.5 * rho * u * u
            U_data[:,:,i,1] = 0
        else:
            rho = 0.125
            u = 0
            P = 0.1
            U_data[:,:,i,0] = 0.0
            U_data[0,:,i,1] = rho
            U_data[1,:,i,1] = rho * u
            U_data[2,:,i,1] = P / (gamma1 - 1) + 0.5 * rho * u * u
def interface_cell_idx(phi_input):
    # 标记界面点和界面单元 小于等于0的点是相0 大于0的点是相1
    N_edge = len(phi_input)
    point_label[:] = np.zeros(N_edge,dtype=int)*((phi_input<=0))+np.ones(N_edge,dtype=int)*((phi_input>0))
    point_label_add = point_label[0:-1]+point_label[1:] # 两个相加 0相0 1是界面单元 2是相1
    ele_label[:] = point_label_add
    idx_out = np.where(ele_label == 1)[0][0]
    return idx_out
def interface_cell_idx_new(phi_input):
    #界面移动之后的标记
    N_edge = len(phi_input)
    point_label_new[:] = np.zeros(N_edge,dtype=int)*((phi_input<=0))+np.ones(N_edge,dtype=int)*((phi_input>0))
    point_label_add = point_label_new[0:-1]+point_label_new[1:]
    ele_label_new[:] = point_label_add
    idx_out = np.where(ele_label_new == 1)[0][0]
    return idx_out
def update_phase_info(x_i):
    # 更新相的信息 相的体积和质心
    for i in range(n_ele):
        match ele_label[i]:
            case 0:
                phase_vol[0,i] = ele_vol
                phase_vol[1,i] = 0
                phase_centroid[0,i] = ele_centroid[i]
                phase_centroid[1,i] = ele_centroid[i]
            case 1:
                phase_vol[0,i] = math.fabs(x_i-point[i])
                phase_vol[1,i] = math.fabs(point[i+1]-x_i)
                phase_centroid[0,i] = (x_i + point[i])/2
                phase_centroid[1,i] = (point[i+1] + x_i)/2
            case 2:
                phase_vol[0,i] = 0
                phase_vol[1,i] = ele_vol
                phase_centroid[0,i] = ele_centroid[i]
                phase_centroid[1,i] = ele_centroid[i]
def update_phase_info_new(x_i):
    # 更新相的信息 相的体积和质心
    for i in range(n_ele):
        match ele_label_new[i]:
            case 0:
                phase_vol_new[0,i] = ele_vol
                phase_vol_new[1,i] = 0
                phase_centroid_new[0,i] = ele_centroid[i]
                phase_centroid_new[1,i] = ele_centroid[i]
            case 1:
                phase_vol_new[0,i] = math.fabs(x_i-point[i])
                phase_vol_new[1,i] = math.fabs(point[i+1]-x_i)
                phase_centroid_new[0,i] = (x_i + point[i])/2
                phase_centroid_new[1,i] = (point[i+1] + x_i)/2
            case 2:
                phase_vol_new[0,i] = 0
                phase_vol_new[1,i] = ele_vol
                phase_centroid_new[0,i] = ele_centroid[i]
                phase_centroid_new[1,i] = ele_centroid[i]
def update_interface_info():
    # 更新界面单元的信息
    IC_v = np.sum(phase_vol[:, IC_idx - 1:IC_idx + 2],axis=1)
    IC_c = np.sum(phase_centroid[:, IC_idx - 1:IC_idx + 2] * phase_vol[:, IC_idx - 1:IC_idx + 2],axis=1) / IC_v
    return IC_v,IC_c
def update_interface_info_new():
    # 更新界面单元的信息
    IC_v_n = np.sum(phase_vol_new[:, IC_idx - 1:IC_idx + 2],axis=1)
    IC_c_n = np.sum(phase_centroid_new[:, IC_idx - 1:IC_idx + 2] * phase_vol_new[:, IC_idx - 1:IC_idx + 2],axis=1) / IC_v_n
    return IC_v_n,IC_c_n
def compose_interface_val():
    #组合界面单元上的值
    for phase in range(2):
        Uf = np.zeros([DIM + 2, M_b, 3], dtype=float)
        for i in range(IC_idx - 1, IC_idx + 2):
            # 子单元上的积分点
            if phase_vol[phase, i] > eps:
                x_int_sc = 0.5 * phase_vol[phase, i] * x_int_ref + phase_centroid[phase, i]
                ele_bias_sc = bais(x_int_sc, phase_centroid[phase, i], phase_vol[phase, i])
                ele_bias_sc_IC = bais(x_int_sc, IC_centroid[phase], IC_vol[phase])
                Uf[:, :, i - (IC_idx - 1)] = (U_data[:, :, i, phase] @ ele_bias_sc) @ np.transpose(
                    ele_bias_sc_IC * wi) * phase_vol[phase, i]
        x_int_IC = 0.5 * IC_vol[phase] * x_int_ref + IC_centroid[phase]
        ele_bias_IC = bais(x_int_IC, IC_centroid[phase], IC_vol[phase])
        M = (ele_bias_IC * wi) @ np.transpose(ele_bias_IC) * IC_vol[phase]
        U_IC[:, :, phase] = np.sum(Uf, axis=2) / np.diagonal(M)
def updata_val_backup(p_v,p_c):
    #先计算临时界面单元的体积和重心
    temp_IC_v = np.sum(p_v[:,IC_idx_new-1:IC_idx_new+2],axis=1)
    temp_IC_c = np.sum(p_c[:,IC_idx_new-1:IC_idx_new+2]*p_v[:,IC_idx_new-1:IC_idx_new+2],axis=1)/temp_IC_v
    temp_IC_v_n = np.sum(phase_vol_new[:,IC_idx_new-1:IC_idx_new+2],axis=1)
    temp_IC_c_n = np.sum(phase_centroid_new[:,IC_idx_new-1:IC_idx_new+2]*phase_vol_new[:,IC_idx_new-1:IC_idx_new+2],axis=1)/temp_IC_v_n
    #更新临时界面单元上的值
    for phase in range(2):
        Uf = np.zeros([DIM+2,M_b,3],dtype=float)
        for i in range(IC_idx_new-1,IC_idx_new+2):
            if p_v[phase,i] > eps:
                x_int_sc = 0.5 * p_v[phase,i] * x_int_ref + p_c[phase,i]
                ele_bias_sc = bais(x_int_sc, p_c[phase,i], p_v[phase,i])
                ele_bias_sc_IC = bais(x_int_sc, temp_IC_c[phase], temp_IC_v[phase])
                Uf[:,:,i-(IC_idx_new-1)] = (U_data_cp[:,:,i,phase] @ ele_bias_sc) @ np.transpose(ele_bias_sc_IC*wi) * p_v[phase,i]
        x_int_IC = 0.5 * temp_IC_v[phase] * x_int_ref + temp_IC_c[phase]
        ele_bias_IC = bais(x_int_IC, temp_IC_c[phase], temp_IC_v[phase])
        M = (ele_bias_IC * wi) @ np.transpose(ele_bias_IC) * temp_IC_v[phase]
        U_IC_cp[:,:,phase] = np.sum(Uf,axis=2)/np.diagonal(M)
        #
        x_int_IC_n = 0.5 * temp_IC_v_n[phase] * x_int_ref + temp_IC_c_n[phase]
        ele_bias_IC_n = bais(x_int_IC_n, temp_IC_c_n[phase], temp_IC_v_n[phase])
        M_n = (ele_bias_IC_n * wi) @ np.transpose(ele_bias_IC_n) * temp_IC_v_n[phase]
        U_IC_cp[:,:,phase] = (U_IC_cp[:,:,phase]*np.diagonal(M))/np.diagonal(M_n)

def decompose_interface_val():
    #分解界面单元上的值到普通单元上去
    for phase in range(2):
        for i in range(IC_idx - 1, IC_idx + 2):
            # 取出总体的基函数 然后分配到单元上去
            if phase_vol_new[phase, i] > eps:
                x_int_sc = 0.5 * phase_vol_new[phase, i] * x_int_ref + phase_centroid_new[phase, i]
                ele_bias_sc = bais(x_int_sc, phase_centroid_new[phase, i], phase_vol_new[phase, i])
                ele_bias_sc_IC = bais(x_int_sc, IC_centroid_new[phase], IC_vol_new[phase])
                F_UIC = (U_IC[:,:,phase] @ ele_bias_sc_IC) @ np.transpose(ele_bias_sc * wi) * phase_vol_new[phase, i]
                Mi = (ele_bias_sc * wi) @ np.transpose(ele_bias_sc) * phase_vol_new[phase, i]
                U_data[:,:,i,phase] = F_UIC / np.diagonal(Mi)
            else:
                U_data[:,:,i,phase] = 0

def update_interface_val1(dt,dU_ic):
    # 更新界面单元上的守恒量
    for phase in range(2):
        x_int_IC = 0.5 * IC_vol[phase] * x_int_ref + IC_centroid[phase]
        ele_bias_IC = bais(x_int_IC, IC_centroid[phase], IC_vol[phase])
        x_int_IC_n = 0.5 * IC_vol_new[phase] * x_int_ref + IC_centroid_new[phase]
        ele_bias_IC_n = bais(x_int_IC_n, IC_centroid_new[phase], IC_vol_new[phase])
        M = (ele_bias_IC * wi) @ np.transpose(ele_bias_IC) * IC_vol[phase]
        M_n = (ele_bias_IC_n * wi) @ np.transpose(ele_bias_IC_n) * IC_vol_new[phase]
        U_IC[:,:,phase] = (U_IC[:,:,phase]*np.diagonal(M)-dt*dU_ic[:,:,phase])/np.diagonal(M_n)

def update_interface_val2(dt,dU_ic):
    #更新界面单元上的守恒量 是二阶格式的第二步
    for phase in range(2):
        x_int_IC = 0.5 * IC_vol[phase] * x_int_ref + IC_centroid[phase]
        ele_bias_IC = bais(x_int_IC, IC_centroid[phase], IC_vol[phase])
        x_int_IC_n = 0.5 * IC_vol_new[phase] * x_int_ref + IC_centroid_new[phase]
        ele_bias_IC_n = bais(x_int_IC_n, IC_centroid_new[phase], IC_vol_new[phase])
        M = (ele_bias_IC * wi) @ np.transpose(ele_bias_IC) * IC_vol[phase]
        M_n = (ele_bias_IC_n * wi) @ np.transpose(ele_bias_IC_n) * IC_vol_new[phase]
        U_IC[:,:,phase] = 0.5*(U_IC_cp[:,:,phase]*np.diagonal(M)+U_IC[:,:,phase]*np.diagonal(M)-dt*dU_ic[:,:,phase])/np.diagonal(M_n)

def TVB_limiter1(U_data,U_IC):
    #P1 TVB 限制器
    DU_p= np.zeros([DIM+2,n_ele,2],dtype=float)
    DU_m= np.zeros([DIM+2,n_ele,2],dtype=float)
    DU_p[:,0:-1,:] = U_data[:,0,1:,:]-U_data[:,0,0:-1,:]
    DU_m[:,1:,:] = U_data[:,0,1:,:]-U_data[:,0,0:-1,:]
    #靠近界面单元上的值 单独处理
    DU_p[:,IC_idx-2,0] = U_IC[:,0,0] - U_data[:,0,IC_idx-2,0]
    DU_m[:,IC_idx+2,1] = U_data[:,0,IC_idx+2,1] - U_IC[:,0,1]
    # 单元的边界上施加限制 单边的
    ele_bias_r = bais(point[1:], ele_centroid, ele_vol)
    U_ele_r0 = np.sum(ele_bias_r * U_data[:,:,:,0], axis=1)
    U_ele_r1 = np.sum(ele_bias_r * U_data[:,:,:,1], axis=1)
    dUp0 = U_ele_r0 - U_data[:,0,:,0]
    dUp1 = U_ele_r1 - U_data[:,0,:,1]
    dUp_mod0 = minmod(dUp0,DU_p[:,:,0],DU_m[:,:,0],M_para,ele_vol)
    dUp_mod1 = minmod(dUp1,DU_p[:,:,1],DU_m[:,:,1],M_para,ele_vol)
    U_data[:,1,0:IC_idx-1,0] = dUp_mod0[:,0:IC_idx-1]
    U_data[:,1,IC_idx+2:,1] = dUp_mod1[:,IC_idx+2:]
    # 界面单元上单独的限制器
    U0IC0 = U_IC[:,0,0]
    U0IC1 = U_IC[:,0,1]
    ele_bias_l_IC0 = bais_single(point[IC_idx-1], IC_centroid_new[0], IC_vol_new[0])
    ele_bias_r_IC1 = bais_single(point[IC_idx+2], IC_centroid_new[1], IC_vol_new[1])
    U_IC0_l = np.sum(ele_bias_l_IC0 * U_IC[:,:,0], axis=1)
    U_IC1_r = np.sum(ele_bias_r_IC1 * U_IC[:,:,1], axis=1)
    dUp_IC0 = U0IC0 - U_IC0_l
    dUp_IC1 = U_IC1_r - U0IC1
    # 代入限制器函数
    dUp_IC0_mod = minmod(dUp_IC0,DU_p[:,IC_idx-2,0],U_IC[:,0,1]-U_IC[:,0,0],M_para,IC_vol_new[0])
    dUp_IC1_mod = minmod(dUp_IC1,U_IC[:,0,1]-U_IC[:,0,0],DU_m[:,IC_idx+2,1],M_para,IC_vol_new[1])
    U_IC[:,1,0] = dUp_IC0_mod
    U_IC[:,1,1] = dUp_IC1_mod
    # 限制新的体积
def TVB_limiter1_subcell(U_data):
    #subcell limiter
    # 只针对子单元上的值进行限制 这个只能分开进行
    # IC0 是左端点 IC1是右端点
    DU_p= np.zeros([DIM+2,n_ele,2],dtype=float)
    DU_m= np.zeros([DIM+2,n_ele,2],dtype=float)
    DU_p[:,0:-1,:] = U_data[:,0,1:,:]-U_data[:,0,0:-1,:]
    DU_m[:,1:,:] = U_data[:,0,1:,:]-U_data[:,0,0:-1,:]
    for i in range(IC_idx-1,IC_idx+2):
        # 取边界上的值
        if phase_vol_new[0,i] > eps:
            ele_bias_l = bais_single(point[i],phase_centroid_new[0,i],phase_vol_new[0,i])
            U_SC_l = np.sum(ele_bias_l * U_data[:,:,i,0],axis=1)
            dUp0 = U_data[:,0,i,0] - U_SC_l
            if i != IC_idx_new:
                dUp_mod0 = minmod(dUp0,DU_p[:,i,0],DU_m[:,i,0],0,phase_vol_new[0,i])
                U_data[:,1,i,0] = dUp_mod0
            else:
                dUp_mod0 = minmod(dUp0,DU_m[:,i,0],DU_m[:,i,0],0,phase_vol_new[0,i])
                U_data[:,1,i,0] = dUp_mod0

        if phase_vol_new[1,i] > eps:
            ele_bias_r = bais_single(point[i+1],phase_centroid_new[1,i],phase_vol_new[1,i])
            U_SC_r = np.sum(ele_bias_r * U_data[:,:,i,1],axis=1)
            dUp1 = U_SC_r - U_data[:,0,i,1]
            if i != IC_idx_new:
                dUp_mod1 = minmod(dUp1,DU_p[:,i,1],DU_m[:,i,1],0,phase_vol_new[1,i])
                U_data[:,1,i,1] = dUp_mod1
            else:
                dUp_mod1 = minmod(dUp1,DU_p[:,i,1],DU_p[:,i,1],0,phase_vol_new[1,i])
                U_data[:,1,i,1] = dUp_mod1

def oneD_Riemann(Ul,Ur):
    # 调用多介质黎曼问题求解器 获得一维黎曼问题的精确解 并返回界面的速度
    UlIC1d = Fd1d.Fluid_1d(Ul,gamma=gamma0)
    UrIC1d = Fd1d.Fluid_1d(Ur,gamma=gamma1)
    WL = Fdexact.Fluid(UlIC1d.rho,UlIC1d.u,UlIC1d.p,gamma0,0)
    WR = Fdexact.Fluid(UrIC1d.rho,UrIC1d.u,UrIC1d.p,gamma1,0)
    ERS_solver = ERS.ExactRiemannSolver(WL, WR, tol, maxit)
    [p, u] = ERS_solver.solver()
    IC_flux[1] = p
    IC_flux[2] = p * u
    return u

def get_flux(U_data,max_speed):
    dU = np.zeros([DIM+2,M_b,Nx,2],dtype=float)
    dUIC = np.zeros([DIM+2,M_b,2],dtype=float)
    # 重构出单元边界上的值
    # 从0到倒数第二个顶点的右边界 从1到最后一个顶点的左边界
    edge_bias_r = bais(point[0:-1], ele_centroid, ele_vol)
    edge_bias_l = bais(point[1:], ele_centroid, ele_vol)
    Ul = np.zeros([DIM + 2, n_edge], dtype=float)
    Ur = np.zeros([DIM + 2, n_edge], dtype=float)

    #重构出边界上的值
    Ul[:, 1:IC_idx] = np.sum(edge_bias_l[:, 0:IC_idx - 1] * U_data[:, :, 0:IC_idx - 1,0], axis=1)
    Ur[:, 0:IC_idx - 1] = np.sum(edge_bias_r[:, 0:IC_idx - 1] * U_data[:, :, 0:IC_idx - 1,0], axis=1)
    Ul[:, IC_idx + 3:] = np.sum(edge_bias_l[:, IC_idx + 2:] * U_data[:, :, IC_idx + 2:,1], axis=1)
    Ur[:, IC_idx + 2:-1] = np.sum(edge_bias_r[:, IC_idx + 2:] * U_data[:, :, IC_idx + 2:,1], axis=1)
    # IC0的左端点 和 IC1的右端点 上的值
    edge_bias_r_IC0 = bais_single(point[IC_idx - 1], IC_centroid[0], IC_vol[0])
    edge_bias_l_IC1 = bais_single(point[IC_idx + 2], IC_centroid[1], IC_vol[1])
    Ur[:, IC_idx - 1] = np.sum(edge_bias_r_IC0 * U_IC[:,:,0], axis=1)
    Ul[:, IC_idx + 2] = np.sum(edge_bias_l_IC1 * U_IC[:,:,1], axis=1)
    # 周期边界
    Ul[:, 0] = Ur[:, 0]
    Ur[:, -1] = Ul[:, -1]
    # 计算通量
    Fu_edge = np.zeros([DIM + 2, n_edge], dtype=float)
    for i in range(n_edge):
        if i>=IC_idx and i<=IC_idx+1:
            pass
        else:
            gamma_temp = gamma0
            if i<IC_idx-1:
                gamma_temp = gamma0
            else:
                gamma_temp = gamma1
            Ul1d = Fd1d.Fluid_1d(Ul[:, i],gamma=gamma_temp)
            Ur1d = Fd1d.Fluid_1d(Ur[:, i],gamma=gamma_temp)
            NF1d = NF.NumericalFlux(Ul1d, Ur1d)
            max_speed = max(max_speed, Ul1d.max_speed())
            max_speed = max(max_speed, Ur1d.max_speed())
            Fu_edge[:, i] = NF1d.LLF()
    # 遍历非界面单元 在单元的内部计算增加量
    for i in range(n_ele):
        if i>=IC_idx-1 and i<=IC_idx+1:
            pass
        else:
            #建立单元内左右端点的值
            point_lr = [ele_centroid[i] - 0.5 * ele_vol, ele_centroid[i] + 0.5 * ele_vol]
            bais_lr = bais(point_lr, ele_centroid[i], ele_vol)
            V1 = np.outer(Fu_edge[:, i + 1], bais_lr[:, 1]) - np.outer(Fu_edge[:, i], bais_lr[:, 0])
            #内部的积分点值
            x_int = 0.5 * ele_vol * x_int_ref + ele_centroid[i]
            bais_val = bais(x_int, ele_centroid[i], ele_vol)
            bais_val_x = bais_x(x_int, ele_centroid[i], ele_vol)
            bais_val_x_wi = bais_val_x * wi
            u = np.zeros([DIM+2,W],dtype=float)
            gamma_temp = gamma0
            if i<IC_idx-1:
                u[:,:] = U_data[:, :, i,0]@ bais_val
                gamma_temp = gamma0
            else:
                u[:,:] = U_data[:, :, i,1]@ bais_val
                gamma_temp = gamma1
            V2 = np.zeros([DIM + 2, M_b], dtype=float)
            for j in range(W):
                U1d_in = Fd1d.Fluid_1d(u[:, j],gamma=gamma_temp)
                FU1d = U1d_in.flux()
                V2[:, :] -= ele_vol * np.outer(FU1d, bais_val_x_wi[:, j])
            M = (bais_val * wi) @ np.transpose(bais_val) * ele_vol
            if i < IC_idx - 1:
                dU[:, :, i,0] = (V1 + V2) / np.diagonal(M)
            else:
                dU[:, :, i,1] = (V1 + V2) / np.diagonal(M)
    # 计算界面单元内
    IC_edge_label = [IC_idx-1,IC_idx+2]
    for phase in range(2):
        point_lr_IC = [IC_centroid[phase] - 0.5 * IC_vol[phase], IC_centroid[phase] + 0.5 * IC_vol[phase]]
        bais_lr_IC = bais(point_lr_IC, IC_centroid[phase], IC_vol[phase])
        V1_IC = np.zeros([DIM + 2, M_b], dtype=float)
        gamma_temp = gamma0
        if phase == 0:
            gamma_temp = gamma0
            V1_IC += np.outer(IC_flux, bais_lr_IC[:, 1]) - np.outer(Fu_edge[:, IC_edge_label[phase]], bais_lr_IC[:, 0])
        elif phase == 1:
            gamma_temp = gamma1
            V1_IC += np.outer(Fu_edge[:, IC_edge_label[phase]], bais_lr_IC[:, 1]) - np.outer(IC_flux, bais_lr_IC[:, 0])
        # 单元内部的增量
        V2_IC = np.zeros([DIM + 2, M_b], dtype=float)
        x_int_IC = 0.5 * IC_vol[phase] * x_int_ref + IC_centroid[phase]
        bais_val_IC = bais(x_int_IC, IC_centroid[phase], IC_vol[phase])
        bais_val_x_IC = bais_x(x_int_IC, IC_centroid[phase], IC_vol[phase])
        bais_val_x_wi_IC = bais_val_x_IC * wi
        u_IC = U_IC[:,:,phase] @ bais_val_IC
        for j in range(W):
            U1d_in = Fd1d.Fluid_1d(u_IC[:, j],gamma=gamma_temp)
            FU1d_IC = U1d_in.flux()
            V2_IC[:, :] -= IC_vol[phase] * np.outer(FU1d_IC, bais_val_x_wi_IC[:, j])
        dUIC[:, :, phase] = (V1_IC + V2_IC)
    return dU,dUIC,max_speed


#建立主函数的
if __name__ == '__main__':
    # 限制器参数 TVM
    M_para = 0
    M_b = 2
    Nx = 200
    max_t = 0.15
    DIM = 1
    CFL = 0.1
    x_min = 0
    x_max = 1
    eps = 1e-12
    # 流体的参数信息
    max_speed = 0
    gamma0 = 1.4
    gamma1 = 1.4
    # 单元的几何信息
    point = np.linspace(x_min, x_max, Nx + 1)
    n_edge = np.shape(point)[0]
    n_ele = n_edge - 1
    ele_vol = (x_max - x_min) / n_ele
    ele_centroid = 0.5 * (point[0:-1] + point[1:])
    # 积分点的信息
    W = 4
    x_int_ref, wi = roots_legendre(W)
    wi *= 0.5
    # 对数据进行初始化
    U_data = np.zeros([DIM+2,M_b,Nx,2],dtype=float)
    init_fluid_data(U_data,ele_centroid)
    '''
    界面相关的几何信息
    '''
    IC_idx = -1  # 界面所但在单元的指标
    # IC_idx_new = -1
    point_label = -np.ones(n_edge, dtype=int)  # 顶点的标记
    point_label_new = -np.ones(n_edge, dtype=int)
    ele_label = -np.ones(n_ele, dtype=int)  # 单元的标记
    ele_label_new = -np.ones(n_ele, dtype=int)
    # 存界面单元的值
    U_IC = np.zeros([DIM+2,M_b,2],dtype=float)
    U_IC_cp = np.zeros([DIM+2,M_b,2],dtype=float)
    # 界面单元的体积和质心
    IC_vol = np.zeros(2, dtype=float)
    IC_centroid = np.zeros(2, dtype=float)
    IC_vol_new = np.zeros(2, dtype=float)
    IC_centroid_new = np.zeros(2, dtype=float)
    # 相体积和相重心 和备份
    phase_vol = np.zeros([2, n_ele], dtype=float)
    phase_centroid = np.zeros([2, n_ele], dtype=float)
    phase_vol_new = np.zeros([2, n_ele], dtype=float)
    phase_centroid_new = np.zeros([2, n_ele], dtype=float)

    # 界面上的通量（移动边界上的通量）
    IC_flux = np.zeros(DIM + 2, dtype=float)
    # 多介质黎曼问题的参数
    tol = 1e-12
    maxit = 1000
    # 界面的初始位置 与 水平集函数的初值 phi
    x_I = 0.5
    x_I_cp = 0.5
    phi = point - x_I
    #开始计时 准备运算
    t = 0
    is_stop = False
    iter = 0
    while (not is_stop):
        # 备份解
        U_data_cp = U_data.copy()
        x_I_cp = x_I
        #stage 1
        IC_idx = interface_cell_idx(phi)
        update_phase_info(x_I)
        IC_vol,IC_centroid=update_interface_info()
        #备份n时刻的体积和质心
        phase_vol_cp = phase_vol.copy()
        phase_centroid_cp = phase_centroid.copy()
        compose_interface_val()
        u_star = oneD_Riemann(U_IC[:,0,0], U_IC[:,0,1])
        max_speed = 0
        dU, dUIC, max_speed = get_flux(U_data, max_speed)
        # 计算时间步长
        dt = CFL * ele_vol / max_speed
        if (t + dt > max_t and t <= max_t):
            dt = max_t - t
            is_stop = True
        t += dt
        x_I_new = x_I + u_star * dt
        phi_new = point - x_I_new
        IC_idx_new = interface_cell_idx_new(phi_new)
        update_phase_info_new(x_I_new)
        IC_vol_new, IC_centroid_new = update_interface_info_new()
        U_data = U_data - dt * dU
        update_interface_val1(dt, dUIC)
        TVB_limiter1(U_data, U_IC)
        decompose_interface_val()
        updata_val_backup(phase_vol_cp,phase_centroid_cp)
        phi = phi_new
        x_I = x_I_new
        IC_idx = IC_idx_new
        #stage 2
        IC_idx = interface_cell_idx(phi)
        update_phase_info(x_I)
        IC_vol,IC_centroid=update_interface_info()
        compose_interface_val()
        u_star = oneD_Riemann(U_IC[:,0,0], U_IC[:,0,1])
        dU, dUIC, max_speed = get_flux(U_data, max_speed)
        x_I_new = 0.5*x_I_cp+0.5*(x_I + u_star * dt)
        phi_new = point - x_I_new
        IC_idx_new = interface_cell_idx_new(phi_new)
        update_phase_info_new(x_I_new)
        IC_vol_new,IC_centroid_new= update_interface_info_new()
        U_data = 0.5*U_data_cp+0.5*(U_data - dt * dU)
        update_interface_val2(dt, dUIC)
        TVB_limiter1(U_data, U_IC)
        decompose_interface_val()
        phi = phi_new
        x_I = x_I_new
        IC_idx = IC_idx_new







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
maxit = 1000
N=1000
WL = Fdexact.Fluid(rho_L,u_L,p_L,gamma_L,p_inf_L)
WR = Fdexact.Fluid(rho_R,u_R,p_R,gamma_R,p_inf_R)
ERS_solver = ERS.ExactRiemannSolver(WL,WR,eps,maxit)
[p_star,u_star]=ERS_solver.solver()
Complete_solver = ERCS.Riemann_Complete_solve(p_star,u_star,WL,WR,t,x_L,x_R,N)
[x,rho,u,p] = Complete_solver.solve()


#输出结果的展示
Bc = bais(ele_centroid, ele_centroid, ele_vol)
Rho0 = np.sum(Bc[0,:]*U_data[0,:,:,0],axis=0)
Rho1 = np.sum(Bc[0,:]*U_data[0,:,:,1],axis=0)
Rho = np.maximum(Rho0,Rho1)

plt.figure()
plt.plot(x+0.5,rho,label='Exact')
plt.plot(ele_centroid,Rho,'*-',label='DG P1')
plt.legend()
plt.show()
plt.clf()

