import numpy as np
'''
边界条件
'''
def boundary_value(U, label=1) -> np.array:
    '''
    默认是透射边界条件
    :param U: 输入向量
    :param label: 边界条件的类型 1是透射边界条件 2是反射边界条件
    :return: 边界值
    '''
    if label == 1:
        return U
    elif label == 2:
        U[1] = -U[1]
        return U
    else:
        return U