function R = char_matrix(uc, gas)
%CHAR_MATRIX Euler 方程的特征向量矩阵（用于 P2 特征变量限制器）
    [rho, u, p] = gas.primitive(uc);
    c = gas.sound_speed(uc);
    H = (uc(3) + p) / rho;
    R = [1.0, 1.0, 1.0;
         u - c, u, u + c;
         H - u * c, 0.5 * u * u, H + u * c] + 1.0e-12 * eye(3);
end
