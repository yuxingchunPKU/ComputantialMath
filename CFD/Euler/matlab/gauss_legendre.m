function [x, w] = gauss_legendre(n)
%GAUSS_LEGENDRE n 点 Gauss-Legendre 求积节点与权重（参考区间 [-1, 1]）
%
%   通过 Golub-Welsch 算法（Jacobi 矩阵的特征分解）计算，
%   不依赖任何工具箱。权重满足 sum(w) = 2。
%   对应 scipy.special.roots_legendre。
if n < 1
    error('gauss_legendre:n', 'n must be positive');
end
if n == 1
    x = 0;
    w = 2;
    return;
end

k = (1:n - 1).';
beta = k ./ sqrt(4 * k .^ 2 - 1);
J = diag(beta, 1) + diag(beta, -1);

[V, D] = eig(J);
[d, idx] = sort(diag(D));
x = d(:);
w = 2 * (V(1, idx).') .^ 2;
w = w(:);
end
