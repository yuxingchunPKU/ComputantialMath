function out = apply_basis(B, U)
%APPLY_BASIS 在 n 个点上展开 DG 多项式
%
%   B : M x n，基函数在 n 个点的取值
%   U : 3 x M x n，DG 系数
%   返回 3 x n 的守恒量取值。
    M = size(B, 1);
    n = size(B, 2);
    if n == 0
        out = zeros(size(U, 1), 0);
        return;
    end
    Bb = reshape(B, 1, M, n);
    out = reshape(sum(Bb .* U, 2), size(U, 1), n);
end
