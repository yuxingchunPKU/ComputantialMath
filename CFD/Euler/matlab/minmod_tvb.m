function out = minmod_tvb(a, b, c, M, dx)
%MINMOD_TVB 带 TVB 修正的向量化 minmod 限制器
%
%   a 为候选斜率，b、c 为单侧差分；M = 0 时为经典 minmod。
%   对应 Python 版本 euler_unified/numerics.py 中的 minmod。
if nargin < 4 || isempty(M)
    M = 0.0;
end
if nargin < 5 || isempty(dx)
    dx = 1.0;
end

bc_min = min(b .* sign(a), c .* sign(a));
small = abs(a) <= M * dx * dx;
limited = sign(a) .* max(0.0, min(abs(a), bc_min));
out = limited;
out(small) = a(small);
end
