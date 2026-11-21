function out = minmod2(a, b)
%MINMOD2 有限体积斜率使用的两个参数的经典 minmod 限制器
out = zeros(size(a));
mask = a .* b > 0.0;
out(mask) = sign(a(mask)) .* min(abs(a(mask)), abs(b(mask)));
end
