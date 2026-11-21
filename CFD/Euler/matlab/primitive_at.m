function [rho, u, p] = primitive_at(cfg, x)
%PRIMITIVE_AT 计算初始原始变量在 x 处的取值
%
%   支持三种初始条件：smooth、constant、sod。
%   对应 Python 版本 euler_unified/initial.py 中的 primitive_at。
x = x(:).';
initial = cfg.initial;

if strcmp(initial.type, 'smooth')
    rho = initial.rho0 + initial.rho_amp * sin(initial.k * pi * x);
    u = initial.u0 * ones(size(x));
    p = initial.p0 * ones(size(x));
    return;
end

left = initial.left;
right = initial.right;
if strcmp(initial.type, 'constant')
    rho = left.rho * ones(size(x));
    u = left.u * ones(size(x));
    p = left.p * ones(size(x));
    return;
end
if strcmp(initial.type, 'sod')
    mask = x < initial.x_interface;
    rho = right.rho * ones(size(x));
    u = right.u * ones(size(x));
    p = right.p * ones(size(x));
    rho(mask) = left.rho;
    u(mask) = left.u;
    p(mask) = left.p;
    return;
end
error('primitive_at:type', 'unsupported initial type: %s', initial.type);
end
