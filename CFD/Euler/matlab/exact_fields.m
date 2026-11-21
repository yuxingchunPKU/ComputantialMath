function [rho, u, p] = exact_fields(cfg, x, t)
%EXACT_FIELDS 在 x 处返回时刻 t 的精确（参考）解
%
%   光滑输运问题使用解析解；其余情况调用精确 Riemann 求解器。
%   对应 Python 版本 euler_unified/initial.py 中的 exact_fields。
initial = cfg.initial;
if strcmp(initial.type, 'smooth')
    rho = initial.rho0 + initial.rho_amp * ...
          sin(initial.k * pi * (x - initial.u0 * t));
    u = initial.u0 * ones(size(x));
    p = initial.p0 * ones(size(x));
    return;
end

[left, right] = left_right_fluids(cfg);
[rho, u, p] = exact_riemann(left, right, x, t, initial.x_interface);
end

function [left, right] = left_right_fluids(cfg)
    if strcmp(cfg.medium, 'single')
        left = merge2(cfg.initial.left, cfg.fluids);
        right = merge2(cfg.initial.right, cfg.fluids);
    else
        left = merge2(cfg.initial.left, cfg.fluids.left);
        right = merge2(cfg.initial.right, cfg.fluids.right);
    end
end

function out = merge2(a, b)
    out = a;
    fns = fieldnames(b);
    for k = 1:numel(fns)
        out.(fns{k}) = b.(fns{k});
    end
end
