function [rho, u, p] = exact_riemann(left, right, x, t, x_interface)
%EXACT_RIEMANN 在网格 x 上采样精确 Riemann 解
%
%   left、right 为包含 rho、u、p、gamma 与可选 p_inf 的结构体；
%   初始间断位于 x_interface。
%   对应 Python 版本 euler_unified/numerics.py 中的 exact_riemann。
add_euler_paths();

p_inf_L = 0.0;
if isfield(left, 'p_inf') && ~isempty(left.p_inf)
    p_inf_L = left.p_inf;
end
p_inf_R = 0.0;
if isfield(right, 'p_inf') && ~isempty(right.p_inf)
    p_inf_R = right.p_inf;
end

WL = Fluid(left.rho, left.u, left.p, left.gamma, p_inf_L);
WR = Fluid(right.rho, right.u, right.p, right.gamma, p_inf_R);
solver = ExactRiemannSolver(WL, WR, 1.0e-12, 1000);
[p_star, u_star] = solver.solver();

x = x(:).';
x_L = x(1) - x_interface;
x_R = x(end) - x_interface;
complete = RiemannCompleteSolve(p_star, u_star, WL, WR, t, x_L, x_R, ...
                                max(numel(x) - 1, 1));
[xs, rho, u, p] = complete.solve();

% 网格点与完全解采样点仅有舍入差别时直接使用；否则做线性插值。
if numel(xs) ~= numel(x) || ...
        any(abs(xs(:) + x_interface - x(:)) > 1.0e-10 * max(1.0, max(abs(x))))
    [xs_sorted, order] = sort(xs);
    rho = interp1(xs_sorted, rho(order), x - x_interface, 'linear');
    u = interp1(xs_sorted, u(order), x - x_interface, 'linear');
    p = interp1(xs_sorted, p(order), x - x_interface, 'linear');
end
end
