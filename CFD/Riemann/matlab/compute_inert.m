function [x, rho, u, p, star, waves, info] = compute_inert(cs)
%COMPUTE_INERT 非反应刚性气体黎曼问题
%
%   cs 为算例结构体（由 JSON 配置读入并合并缺省值）。
d = cs.domain;
num = cs.numerical;
tol = num.tol;
maxit = num.maxit;
method = 'newton';
if isfield(num, 'method') && ~isempty(num.method)
    method = num.method;
end

WL = make_fluid(cs.left, false);
WR = make_fluid(cs.right, false);
solver = ExactRiemannSolver(WL, WR, tol, maxit);
if strcmpi(method, 'brent')
    [p_star, u_star] = solver.solver_brent();
else
    [p_star, u_star] = solver.solver();
end

complete = RiemannCompleteSolve(p_star, u_star, WL, WR, ...
                               d.t_end, d.x_L, d.x_R, d.N);
[x, rho, u, p] = complete.solve();

left = nonreactive_left_wave(WL, p_star, u_star);
right = nonreactive_right_wave(WR, p_star, u_star);
contact = struct('name', '接触间断', 'type', '-', ...
                 'speeds', struct('S', u_star));
waves = {left, contact, right};

star = struct('p_star', p_star, 'u_star', u_star, ...
              'rho_star_L', left.rho_star, ...
              'rho_star_R', right.rho_star);

info = struct('method', method);
end
