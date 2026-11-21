function [x, rho, u, p, star, waves, info] = compute_deflagration(cs)
%COMPUTE_DEFLAGRATION 爆燃波黎曼问题
d = cs.domain;
num = cs.numerical;
tol = num.tol;
maxit = num.maxit;
K = cs.deflagration.K;
Q = 2;
if isfield(cs.deflagration, 'Q') && ~isempty(cs.deflagration.Q)
    Q = cs.deflagration.Q;
end

WL = make_fluid(cs.left, true);
WR = make_fluid(cs.right, true);
solver = ExactRiemannSolverDeflagration(WL, WR, K, Q, tol, maxit, true);
[p_star, u_star, p0, rho0, V] = solver.solver_brent();

complete = RiemannCompleteSolDef(p_star, u_star, p0, rho0, WL, WR, ...
                                 d.t_end, d.x_L, d.x_R, d.x_c, d.N, true, ...
                                 V, solver.rho1, solver.u0);
[x, rho, u, p] = complete.complete_solution();

left = nonreactive_left_wave(WL, p_star, u_star);
right = nonreactive_right_wave(WR, p0, solver.u0);
flame = struct('name', '爆燃波', 'type', '火焰', ...
               'speeds', struct('V', V));
contact = struct('name', '接触间断', 'type', '-', ...
                 'speeds', struct('S', u_star));
waves = {left, contact, flame, right};

star = struct('p_star', p_star, 'u_star', u_star, ...
              'rho_star_L', left.rho_star, ...
              'rho1', solver.rho1, ...
              'p0', p0, 'u0', solver.u0, 'rho0', rho0, ...
              'V', V, 'K', K, 'Q', Q);

info = struct('method', 'brent');
end
