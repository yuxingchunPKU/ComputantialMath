function [x, rho, u, p, star, waves, info] = compute_detonation(cs)
%COMPUTE_DETONATION 爆轰波黎曼问题（强爆轰或 CJ 爆轰 + 稀疏波）
d = cs.domain;
num = cs.numerical;
tol = num.tol;
maxit = num.maxit;
method = 'newton';
if isfield(num, 'method') && ~isempty(num.method)
    method = num.method;
end

WL = make_fluid(cs.left, true);
WR = make_fluid(cs.right, true);
solver = ExactRiemannSolverDetonation(WL, WR, tol, maxit, true);
if strcmpi(method, 'brent')
    [p_star, u_star] = solver.solver_brent();
else
    [p_star, u_star] = solver.solver();
end
p_cj = solver.p_cj;
u_cj = solver.u_cj;
rho_cj = solver.rho_cj;

complete = RiemannCompleteSolDet(p_star, u_star, WL, WR, ...
                                 d.t_end, d.x_L, d.x_R, d.x_c, d.N, true);
[x, rho, u, p] = complete.complete_solution();

left = nonreactive_left_wave(WL, p_star, u_star);
g = WR.gamma;
if p_star >= p_cj
    % 强爆轰：单道激波
    m = (p_star - WR.p) / (u_star - WR.u);
    S_det = WR.u + m / WR.rho;
    rho_star_R = WR.rho * (S_det - WR.u) / (S_det - u_star);
    right = struct('name', '右波', 'type', '强爆轰波', ...
                   'speeds', struct('S_det', S_det), ...
                   'rho_star', rho_star_R);
    wave_type = '强爆轰';
else
    % CJ 爆轰 + 稀疏波复合波
    a_cj = sqrt(g * p_cj / rho_cj);
    a_star_R = a_cj * (p_star / p_cj) ^ ((g - 1) / (2 * g));
    S_det = u_cj + a_cj;
    S_tail = u_star + a_star_R;
    rho_star_R = rho_cj * (p_star / p_cj) ^ (1.0 / g);
    speeds = struct('S_det', S_det, 'S_tail', S_tail);
    right = struct('name', '右波', 'type', 'CJ 爆轰 + 稀疏波', ...
                   'speeds', speeds, 'rho_star', rho_star_R);
    wave_type = 'CJ 爆轰';
end

contact = struct('name', '接触间断', 'type', '-', ...
                 'speeds', struct('S', u_star));
waves = {left, contact, right};

star = struct('p_star', p_star, 'u_star', u_star, ...
              'rho_star_L', left.rho_star, ...
              'rho_star_R', right.rho_star, ...
              'p_cj', p_cj, 'u_cj', u_cj, 'rho_cj', rho_cj, ...
              'right_wave_type', wave_type);

info = struct('method', method);
end
