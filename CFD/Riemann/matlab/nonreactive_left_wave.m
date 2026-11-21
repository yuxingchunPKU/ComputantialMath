function wave = nonreactive_left_wave(W, p_star, u_star)
%NONREACTIVE_LEFT_WAVE WL 到中间状态的非反应波（激波或稀疏波）
%
%   wave 结构体包含 name、type、speeds（动态字段）和 rho_star。
g = W.gamma;
rho0 = W.rho;
u0 = W.u;
p0 = W.p;
pinf = 0.0;
if isprop(W, 'p_inf') && ~isempty(W.p_inf)
    pinf = W.p_inf;
end
mu2 = (g - 1) / (g + 1);

if p_star >= p0
    A = 2.0 / ((g + 1) * rho0);
    B = (g - 1) * (p0 + pinf) / (g + 1);
    speed = u0 - (1.0 / rho0) * sqrt((B + pinf + p_star) / A);
    rho_star = rho0 * ((g + 1) * (p_star + pinf) + (g - 1) * (p0 + pinf)) / ...
                      ((g - 1) * (p_star + pinf) + (g + 1) * (p0 + pinf));
    wave = struct('name', '左波', 'type', '激波', ...
                  'speeds', struct('S_L', speed), 'rho_star', rho_star);
else
    a0 = W.sound_speed();
    a_star = a0 * ((p_star + pinf) / (p0 + pinf)) ^ ((g - 1) / (2 * g));
    rho_star = rho0 * ((p_star + pinf) / (p0 + pinf)) ^ (1.0 / g);
    speeds = struct('S_head', u0 - a0, 'S_tail', u_star - a_star);
    wave = struct('name', '左波', 'type', '稀疏波', ...
                  'speeds', speeds, 'rho_star', rho_star);
end
end
