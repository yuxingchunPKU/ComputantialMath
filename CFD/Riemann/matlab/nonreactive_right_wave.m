function wave = nonreactive_right_wave(W, p_star, u_star)
%NONREACTIVE_RIGHT_WAVE 中间状态到 WR 的非反应波（激波或稀疏波）
g = W.gamma;
rho0 = W.rho;
u0 = W.u;
p0 = W.p;
pinf = 0.0;
if isprop(W, 'p_inf') && ~isempty(W.p_inf)
    pinf = W.p_inf;
end

if p_star >= p0
    A = 2.0 / ((g + 1) * rho0);
    B = (g - 1) * (p0 + pinf) / (g + 1);
    speed = u0 + (1.0 / rho0) * sqrt((B + pinf + p_star) / A);
    rho_star = rho0 * ((g + 1) * (p_star + pinf) + (g - 1) * (p0 + pinf)) / ...
                      ((g - 1) * (p_star + pinf) + (g + 1) * (p0 + pinf));
    wave = struct('name', '右波', 'type', '激波', ...
                  'speeds', struct('S_R', speed), 'rho_star', rho_star);
else
    a0 = W.sound_speed();
    a_star = a0 * ((p_star + pinf) / (p0 + pinf)) ^ ((g - 1) / (2 * g));
    rho_star = rho0 * ((p_star + pinf) / (p0 + pinf)) ^ (1.0 / g);
    speeds = struct('S_tail', u_star + a_star, 'S_head', u0 + a0);
    wave = struct('name', '右波', 'type', '稀疏波', ...
                  'speeds', speeds, 'rho_star', rho_star);
end
end
