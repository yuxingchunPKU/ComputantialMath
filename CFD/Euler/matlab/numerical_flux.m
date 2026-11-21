function F = numerical_flux(UL, UR, gas, kind)
%NUMERICAL_FLUX 一维 Euler 方程组的数值通量
%
%   UL, UR : 3 x n 的左右状态
%   gas    : StiffenedGas 对象
%   kind   : 'llf' | 'rusanov' | 'hll' | 'hllc'
%   对应 Python 版本 euler_unified/numerics.py 中的 numerical_flux。
if nargin < 4 || isempty(kind)
    kind = 'llf';
end
kind = lower(kind);
if strcmp(kind, 'rusanov')
    kind = 'llf';
end

FL = gas.flux(UL);
FR = gas.flux(UR);
[rhoL, uL, pL] = gas.primitive(UL);
[rhoR, uR, pR] = gas.primitive(UR);
aL = gas.sound_speed(UL);
aR = gas.sound_speed(UR);

if strcmp(kind, 'llf')
    alpha = max(abs(uL) + aL, abs(uR) + aR);
    F = 0.5 * (FL + FR) - 0.5 * alpha .* (UR - UL);
    return;
end

% Davis 波速估计，HLL 与 HLLC 共用
SL = min(uL - aL, uR - aR);
SR = max(uL + aL, uR + aR);

if strcmp(kind, 'hll')
    denom = SR - SL;
    denom(abs(denom) < 1.0e-14) = 1.0e-14;
    FHLL = (SR .* FL - SL .* FR + SL .* SR .* (UR - UL)) ./ denom;
    F = FHLL;
    F(:, SR <= 0.0) = FR(:, SR <= 0.0);
    F(:, SL >= 0.0) = FL(:, SL >= 0.0);
    return;
end

if ~strcmp(kind, 'hllc')
    error('numerical_flux:kind', 'unknown flux: %s', kind);
end

% HLLC：先计算所有分支，再用掩码选择
denom_star = rhoL .* (SL - uL) - rhoR .* (SR - uR);
denom_star(abs(denom_star) < 1.0e-14) = 1.0e-14;
SM = (pR - pL + rhoL .* uL .* (SL - uL) ...
      - rhoR .* uR .* (SR - uR)) ./ denom_star;

UstarL = star_state(UL, rhoL, uL, pL, SL, SM);
UstarR = star_state(UR, rhoR, uR, pR, SR, SM);
FstarL = FL + SL .* (UstarL - UL);
FstarR = FR + SR .* (UstarR - UR);

F = FstarR;
F(:, SM >= 0.0) = FstarL(:, SM >= 0.0);
F(:, SR <= 0.0) = FR(:, SR <= 0.0);
F(:, SL >= 0.0) = FL(:, SL >= 0.0);

% Riemann 扇退化时退回 LLF
bad = any(~isfinite(F), 1);
if any(bad)
    alpha = max(abs(uL) + aL, abs(uR) + aR);
    Fllf = 0.5 * (FL + FR) - 0.5 * alpha .* (UR - UL);
    F(:, bad) = Fllf(:, bad);
end
end

function Us = star_state(U, rho, u, p, S, SM)
    d1 = safe_den(S - SM);
    d2 = safe_den(S - u);
    rho_star = rho .* (S - u) ./ d1;
    e_star = U(3, :) ./ rho + (SM - u) .* (SM + p ./ (rho .* d2));
    Us = [rho_star; rho_star .* SM; rho_star .* e_star];
end

function d = safe_den(d)
    small = abs(d) < 1.0e-14;
    if any(small)
        s = ones(size(d));
        s(d < 0.0) = -1.0;
        d(small) = 1.0e-14 * s(small);
    end
end
