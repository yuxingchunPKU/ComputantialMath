function W = make_fluid(spec, reactive)
%MAKE_FLUID 由配置结构体构造 Fluid / FluidReact 对象
%
%   spec 至少包含 rho、u、p；gamma 缺省 1.4。
%   非反应流使用 p_inf（缺省 0），反应流使用 q（缺省 0）。
if reactive
    gamma = 1.4;
    if isfield(spec, 'gamma') && ~isempty(spec.gamma)
        gamma = spec.gamma;
    end
    q = 0.0;
    if isfield(spec, 'q') && ~isempty(spec.q)
        q = spec.q;
    end
    W = FluidReact(spec.rho, spec.u, spec.p, gamma, q);
else
    gamma = 1.4;
    if isfield(spec, 'gamma') && ~isempty(spec.gamma)
        gamma = spec.gamma;
    end
    p_inf = 0.0;
    if isfield(spec, 'p_inf') && ~isempty(spec.p_inf)
        p_inf = spec.p_inf;
    end
    W = Fluid(spec.rho, spec.u, spec.p, gamma, p_inf);
end
end
