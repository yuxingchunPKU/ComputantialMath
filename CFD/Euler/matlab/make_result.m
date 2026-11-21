function res = make_result(cfg, x, rho, u, p, dt, steps)
%MAKE_RESULT 组装求解结果结构体，并尝试计算精确解
    res = struct();
    res.x = x;
    res.rho = rho;
    res.u = u;
    res.p = p;
    res.rho_exact = [];
    res.u_exact = [];
    res.p_exact = [];
    res.exact_error = '';
    res.dt = dt;
    res.steps = steps;

    try
        [re, ue, pe] = exact_fields(cfg, x, cfg.time.t_end);
        res.rho_exact = re;
        res.u_exact = ue;
        res.p_exact = pe;
    catch err
        res.exact_error = err.message;
    end
end
