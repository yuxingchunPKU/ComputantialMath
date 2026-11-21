function metrics = euler_error_metrics(res)
%EULER_ERROR_METRICS 相对精确解的 L1 / Linf 误差
%
%   对应 Python 版本 euler_unified/run.py 中的 _error_metrics。
%   返回 struct，每个量对应 [L1, Linf]；无精确解时为 [NaN, NaN]。
metrics = struct();
keys = {'rho', 'u', 'p'};
for k = 1:numel(keys)
    key = keys{k};
    exact = res.([key '_exact']);
    if isempty(exact)
        metrics.(key) = [NaN, NaN];
        continue;
    end
    diff = res.(key)(:) - exact(:);
    metrics.(key) = [mean(abs(diff)), max(abs(diff))];
end
end
