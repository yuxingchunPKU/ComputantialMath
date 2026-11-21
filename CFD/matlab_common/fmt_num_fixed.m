function s = fmt_num_fixed(v)
%FMT_NUM_FIXED 报告中的数值格式（对应 Python 版 Riemann run_cases.py 的 fnum）
    if isempty(v)
        s = '-';
        return;
    end
    v = double(v);
    if v == 0
        s = '0';
    elseif abs(v) >= 1.0e4 || abs(v) < 1.0e-3
        s = sprintf('%.6e', v);
    else
        s = sprintf('%.6f', v);
        s = regexprep(s, '0+$', '');
        s = regexprep(s, '\.$', '');
    end
end
