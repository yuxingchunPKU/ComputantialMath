function s = fmt_num(v)
%FMT_NUM 报告中的数值格式（对应 Python 版 Euler run.py 的 _fmt）
    if isempty(v)
        s = '—';
        return;
    end
    v = double(v);
    if v == 0
        s = '0';
    elseif abs(v) >= 1.0e4 || abs(v) < 1.0e-3
        s = sprintf('%.3e', v);
    else
        s = sprintf('%.6g', v);
    end
end
