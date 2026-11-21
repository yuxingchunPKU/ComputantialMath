function rel = relative_to(target, base)
%RELATIVE_TO 计算 target 相对于目录 base 的相对路径（使用 '/' 分隔）
    t = split_path(absolute_path(target));
    b = split_path(absolute_path(base));

    n = 0;
    while n < numel(t) && n < numel(b) && strcmp(t{n + 1}, b{n + 1})
        n = n + 1;
    end

    parts = [repmat({'..'}, 1, numel(b) - n), t(n + 1:end)];
    if isempty(parts)
        rel = '.';
    else
        rel = strjoin(parts, '/');
    end
end

function parts = split_path(p)
    p = strrep(char(p), '\', '/');
    parts = regexp(p, '/', 'split');
    parts = parts(~cellfun(@isempty, parts));
end
