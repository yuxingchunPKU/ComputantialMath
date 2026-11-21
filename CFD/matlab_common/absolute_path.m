function p = absolute_path(p)
%ABSOLUTE_PATH 把相对路径转换为基于当前目录的绝对路径
    p = char(p);
    if isempty(p)
        p = pwd;
        return;
    end
    if p(1) == '/'
        return;
    end
    if numel(p) > 1 && p(2) == ':'
        return;
    end
    p = fullfile(pwd, p);
end
