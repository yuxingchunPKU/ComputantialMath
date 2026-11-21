function out = deep_merge(base, override)
%DEEP_MERGE 递归合并两个结构体，override 中的字段优先
%
%   对应 Python 版本中的 deep_merge / _deep_update。
%   数组、空值等非结构体字段直接覆盖。
out = base;
if isempty(override) || ~isstruct(override)
    return;
end
fns = fieldnames(override);
for k = 1:numel(fns)
    f = fns{k};
    if isfield(out, f) && isstruct(out.(f)) && isstruct(override.(f))
        out.(f) = deep_merge(out.(f), override.(f));
    else
        out.(f) = override.(f);
    end
end
end
