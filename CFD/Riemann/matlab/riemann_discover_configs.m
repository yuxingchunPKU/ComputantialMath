function paths = riemann_discover_configs(config_args)
%RIEMANN_DISCOVER_CONFIGS 返回待读取的算例文件（按 inert → detonation → deflagration 排序）
if nargin < 1
    config_args = {};
end
if ischar(config_args)
    config_args = {config_args};
end

if isempty(config_args)
    default_dir = fullfile(riemann_matlab_root(), 'cases');
    if exist(default_dir, 'dir') ~= 7
        error('riemann_discover_configs:dir', ...
              '默认算例目录不存在: %s', default_dir);
    end
    paths = list_json(default_dir);
    if isempty(paths)
        error('riemann_discover_configs:empty', ...
              '在 %s 中未找到 JSON 配置文件', default_dir);
    end
    return;
end

paths = {};
for k = 1:numel(config_args)
    item = config_args{k};
    if exist(item, 'dir') == 7
        found = list_json(item);
        if isempty(found)
            error('riemann_discover_configs:empty', ...
                  '目录 %s 中没有 JSON 配置文件', item);
        end
        paths = [paths, found]; %#ok<AGROW>
    elseif exist(item, 'file') == 2
        paths{end + 1} = item; %#ok<AGROW>
    else
        error('riemann_discover_configs:missing', '配置文件不存在: %s', item);
    end
end

seen = {};
unique_paths = {};
for k = 1:numel(paths)
    key = absolute_path(paths{k});
    if ~any(strcmp(seen, key))
        seen{end + 1} = key; %#ok<AGROW>
        unique_paths{end + 1} = paths{k}; %#ok<AGROW>
    end
end
paths = unique_paths;
end

function paths = list_json(d)
    listing = dir(fullfile(d, '*.json'));
    order = {'inert', 'detonation', 'deflagration'};
    keys = cell(1, numel(listing));
    for k = 1:numel(listing)
        [~, stem, ~] = fileparts(listing(k).name);
        idx = find(strcmp(order, stem), 1);
        if isempty(idx)
            idx = numel(order) + 1;
        end
        keys{k} = sprintf('%02d_%s', idx, listing(k).name);
    end
    [~, perm] = sort(keys);
    listing = listing(perm);
    paths = cell(1, numel(listing));
    for k = 1:numel(listing)
        paths{k} = fullfile(d, listing(k).name);
    end
end
