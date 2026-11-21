function paths = discover_config_paths(config_args)
%DISCOVER_CONFIG_PATHS 返回待运行的 JSON 配置文件列表
%
%   config_args 可为文件、目录组成的元胞数组；为空时使用 configs/*.json。
%   对应 Python 版本 euler_unified/config.py 中的 discover_config_paths。
if nargin < 1
    config_args = {};
end
if ischar(config_args)
    config_args = {config_args};
end

if isempty(config_args)
    default_dir = fullfile(euler_matlab_root(), 'configs');
    if exist(default_dir, 'dir') ~= 7
        error('discover_config_paths:dir', ...
              'default config directory not found: %s', default_dir);
    end
    paths = list_json(default_dir);
    if isempty(paths)
        error('discover_config_paths:empty', ...
              'no JSON files found in %s', default_dir);
    end
    return;
end

paths = {};
for k = 1:numel(config_args)
    item = config_args{k};
    if exist(item, 'dir') == 7
        found = list_json(item);
        if isempty(found)
            error('discover_config_paths:empty', ...
                  'no JSON files found in %s', item);
        end
        paths = [paths, found]; %#ok<AGROW>
    elseif exist(item, 'file') == 2
        paths{end + 1} = item; %#ok<AGROW>
    else
        error('discover_config_paths:missing', ...
              'configuration not found: %s', item);
    end
end

% 按绝对路径去重并保持原有顺序
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
    paths = cell(1, numel(listing));
    for k = 1:numel(listing)
        paths{k} = fullfile(d, listing(k).name);
    end
end
