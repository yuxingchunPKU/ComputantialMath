function opts = parse_args(args)
%PARSE_ARGS 解析 --key value / --key=value / --flag 形式的命令行参数
%
%   opts 结构体中，字段名中的 '-' 被替换为 '_'。
%   多值参数：config、case（收集后续全部非 -- 开头的参数）。
%   开关参数：list、quiet、no-report、no-plot。
    opts = struct();
    multi_keys = {'config', 'case'};
    flag_keys = {'list', 'quiet', 'no-report', 'no-plot'};

    i = 1;
    while i <= numel(args)
        a = args{i};
        if ~ischar(a)
            error('parse_args:type', '命令行参数必须是字符串');
        end
        if ~strncmp(a, '--', 2)
            i = i + 1;
            continue;
        end

        key = a(3:end);
        eq = strfind(key, '=');
        if ~isempty(eq)
            value = key(eq(1) + 1:end);
            key = key(1:eq(1) - 1);
            opts.(strrep(key, '-', '_')) = value;
            i = i + 1;
            continue;
        end

        name = strrep(key, '-', '_');
        if any(strcmp(key, flag_keys))
            opts.(name) = true;
            i = i + 1;
        elseif any(strcmp(key, multi_keys))
            values = {};
            j = i + 1;
            while j <= numel(args) && ischar(args{j}) && ~strncmp(args{j}, '--', 2)
                values{end + 1} = args{j}; %#ok<AGROW>
                j = j + 1;
            end
            opts.(name) = values;
            i = j;
        else
            if i + 1 <= numel(args) && ischar(args{i + 1}) && ...
                    ~strncmp(args{i + 1}, '--', 2)
                opts.(name) = args{i + 1};
                i = i + 2;
            else
                opts.(name) = true;
                i = i + 1;
            end
        end
    end
end
