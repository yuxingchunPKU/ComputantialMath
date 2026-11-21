function p = add_common_path()
%ADD_COMMON_PATH 把 CFD/matlab_common 加入搜索路径
%
%   matlab_common 与本函数所在目录同级。
    here = fileparts(mfilename('fullpath'));
    p = fullfile(here);
    if exist(p, 'dir')
        addpath(p);
    end
end
