function root = add_riemann_paths()
%ADD_RIEMANN_PATHS 把 Riemann/matlab 与 CFD/matlab_common 加入搜索路径
    here = fileparts(mfilename('fullpath'));
    root = here;
    addpath(here);

    cfd_root = fileparts(fileparts(here));
    common = fullfile(cfd_root, 'matlab_common');
    if exist(common, 'dir')
        addpath(common);
    end
end
