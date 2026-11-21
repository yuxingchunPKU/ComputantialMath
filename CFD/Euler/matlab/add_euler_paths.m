function root = add_euler_paths()
%ADD_EULER_PATHS 把 Euler/matlab、matlab_common 与 Riemann/matlab 加入搜索路径
    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(here);

    cfd_root = fileparts(root);
    common = fullfile(cfd_root, 'matlab_common');
    if exist(common, 'dir')
        addpath(common);
    end
    riemann = fullfile(cfd_root, 'Riemann', 'matlab');
    if exist(riemann, 'dir')
        addpath(riemann);
    end
end
