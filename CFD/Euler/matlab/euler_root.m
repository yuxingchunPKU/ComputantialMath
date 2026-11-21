function root = euler_root()
%EULER_ROOT CFD/Euler 目录（包含 Python 版 configs、figures 等）
    here = fileparts(mfilename('fullpath'));      % CFD/Euler/matlab
    root = fileparts(here);                       % CFD/Euler
end
