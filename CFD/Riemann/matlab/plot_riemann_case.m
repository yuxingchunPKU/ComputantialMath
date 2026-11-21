function figures = plot_riemann_case(cs, x, rho, u, p, waves, out_dir, plot_cfg)
%PLOT_RIEMANN_CASE 绘制 rho/u/p 精确解与三联图，并标出各道波的位置
%
%   对应 Python 版本 code/run_cases.py 中的 plot_case。
if nargin < 8 || isempty(plot_cfg)
    plot_cfg = struct();
end
dpi = 200;
if isfield(plot_cfg, 'dpi') && ~isempty(plot_cfg.dpi)
    dpi = plot_cfg.dpi;
end
figsize = [4.2, 3.0];
if isfield(plot_cfg, 'figsize') && ~isempty(plot_cfg.figsize)
    figsize = double(plot_cfg.figsize(:).');
end
line_color = '#1f77b4';
if isfield(plot_cfg, 'line_color') && ~isempty(plot_cfg.line_color)
    line_color = plot_cfg.line_color;
end
color = hex_to_rgb(line_color);

if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

d = cs.domain;
x_L = d.x_L;
x_R = d.x_R;
positions = wave_positions(cs, waves);

quantities = {rho, u, p};
ylabels = {'\rho', 'u', 'p'};
titles = {'density', 'velocity', 'pressure'};
keys = {'rho', 'u', 'p'};

figures = struct();
for q = 1:3
    fig = figure('visible', 'off', 'color', 'w', ...
                 'paperunits', 'inches', 'papersize', figsize, ...
                 'paperposition', [0, 0, figsize]);
    ax = axes('parent', fig);
    plot_series(ax, x, quantities{q}, positions, x_L, x_R, color, ...
                ylabels{q}, sprintf('%s: %s', cs.name, titles{q}), true);
    path = fullfile(out_dir, [keys{q} '.png']);
    print(fig, path, '-dpng', sprintf('-r%d', dpi));
    close(fig);
    figures.(keys{q}) = path;
end

% 三联图（显式坐标位置，避免标题与上一幅图的 x 轴标签重叠）
ofz = [10.0, 7.6];
fig = figure('visible', 'off', 'color', 'w', ...
             'paperunits', 'inches', 'papersize', ofz, ...
             'paperposition', [0, 0, ofz]);
bottoms = [0.705, 0.415, 0.105];
for q = 1:3
    ax = axes('parent', fig, 'position', [0.13, bottoms(q), 0.80, 0.19]);
    plot_series(ax, x, quantities{q}, positions, x_L, x_R, color, ...
                ylabels{q}, titles{q}, false);
    if q < 3
        xlabel(ax, '');
    end
end
ax_title = axes('parent', fig, 'position', [0, 0.94, 1, 0.05], ...
                'visible', 'off');
text(0.5, 0.5, sprintf('%s - exact solution at t = %g', ...
                       cs.name, d.t_end), ...
     'parent', ax_title, 'horizontalalignment', 'center', ...
     'verticalalignment', 'middle', 'interpreter', 'none', ...
     'fontsize', 11, 'fontweight', 'bold');
path = fullfile(out_dir, 'overview.png');
print(fig, path, '-dpng', sprintf('-r%d', dpi));
close(fig);
figures.overview = path;
end

% ----------------------------------------------------------------------
function positions = wave_positions(cs, waves)
d = cs.domain;
positions = [];
for k = 1:numel(waves)
    speeds = waves{k}.speeds;
    fns = fieldnames(speeds);
    for j = 1:numel(fns)
        xp = d.x_c + speeds.(fns{j}) * d.t_end;
        if xp > d.x_L && xp < d.x_R
            positions(end + 1) = xp; %#ok<AGROW>
        end
    end
end
end

function plot_series(ax, x, values, positions, x_L, x_R, color, ...
                     ylabel_text, title_text, add_legend)
x = x(:);
values = values(:);
hold(ax, 'on');
h = plot(ax, x, values, 'color', color, 'linewidth', 1.8);
yl = ylim(ax);
for k = 1:numel(positions)
    plot(ax, [positions(k), positions(k)], yl, 'color', [0.55, 0.55, 0.55], ...
         'linestyle', '--', 'linewidth', 0.8);
end
ylim(ax, yl);
xlim(ax, [x_L, x_R]);
xlabel(ax, 'x');
ylabel(ax, ylabel_text);
title(ax, title_text, 'interpreter', 'none');
grid(ax, 'on');
set(ax, 'gridalpha', 0.3);
if add_legend
    legend(h, {'Exact'}, 'location', 'northeast');
end
end
