function paths = plot_euler_case(cfg, res)
%PLOT_EULER_CASE 为单个算例生成 rho/u/p 图像与三联图
%
%   对应 Python 版本 euler_unified/plotting.py 的 plot_case。
%   图像保存在 <matlab>/figures/<case>/ 下。
out = euler_output_dir(cfg);
if exist(out, 'dir') ~= 7
    mkdir(out);
end

surface = hex_to_rgb('#fcfcfb');
text2 = hex_to_rgb('#52514e');
gridc = hex_to_rgb('#e4e3df');
cnum = hex_to_rgb('#2a78d6');
cex = hex_to_rgb('#eb6834');

dpi = cfg.output.dpi;
fsz = cfg.output.figsize;
keys = {'rho', 'u', 'p'};
ylabels = {'\rho', 'u', 'p'};
titles = {'Density', 'Velocity', 'Pressure'};

paths = struct();
for q = 1:3
    fig = figure('visible', 'off', 'color', surface, ...
                 'paperunits', 'inches', 'papersize', fsz, ...
                 'paperposition', [0, 0, fsz]);
    ax = axes('parent', fig);
    plot_quantity(ax, res, keys{q}, ylabels{q}, ...
                  sprintf('%s: %s', cfg.name, titles{q}), ...
                  surface, text2, gridc, cnum, cex);
    p = fullfile(out, [keys{q} '.png']);
    print(fig, p, '-dpng', sprintf('-r%d', dpi));
    close(fig);
    paths.(keys{q}) = p;
end

% 三联图
ofz = [11.5, 3.1];
fig = figure('visible', 'off', 'color', surface, ...
             'paperunits', 'inches', 'papersize', ofz, ...
             'paperposition', [0, 0, ofz]);
handles = [];
labels = {};
for q = 1:3
    ax = subplot(1, 3, q);
    h = plot_quantity(ax, res, keys{q}, ylabels{q}, titles{q}, ...
                      surface, text2, gridc, cnum, cex);
    if q == 1
        handles = h;
        labels = {'Numerical', 'Exact'};
    end
end
if numel(handles) >= 2
    try
        legend(handles(1:2), labels, 'location', 'southoutside', ...
               'orientation', 'horizontal');
    catch
        legend(handles(1:2), labels, 'location', 'northeast');
    end
end
p = fullfile(out, 'overview.png');
print(fig, p, '-dpng', sprintf('-r%d', dpi));
close(fig);
paths.overview = p;
end

function h = plot_quantity(ax, res, key, ylabel_text, title_text, ...
                           surface, text2, gridc, cnum, cex)
x = res.x(:);
y = res.(key);
y = y(:);
exact = res.([key '_exact']);

hold(ax, 'on');
h1 = plot(ax, x, y, 'color', cnum, 'linewidth', 2.0);
h = h1;
if ~isempty(exact)
    exact = exact(:);
    h2 = plot(ax, x, exact, 'color', cex, 'linewidth', 1.8, ...
              'linestyle', '--');
    h = [h1, h2];
end

% 近常量（如光滑输运算例中的 u、p）的自适应纵轴会压成 1e-15 的窗口。
values = y;
if ~isempty(exact)
    values = [values; exact];
end
lo = min(values);
hi = max(values);
scale = max([abs(lo), abs(hi), 1.0]);
if hi - lo < 1.0e-8 * scale
    center = 0.5 * (lo + hi);
    pad = max(0.1 * abs(center), 1.0e-3);
    ylim(ax, [center - pad, center + pad]);
end

set(ax, 'color', surface, 'xcolor', text2, 'ycolor', text2, ...
    'fontsize', 9, 'box', 'off');
try
    set(ax, 'gridcolor', gridc, 'gridalpha', 0.8);
catch
end
grid(ax, 'on');
set(ax, 'layer', 'bottom');
xlabel(ax, 'x', 'color', text2, 'fontsize', 9);
ylabel(ax, ylabel_text, 'color', text2, 'fontsize', 9);
title(ax, title_text, 'color', hex_to_rgb('#0b0b0b'), 'fontsize', 10);
if numel(h) >= 2
    legend(ax, h, {'Numerical', 'Exact'}, 'location', 'northeast', ...
           'box', 'off');
end
end
