function c = hex_to_rgb(h)
%HEX_TO_RGB 把 '#rrggbb' 形式的颜色转换为 MATLAB 的 [r g b]
    h = strrep(char(h), '#', '');
    c = [hex2dec(h(1:2)), hex2dec(h(3:4)), hex2dec(h(5:6))] / 255;
end
