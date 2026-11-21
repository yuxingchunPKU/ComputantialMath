function out = reflect_state(U)
%REFLECT_STATE 固壁边界上的守恒量镜像（速度反向）
out = U;
out(2, :) = -out(2, :);
end
