%已知 rho0 u0 p0 gamma 和 rho1 计算波后状态

% 初始值
rho0 =1.0;
u0 = 0.0;
p0 = 1e5;
rho1 = 1.3333;
gamma = 1.4;

% 导出值
a =sqrt(gamma*p0/rho0);
Ma = u0/a;
ratio_rho = rho1/rho0;

% 计算值
Ms = Ma +sqrt(2*ratio_rho/((gamma+1)-ratio_rho*(gamma-1)));
s = a*Ms;
ratio_p = (2*gamma*(Ma-Ms)^2-(gamma-1))/(gamma+1);
p1= p0*ratio_p;
u1 = u0/ratio_rho + s*(1-1/ratio_rho);

% 输出波后的状态
fprintf('after shock density is %.15f\n',rho1);
fprintf('after shock velocity is %.15f\n',u1);
fprintf('after shock pressure is %.15f\n',p1);
fprintf('shock speed is %.15f\n',s);