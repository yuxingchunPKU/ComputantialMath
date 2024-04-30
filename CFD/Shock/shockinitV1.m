%已知 rho0 u0 p0 gamma 和 rho1 计算波后状态
rho0 =1.0;
u0 = 0.0;
p0 = 1e5;
p1 = 1.5e5;
gamma = 1.4;

% 导出值
a =sqrt(gamma*p0/rho0);
Ma = u0/a;
ratio_p = p1/p0;

% 计算值
Ms = Ma +sqrt((gamma-1)/(2*gamma)+ratio_p*((gamma+1)/(2*gamma)) );
s = a*Ms;
ratio_g = (gamma-1)/(gamma+1);
% 下面两个计算的方式等价
rho1 =rho0*(ratio_p+ratio_g)/(ratio_g*ratio_p+1);
% rho1 =rho0*(gamma+1)*(Ma-Ms).^2/((gamma-1)*(Ms-Ma).^2+2);
ratio_rho = rho1/rho0;
u1 = u0/ratio_rho + s*(1-1/ratio_rho);

% 输出波后的状态
fprintf('after shock density is %.15f\n',rho1);
fprintf('after shock velocity is %.15f\n',u1);
fprintf('after shock pressure is %.15f\n',p1);
fprintf('shock speed is %.15f\n',s);


