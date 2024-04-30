Mach= 1.22;
% group data 1
% rho_air = 1.2;
% p_air = 1.01e5;
% gamma_air = 1.4;
% group data 2
% rho_air = 1.29;
% p_air = 101325;
% gamma_air = 1.4;
% % group data 3
% rho_air = 0.95;
% p_air = 0.8e5;
% gamma_air = 1.4;
% group data 4
% rho_air = 1.225;
% p_air = 1.013e5;
% gamma_air = 1.4;
% group data 5

% 波前的状态
rho_air = 1.0;
p_air = 1;
gamma_air = 1.4;

a_air=sqrt(gamma_air*p_air/rho_air);

p_ratio = (2*gamma_air*(Mach).^2-(gamma_air-1))/(gamma_air+1);
p_shock = p_air * p_ratio;

rho_ratio = ((gamma_air+1)*(Mach).^2)/((gamma_air-1)*(Mach).^2+2);
rho_shock = rho_air*rho_ratio;

% Ma*a 是激波的速度
% 根据RH条件确定波后流体的速度
u_flow = Mach*a_air*(rho_shock-rho_air)/rho_shock;
s_shock = Mach*a_air;

% 输出波后的状态
fprintf('after shock density is %.15f\n',rho_shock);
fprintf('after shock velocity is %.15f\n',u_flow);
fprintf('after shock pressure is %.15f\n',p_shock);
fprintf('shock speed is %.15f\n',s_shock);



