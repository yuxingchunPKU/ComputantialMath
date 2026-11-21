function mesh = euler_mesh(cfg)
%EULER_MESH 由配置构造一维均匀网格
    m = cfg.mesh;
    mesh = struct();
    mesh.x_min = m.x_min;
    mesh.x_max = m.x_max;
    mesh.nx = m.nx;
    mesh.dx = (m.x_max - m.x_min) / m.nx;
    mesh.edges = linspace(m.x_min, m.x_max, m.nx + 1);
    mesh.centers = 0.5 * (mesh.edges(1:end - 1) + mesh.edges(2:end));
    mesh.boundary = m.boundary;
end
