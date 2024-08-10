#include <limits>
#include <set>
#include <string>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>
#include <unistd.h>

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <AFEPack/HGeometry.h>
#include <AFEPack/DGFEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/mpi/MPI_HGeometry.h>
#include <AFEPack/mpi/MPI_LoadBalance.h>
#include <AFEPack/mpi/MPI_Controller.h>
#include <AFEPack/mpi/MPI_FaceData.h>
#include <AFEPack/mpi/MPI_SyncProp.h>
#include <AFEPack/mpi/MPI_SyncDOF.h>
#include <AFEPack/mpi/MPI_MemoryReclaimer.h>
#include <AFEPack/mpi/MPI_PeriodHandler.h>
#define DIM 2
int main(int argc, char *argv[])
{
  typedef AFEPack::MPI::HGeometryForest<DIM> h_tree_t;
  typedef AFEPack::MPI::BirdView<h_tree_t> ir_mesh_t;
  int rank, n_rank;
  MPI_Init(&argc, &argv);
  int checkpoint = 1;
  while (0 == checkpoint)
  {
    sleep(5);
  }
  Migration::initialize(MPI_COMM_WORLD);
  h_tree_t h_tree;
  h_tree.set_communicator(MPI_COMM_WORLD);
  ir_mesh_t ir_mesh;
  std::string src_dir = "data/T2";
  std::cout << src_dir << "\n";
  /// 三角形
  TemplateGeometry<DIM> triangle_template_geometry; /* 模板单元信息 */
  CoordTransform<DIM> triangle_coord_transform;     /* 坐标变换 */
  UnitOutNormal<DIM> triangle_unit_out_normal;      /* 单位外法向量 */

  TemplateDOF<DIM> triangle_0_template_dof;                  /* 自由度 */
  BasisFunctionAdmin<double, DIM> triangle_0_basis_function; /* 基函数 */

  TemplateDOF<DIM> triangle_1_template_dof;                  /* 自由度 */
  BasisFunctionAdmin<double, DIM> triangle_1_basis_function; /* 基函数 */

  TemplateDOF<DIM> triangle_1_D_template_dof;                  /* 自由度 */
  BasisFunctionAdmin<double, DIM> triangle_1_D_basis_function; /* 基函数 */

  std::vector<TemplateElement<double, DIM>> template_element;         // 单元模板
  std::vector<TemplateDGElement<DIM - 1, DIM>> edge_template_element; // 边的模板

  triangle_template_geometry.readData("triangle.tmp_geo");
  triangle_coord_transform.readData("triangle.crd_trs");
  triangle_unit_out_normal.readData("triangle.out_nrm");

  triangle_0_template_dof.reinit(triangle_template_geometry);
  triangle_0_template_dof.readData("triangle.0.tmp_dof");
  triangle_0_basis_function.reinit(triangle_0_template_dof);
  triangle_0_basis_function.readData("triangle.0.bas_fun");

  triangle_1_D_template_dof.reinit(triangle_template_geometry);
  triangle_1_D_template_dof.readData("triangle.v1.D.tmp_dof");
  triangle_1_D_basis_function.reinit(triangle_1_D_template_dof);
  triangle_1_D_basis_function.readData("triangle.v1.D.bas_fun");

  triangle_1_template_dof.reinit(triangle_template_geometry);
  triangle_1_template_dof.readData("triangle.1.tmp_dof");
  triangle_1_basis_function.reinit(triangle_1_template_dof);
  triangle_1_basis_function.readData("triangle.1.bas_fun");

  /// 表面线段
  TemplateGeometry<DIM - 1> interval_template_geometry;
  CoordTransform<DIM - 1, DIM> interval_to2d_coord_transform;

  /// 相界面线段
  TemplateGeometry<DIM - 1> template_front_interval;
  CoordTransform<DIM - 1, DIM> front_interval_coord_transform;

  template_element.resize(3);
  template_element[0].reinit(triangle_template_geometry,
                             triangle_0_template_dof,
                             triangle_coord_transform,
                             triangle_0_basis_function,
                             triangle_unit_out_normal);
  template_element[1].reinit(triangle_template_geometry,
                             triangle_1_D_template_dof,
                             triangle_coord_transform,
                             triangle_1_D_basis_function,
                             triangle_unit_out_normal);
  template_element[2].reinit(triangle_template_geometry,
                             triangle_1_template_dof,
                             triangle_coord_transform,
                             triangle_1_basis_function,
                             triangle_unit_out_normal);

  interval_template_geometry.readData("interval.tmp_geo");
  interval_to2d_coord_transform.readData("interval.to2d.crd_trs");

  edge_template_element.resize(1);
  edge_template_element[0].reinit(interval_template_geometry,
                                  interval_to2d_coord_transform);
  // 读取网格文件
  ir_mesh.clear();
  AFEPack::MPI::load_mesh(src_dir, h_tree, ir_mesh);
  // ir_mesh.semiregularize();
  // ir_mesh.regularize(false);
  // Migration::clear_data_buffer(h_tree);
  // if (ir_mesh != NULL)
  // {
  //   ir_mesh->clear();
  // }
  h_tree.clear();
  MPI_Finalize();
  return 0;
}
