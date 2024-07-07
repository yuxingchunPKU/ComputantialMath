#include <bits/stdc++.h>
#include <mpi.h>
struct patchinfo
{
  double ele_vol;
  double U[4];
};

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int myid, rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  // 把结构包装成mpi类型 然后发送出去
  // 数量 有两块 块的数目 每块的位移 每块的类型
  std::vector<int> strublocklens = {1, 4};
  MPI_Aint struindices[2]; // 位移 必须使用这种类型
  // 获得地址 并计算位置
  patchinfo patch;
  MPI_Get_address(&patch.ele_vol, &(struindices[0]));
  MPI_Get_address(&patch.U, &(struindices[1]));
  struindices[1] = struindices[1] - struindices[0];
  struindices[0] = 0;
  std::vector<MPI_Datatype> stru_type = {MPI_DOUBLE, MPI_DOUBLE};
  MPI_Datatype MPI_patch_stuct;
  MPI_Type_create_struct(2, &strublocklens[0], &struindices[0], &stru_type[0], &MPI_patch_stuct);
  // 先提交才能使用
  // std::cout << "struindices[1] " << struindices[1] << std::endl;
  // 建立对象
  MPI_Type_commit(&MPI_patch_stuct);
  // send path
  patchinfo p;
  p.ele_vol = double(rank) / n_rank;
  for (int i = 0; i < 4; ++i)
  {
    p.U[i] = rank + 1;
  }
  // 集合通信
  std::vector<patchinfo> comm_patch(n_rank);
  MPI_Allgather(&p, 1, MPI_patch_stuct, &comm_patch[0], 1, MPI_patch_stuct, MPI_COMM_WORLD);
  if (rank == 0)
  {
    for (auto patch_ele : comm_patch)
    {
      std::cout << "rank " << rank << " " << patch_ele.ele_vol << " " << patch_ele.U[0] << "\n";
    }
  }
  srand(rank + 1);
  std::cout << "rank " << rank << " random " << random() % n_rank << "\n";
  MPI_Finalize();
  return 0;
}
