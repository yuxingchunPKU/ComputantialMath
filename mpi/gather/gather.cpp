#include <bits/stdc++.h>
#include <mpi.h>
#include <unistd.h>
int main(int argc, char *argv[])
{
  int myid, rank, n_rank;
  int i = 1;
  // while (0 == i)
  // {
  //   sleep(5);
  // }
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  // Scatter
  std::vector<int> scatter_vec(n_rank * 2);
  std::iota(scatter_vec.begin(), scatter_vec.end(), 0);
  int scatter_revc[2];
  MPI_Scatter(&scatter_vec[0], 2, MPI_INT, &scatter_revc, 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "rank " << rank << " , scatter_revc:" << scatter_revc[0] << " " << scatter_revc[1] << std::endl;
  // Scatter_v 测试失败了
  //  int scatterv_revc[2];
  //  int scatterv_idx[n_rank] = {0, 2, 4, 6};
  //  int revc_ct = rank % 2;
  // MPI_Scatterv(&scatter_vec[0], &revc_ct, scatterv_idx, MPI_INT, &scatterv_revc, revc_ct, MPI_INT, 0, MPI_COMM_WORLD);
  // gather
  // MPI_Gather();
  std::vector<int> allgaher_data;
  allgaher_data.resize(n_rank);
  MPI_Allgather(&rank, 1, MPI_INT, &allgaher_data[0], 1, MPI_INT, MPI_COMM_WORLD);
  if (rank == 0)
  {
    std::cout << "MPI_Allgather data\n";
    for (auto i : allgaher_data)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  // 不同长度的数据参与集中

  //  Allreduce
  int sum_rank;
  MPI_Allreduce(&rank, &sum_rank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "rank " << rank << " send " << rank << " ,sum rank " << sum_rank << std::endl;
  // 先考虑两个进程的
  std::vector<int> gv_send_vec;
  gv_send_vec.resize(rank);
  std::iota(gv_send_vec.begin(), gv_send_vec.end(), 0);
  int gv_recv_vec[3];
  int gv_displs[2] = {0, 1};
  int gv_recv_count[2] = {1, 2};
  // MPI_Allgatherv(&gv_send_vec[0], rank + 1, MPI_INT, gv_recv_vec, gv_recv_count, gv_displs, MPI_INT, MPI_COMM_WORLD);
  if (rank == 0)
  {
    std::cout << "MPI_Allgatherv data\n";
    for (auto i : gv_recv_vec)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  std::vector<int> rank_vec(n_rank);
  MPI_Allgather(&rank, 1, MPI_INT, &rank_vec[0], 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "rank " << rank << " send " << rank << " ,rank_vec[" << rank << "] " << rank_vec[rank] << std::endl;
  std::vector<int> src_rank(2 * n_rank, 0);
  MPI_Finalize();
  return 0;
}
