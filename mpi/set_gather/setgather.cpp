#include <bits/stdc++.h>
#include <mpi.h>
// 手动实现集合通信
// 先传一般数据，再传特殊的数据
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int myid, rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  // 使用mpi的标签
  MPI_Request request[2 * n_rank];
  MPI_Status status[2 * n_rank];
  // 相互传信息
  std::vector<int> common_vec(n_rank);
  //  手动实现相互传信息
  // 标签可以随便设置
  int tag = rand() % n_rank;
  int n_request = 0;
  for (int i = 0; i < n_rank; ++i)
  {
    MPI_Isend(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &request[n_request++]);
    MPI_Irecv(&common_vec[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &request[n_request++]);
  }
  MPI_Waitall(n_request, request, status);
  for (auto ele : common_vec)
  {
    std::cout << ele << " ";
  }
  std::cout << "\n";

  // 复杂类型的通信
  // 每一个进程中都有一个集合 集合中的长度不一样
  std::set<int> my_msg;
  for (int i = 0; i < rank; ++i)
  {
    my_msg.insert(i);
  }
  std::vector<std::set<int>> comm_msg;
  // 接受数据
  MPI_Finalize();
  return 0;
}
