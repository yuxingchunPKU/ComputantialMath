#include <bits/stdc++.h>
#include <mpi.h>
// 手动实现集合通信
// 先传一般数据，再传特殊的数据
// 传输不定长度的消息
// 传递复杂类型的数据
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
  int n_status = 0;
  std::set<int> my_msg;
  for (int i = 0; i < rank + 1; ++i)
  {
    my_msg.insert(i);
  }
  std::cout << "size :" << my_msg.size() << "\n";
  std::vector<std::set<int>> comm_msg(n_rank);
  int recv_msg_vec[n_rank];
  for (int i = 0; i < n_rank; ++i)
  {
    std::vector<int> my_msg_vec = {my_msg.begin(), my_msg.end()};
    MPI_Isend(&my_msg_vec[0], my_msg_vec.size(), MPI_INT, i, tag, MPI_COMM_WORLD, &request[n_request++]);
    MPI_Recv(recv_msg_vec, i + 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status[n_status++]);
    // 这样做不行 因为这些函数还没收信息 可能下面已经执行了
    // Irecv 不接受也会返回 这会引起冲突
    comm_msg[i].clear();
    for (int j = 0; j < i + 1; ++j)
    {
      comm_msg[i].insert(recv_msg_vec[j]);
    }
  }
  MPI_Waitall(n_request, request, status);
  if (rank == 0)
  {
    for (auto ele_s : comm_msg)
    {
      for (auto ele : ele_s)
      {
        std::cout << ele << " ";
      }
      std::cout << "\n";
    }
  }
  MPI_Finalize();
  return 0;
}
