#include <bits/stdc++.h>
#include <mpi.h>
#include <unistd.h>
int main(int argc, char *argv[])
{
  int rank, n_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  std::stringstream my_string;
  double vol1, vol2;
  vol1 = rank;
  vol2 = n_rank - rank;
  my_string << vol1 << "\n"
            << vol2;
  int msgLen[n_rank];
  // 做一次通信 告知大家的信号长度
  int my_str_size = my_string.str().size();
  //  信息的集中
  MPI_Allgather(&my_str_size, 1, MPI_INT, msgLen, 1, MPI_INT, MPI_COMM_WORLD);
  if (rank == 0)
  {
    for (int i = 0; i < n_rank; ++i)
    {
      std::cout << msgLen[i] << " ";
    }
    std::cout << "\n";
  }
  // 字符串数组
  std::stringstream all_string[n_rank];
  char *p_all_string[n_rank];
  MPI_Request request[2 * n_rank];
  MPI_Status status[2 * n_rank];
  int tag = rand() % n_rank;
  int n_request = 0;
  int n_status = 0;
  const char *my_str_c = my_string.str().c_str();
  for (int i = 0; i < n_rank; ++i)
  {
    MPI_Isend(my_str_c, my_str_size + 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &request[n_request++]);
    MPI_Recv(p_all_string[i], msgLen[i] + 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status[n_status++]);
    // 手动相互通信以获取数据
  }
  // 同步数据
  MPI_Waitall(n_request, request, status);

  MPI_Finalize();
  return 0;
}