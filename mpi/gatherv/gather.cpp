#include <bits/stdc++.h>
#include <mpi.h>
#include <unistd.h>
using namespace std;
#include <stdio.h>
#include <mpi.h>

/*
 * 收集所有任务的数据并将组合数据传送到所有任务
 * int MPI_Allgatherv（const void * sendbuf，int sendcount，MPI_Datatype sendtype，
 *      void * recvbuf，const int * recvcounts，const int * displs，
 *      MPI_Datatype recvtype，MPI_Comm comm）
 *
 * sendbuf：要发送内容的起始地址
 * sendcount：要发送的数量
 * sendtype：要发送数据的类型
 * recvbuf：接收数据要存放的单元的地址
 * recvcounts：这是一个整数数组，包含从每个进程要接收的数据量，比如{0, 1} 从0号进程接收0个，从1号进程接收1个
 * displs：这是一个整数数组，包含存放从每个进程接收的数据相对于recvbuf的偏移地址
 * recvtype：要接收的数据类型
 * comm：通信集合
 */

int main(int argc, char *argv[])
{
  int i;
  int rank, nproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> isend(nproc);
  int iscnt;
  std::vector<int> irecv(nproc * (nproc + 1) / 2);
  // 从每个进程接收数据的个数
  // int ircnt[3] = {1, 2, 3};
  std::vector<int> ircnt(nproc);
  std::iota(ircnt.begin(), ircnt.end(), 1);
  // 存放数据的偏移地址 相对于数值开头的位置
  std::vector<int> idisp(nproc);
  for (int i = 0; i < nproc; ++i)
  {
    idisp[i] = i * (i + 1) / 2;
  }
  if (rank == 0)
    std::cout << "This program must runprocesses: np = " << nproc << std::endl;

  std::iota(isend.begin(), isend.end(), rank);
  // 0
  // 1 2
  // 2 3 4
  // o号进程发送1个，1号发送两个，2号发送3个，总共6个
  iscnt = rank + 1;
  MPI_Allgatherv(&isend[0], iscnt, MPI_INT, &irecv[0], &ircnt[0], &idisp[0], MPI_INT, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::cout << "MPI_Allgatherv data\n";
    for (auto i : irecv)
    {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}
