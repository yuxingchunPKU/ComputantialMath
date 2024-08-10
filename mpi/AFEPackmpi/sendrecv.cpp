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

#include <AFEPack/mpi/MPI_HGeometry.h>
#include <AFEPack/mpi/MPI_ULoadBalance.h>
#include <AFEPack/mpi/MPI_Controller.h>
#include <AFEPack/mpi/MPI_FaceData.h>
#include <AFEPack/mpi/MPI_SyncProp.h>
#include <AFEPack/mpi/MPI_SyncDOF.h>
#include <AFEPack/mpi/MPI_MemoryReclaimer.h>
#include <AFEPack/mpi/MPI_PeriodHandler.h>
struct patch_value_entry
{
  int vol;                 /// 单元片所考虑的相所占据的面积
  std::vector<double> val; /// 单元片上守恒量的积分值
  patch_value_entry()
  {
    vol = 0;
    val.resize(4);
    std::fill(val.begin(), val.end(), 0.0);
  }
  patch_value_entry &operator=(const patch_value_entry &pv)
  {
    vol = pv.vol;
    val = pv.val;
    return *this;
  }
  // 操作符重载
  friend Migration::istream<> &
  operator>>(Migration::istream<> &is,
             patch_value_entry &pv)
  {
    is >> pv.vol >> pv.val;
    return is;
  }
  friend Migration::ostream<> &
  operator<<(Migration::ostream<> &os,
             const patch_value_entry &pv)
  {
    os << pv.vol << pv.val;
    return os;
  }
};
int main(int argc, char *argv[])
{
  int rank, n_rank;
  MPI_Init(&argc, &argv);
  int checkpoint = 1;
  while (0 == checkpoint)
  {
    sleep(5);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  std::cout << "argc:" << argc << std::endl;
  BinaryBuffer<> my_buf;
  Migration::ostream<> my_stream(my_buf);
  patch_value_entry my_pv;
  my_pv.vol = rank;
  std::fill(my_pv.val.begin(), my_pv.val.end(), double(rank) / n_rank);
  // std::cout << my_buf.size() << std::endl;
  my_stream << my_pv;
  int my_stream_size = my_buf.size();
  // std::cout << my_buf.size() << std::endl;
  // 通信 告知信息的长度
  std::vector<int> buf_size_vec(n_rank);
  // 集合通信
  MPI_Allgather(&my_stream_size, 1, MPI_INT, &buf_size_vec[0], 1, MPI_INT, MPI_COMM_WORLD);
  // 检查通信结果
  if (rank == 0)
  {
    for (auto ele_idx : buf_size_vec)
    {
      std::cout << ele_idx << " ";
    }
    std::cout << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // 准备相互传播
  srand(rank);
  int tag = 0; // 消息标签
  // std::cout << "tag " << tag << std::endl;
  int n_request = 0;
  MPI_Request request[2 * n_rank];
  MPI_Status status[2 * n_rank];
  // 共同序列
  std::vector<BinaryBuffer<>> common_buf(n_rank);
  for (int i = 0; i < n_rank; ++i)
  {
    common_buf[i].resize(buf_size_vec[i]);
  }
  for (int i = 0; i < n_rank; ++i)
  {
    MPI_Isend(my_buf.start_address(), my_stream_size, MPI_CHAR, i, tag, MPI_COMM_WORLD, &request[n_request++]);
    MPI_Irecv(common_buf[i].start_address(), buf_size_vec[i], MPI_CHAR, i, tag, MPI_COMM_WORLD, &request[n_request++]);
  }
  MPI_Waitall(n_request, request, status);
  // 数据转化
  for (int i = 0; i < n_rank; ++i)
  {
    Migration::istream<> out_stream(common_buf[i]);
    patch_value_entry pv_out;
    out_stream >> pv_out;
    if (rank == 0)
    {
      std::cout << "rank " << i << " ";
      std::cout << "pv_out.vol:" << pv_out.vol << " ";
      std::cout << "pv_out.val ";
      for (int j = 0; j < 4; ++j)
      {
        std::cout << pv_out.val[j] << " ";
      }
      std::cout << "\n";
    }
  }
  MPI_Finalize();
  return 0;
}
