#include <bits/stdc++.h>
#include <mpi.h>
struct patchinfo
{
  double ele_vol;
  double U[4];
};
// 将数据打包（序列化） 便于传输
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int myid, rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  MPI_Finalize();
  return 0;
}
