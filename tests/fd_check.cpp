#include "grid.hpp"
#include "bratu.hpp"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  Cart2D cart = make_cart2d(MPI_COMM_WORLD);
  Grid g = make_local_grid(cart, 32);
  double lambda = 2.0;

  std::vector<double> v(g.u.size(),0.0), Jv(g.u.size(),0.0), F0, F1;
  for(size_t i=0;i<v.size();++i) v[i] = 0.5 + (i%7)*0.01;

  apply_F(cart, g, lambda); F0 = g.F;
  apply_Jv(cart, g, v, Jv, lambda);

  double eps = 1e-6;
  for(size_t i=0;i<g.u.size();++i) g.u[i] += eps*v[i];
  apply_F(cart, g, lambda); F1 = g.F;

  double num=0.0, den=0.0;
  for(size_t i=0;i<F0.size();++i){
    double diff = (F1[i]-F0[i])/eps - Jv[i];
    num += diff*diff; den += Jv[i]*Jv[i];
  }
  double local[2]={num,den}, global[2]={0,0};
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, cart.comm);
  if(cart.rank==0){
    std::cout<<"FD relative error = "<<std::sqrt(global[0]/std::max(1e-30,global[1]))<<"\n";
  }
  MPI_Finalize();
  return 0;
}
