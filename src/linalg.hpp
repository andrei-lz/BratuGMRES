#pragma once
#include "grid.hpp"
#include "mpi.h"
#include <vector>
#include "debug.hpp"

void axpby(double a, const std::vector<double> &x, double b,
           std::vector<double> &y); // y = a*x + b*y
void scal(double a, std::vector<double> &x);

double dot(const Grid &g, const std::vector<double> &a, const std::vector<double> &b);
double nrm2(const Grid &g, const std::vector<double> &a);

static inline double nrm2_global(MPI_Comm comm, int it, const Grid &g, const std::vector<double> &a){
  double local_ss = dot(g, a, a); // interior-only sum of squares
  double global_ss = local_ss;
  MPI_CALL(
      comm, it, "MPI_Allreduce (nrm2)",
      MPI_Allreduce(MPI_IN_PLACE, &global_ss, 1, MPI_DOUBLE, MPI_SUM, comm));
  return std::sqrt(global_ss);
}

static inline double dot_global(MPI_Comm comm, int it, const Grid &g,
                                const std::vector<double> &a,
                                const std::vector<double> &b){
  double loc = dot(g, a, b);
  double g_dot = loc;
  MPI_CALL(comm, it, "Allreduce(dot)",
                 MPI_Allreduce(MPI_IN_PLACE, &g_dot, 1, MPI_DOUBLE, MPI_SUM, comm));
  return g_dot;
}