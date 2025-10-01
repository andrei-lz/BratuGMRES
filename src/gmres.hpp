#pragma once
#include "grid.hpp"
#include <functional>
#include <mpi.h>
#include <vector>

struct GMRESStats
{
  int iters = 0;
  double relres = 0.0;
  bool converged = false;
};

// Matrix-free GMRES(restart) with right preconditioning callback.
// Av(x, y): y = A*x
// MInv(r, z): z = M^{-1} r
GMRESStats gmres_solve(MPI_Comm comm, int iter_id, int n, int restart,
                       int maxit, double rtol,
                       const std::function<void(const std::vector<double> &,
                                                std::vector<double> &)> &Av,
                       const std::function<void(const std::vector<double> &,
                                                std::vector<double> &)> &MInv,
                       const std::vector<double> &b, std::vector<double> &x,
                       const Grid &grid, std::vector<double> * /*work_buffer*/
);
