#pragma once
#include <mpi.h>
#include <vector>

struct Cart2D
{
  MPI_Comm comm = MPI_COMM_NULL;
  int dims[2]{0, 0};
  int coords[2]{0, 0};
  int nbr_w = MPI_PROC_NULL;
  int nbr_e = MPI_PROC_NULL;
  int nbr_s = MPI_PROC_NULL;
  int nbr_n = MPI_PROC_NULL;
  int rank = -1;
  int size = 0;
  int px = 1, py = 1; // process grid
};

struct Grid
{
  int N_global = 0;     // global interior N (NxN)
  int nx = 0, ny = 0;   // local interior sizes
  int gx0 = 0, gy0 = 0; // global offsets of local interior
  double h = 0.0;
  // Arrays include 1-cell halos on all sides: (nx+2)*(ny+2), row-major
  std::vector<double> u, F, tmp, diagJ;
};

Cart2D make_cart2d(MPI_Comm world);
Grid make_local_grid(const Cart2D &cart, int N_global);

// Exchange 1-cell ghost layers (N,S,E,W). Dirichlet boundary => halos stay zero
// when neighbor is PROC_NULL.
void exchange_halos(const Cart2D &cart, const Grid &g, std::vector<double> &vec);

// Linear index helper
inline int idx(int i, int j, int nx_with_halo) { return j * nx_with_halo + i; }
