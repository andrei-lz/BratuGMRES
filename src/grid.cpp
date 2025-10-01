#include "grid.hpp"
#include <algorithm>
#include <cassert>
#include "debug.hpp"
#include "timing.hpp"

static inline int NX(const Grid &g) { return g.nx + 2; }
static inline int NY(const Grid &g) { return g.ny + 2; }

// Build 2D Cartesian communicator and neighbors.
Cart2D make_cart2d(MPI_Comm world)
{
  Cart2D c;
  c.comm = world;
  MPI_Comm_rank(world, &c.rank);
  MPI_Comm_size(world, &c.size);
  int dims[2] = {0, 0};
  MPI_Dims_create(c.size, 2, dims);
  c.dims[0] = dims[0];
  c.dims[1] = dims[1];
  int periods[2] = {0, 0};

  MPI_Comm cart_comm;
  MPI_Cart_create(world, 2, dims, periods, 1, &cart_comm);

  // Cartesian communicator from here on
  c.comm = cart_comm;
  MPI_Comm_set_errhandler(c.comm, MPI_ERRORS_RETURN); // show errors
  MPI_Comm_rank(c.comm, &c.rank);                     // rank in CART

  MPI_Cart_coords(c.comm, c.rank, 2, c.coords);
  // dim 0: x (i), west/east
  int src, dst;
  MPI_Cart_shift(c.comm, 0, 1, &src, &dst);
  c.nbr_w = src;
  c.nbr_e = dst;
  // dim 1: y (j), south/north
  MPI_Cart_shift(c.comm, 1, 1, &src, &dst);
  c.nbr_s = src;
  c.nbr_n = dst;
  c.px = dims[0];
  c.py = dims[1];
  fprintf(stderr, "[%d] dims=(%d,%d) coords=(%d,%d) W=%d E=%d S=%d N=%d\n",
          c.rank, dims[0], dims[1], c.coords[0], c.coords[1], c.nbr_w, c.nbr_e,
          c.nbr_s, c.nbr_n);

  return c;
}

// Split global N approximately evenly across px,py; compute offsets.
Grid make_local_grid(const Cart2D &cart, int N_global)
{
  Grid g;
  g.N_global = N_global;
  int ix = cart.coords[0], iy = cart.coords[1];
  int nx_base = N_global / cart.px;
  int nx_rem = N_global % cart.px;
  int ny_base = N_global / cart.py;
  int ny_rem = N_global % cart.py;

  g.nx = nx_base + (ix < nx_rem ? 1 : 0);
  g.ny = ny_base + (iy < ny_rem ? 1 : 0);

  g.gx0 = ix * nx_base + std::min(ix, nx_rem);
  g.gy0 = iy * ny_base + std::min(iy, ny_rem);

  g.h = 1.0 / (N_global + 1);

  size_t NXh = g.nx + 2;
  size_t NYh = g.ny + 2;
  g.u.assign(NXh * NYh, 0.0);
  g.F.assign(NXh * NYh, 0.0);
  g.tmp.assign(NXh * NYh, 0.0);
  g.diagJ.assign(NXh * NYh, 1.0);
  fprintf(stderr, "[%d] local nx=%d ny=%d  gx0=%d gy0=%d  Nglob=%d\n",
          cart.rank, g.nx, g.ny, g.gx0, g.gy0, g.N_global);

  return g;
}

static double g_comm_time_accum = 0.0;

double grid_comm_time_get()   { return g_comm_time_accum; }
void   grid_comm_time_reset() { g_comm_time_accum = 0.0; }

// Nonblocking exchange of 1-cell halos in N,S,E,W using derived types for
// columns.
void exchange_halos(const Cart2D &cart, const Grid &g, std::vector<double> &vec)
{
  Timer t; t.tic();
  ScopedPhase _p(cart.comm, "halo", "exchange_halos", /*it*/0);

  const int nxh = g.nx + 2;

  // column type for vertical halos
  MPI_Datatype col_vec, col_type;
  MPI_Type_vector(g.ny, 1, nxh, MPI_DOUBLE, &col_vec);
  MPI_Type_create_resized(col_vec, 0, sizeof(double), &col_type);
  MPI_Type_commit(&col_type);
  MPI_Type_free(&col_vec);

  enum
  {
    TAG_EASTWARD = 101,
    TAG_WESTWARD = 102,
    TAG_SOUTHWARD = 103,
    TAG_NORTHWARD = 104
  };

  MPI_Request reqs[8];
  int rcount = 0;

  // --- Recvs (into halos) ---
  // West halo  (i=0)       <= west neighbor’s east interior (eastward)
  MPI_Irecv(&vec[idx(0, 1, nxh)], 1, col_type, cart.nbr_w,
                      TAG_EASTWARD, cart.comm, &reqs[rcount++]);

  // East halo  (i=nx+1)    <= east neighbor’s west interior (westward)
  MPI_Irecv(&vec[idx(g.nx + 1, 1, nxh)], 1, col_type, cart.nbr_e,
                      TAG_WESTWARD, cart.comm, &reqs[rcount++]);

  // South halo (j=0)       <= south neighbor’s north row (northward)
  MPI_Irecv(&vec[idx(1, 0, nxh)], g.nx, MPI_DOUBLE, cart.nbr_s,
                      TAG_NORTHWARD, cart.comm, &reqs[rcount++]);

  // North halo (j=ny+1)    <= north neighbor’s south row (southward)
  MPI_Irecv(&vec[idx(1, g.ny + 1, nxh)], g.nx, MPI_DOUBLE, cart.nbr_n,
                      TAG_SOUTHWARD, cart.comm, &reqs[rcount++]);

  // --- Sends (from interior borders) ---
  // To WEST neighbor (westward)
  MPI_Isend(&vec[idx(1, 1, nxh)], 1, col_type, cart.nbr_w,
                      TAG_WESTWARD, cart.comm, &reqs[rcount++]);

  // To EAST neighbor (eastward)
  MPI_Isend(&vec[idx(g.nx, 1, nxh)], 1, col_type, cart.nbr_e,
                      TAG_EASTWARD, cart.comm, &reqs[rcount++]);

  // To SOUTH neighbor (southward)
  MPI_Isend(&vec[idx(1, 1, nxh)], g.nx, MPI_DOUBLE, cart.nbr_s,
                      TAG_SOUTHWARD, cart.comm, &reqs[rcount++]);

  // To NORTH neighbor (northward)
  MPI_Isend(&vec[idx(1, g.ny, nxh)], g.nx, MPI_DOUBLE, cart.nbr_n,
                      TAG_NORTHWARD, cart.comm, &reqs[rcount++]);

  // --- Complete all traffic ---
  MPI_Status stats[8];
  int rc = MPI_Waitall(rcount, reqs, stats);
  if (rc != MPI_SUCCESS)
  {
    // show which request failed (helps if some neighbor/rank is wrong)
    for (int i = 0; i < rcount; i++)
    {
      int e = stats[i].MPI_ERROR;
      if (e != MPI_SUCCESS)
      {
        char s[MPI_MAX_ERROR_STRING];
        int l = 0;
        MPI_Error_string(e, s, &l);
        fprintf(stderr, "MPI_Waitall: req %d failed: %.*s\n", i, l, s);
      }
    }
    MPI_Abort(cart.comm, rc);
  }

  MPI_Type_free(&col_type);
  g_comm_time_accum += t.toc();
}
