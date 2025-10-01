#include "gmres.hpp"
#include "linalg.hpp"
#include "mpi.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Compact restarted GMRES for demonstration; adequate for medium N.
GMRESStats gmres_solve(MPI_Comm comm, int iter_id, int n, int restart,
                       int maxit, double rtol,
                       const std::function<void(const std::vector<double> &,
                                                std::vector<double> &)> &Av,
                       const std::function<void(const std::vector<double> &,
                                                std::vector<double> &)> &MInv,
                       const std::vector<double> &b, std::vector<double> &x,
                       const Grid &grid, std::vector<double> * /*work_buffer*/
)
{
  GMRESStats st;
  std::vector<double> r(n), z(n), w(n);
  // r = b - A x
  Av(x, r);
  axpby(1.0, b, -1.0, r);
  double bnorm = nrm2_global(comm, iter_id, grid, b);
  double rnorm = nrm2_global(comm, iter_id, grid, r);
  if (rnorm / bnorm < rtol)
  {
    st.converged = true;
    st.relres = rnorm / bnorm;
    return st;
  }

  std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> H(restart + 1,
                                     std::vector<double>(restart, 0.0));
  std::vector<double> cs(restart, 0.0), sn(restart, 0.0);

  int iter = 0;
  while (iter < maxit)
  {

    double beta = rnorm;
    for (int i = 0; i < n; i++)
      V[0][i] = r[i] / beta;

    std::vector<double> g(restart + 1, 0.0);
    g[0] = beta;
    int j;
    for (j = 0; j < restart && iter < maxit; ++j, ++iter)
    {
      // z = M^{-1} v_j
      MInv(V[j], z);
      // w = A z
      Av(z, w);
      // Arnoldi
      for (int i2 = 0; i2 <= j; i2++)
      {
        H[i2][j] = dot_global(comm, iter, grid, V[i2], w);
        axpby(-H[i2][j], V[i2], 1.0, w);
      }
      H[j + 1][j] = nrm2_global(comm, iter, grid, w);
      if (H[j + 1][j] > 0)
      {
        for (int i = 0; i < n; i++)
          V[j + 1][i] = w[i] / H[j + 1][j];
      }
      // Apply existing Givens
      for (int i2 = 0; i2 < j; i2++)
      {
        double tmp = cs[i2] * H[i2][j] + sn[i2] * H[i2 + 1][j];
        H[i2 + 1][j] = -sn[i2] * H[i2][j] + cs[i2] * H[i2 + 1][j];
        H[i2][j] = tmp;
      }
      // New Givens
      double denom = std::hypot(H[j][j], H[j + 1][j]);
      cs[j] = (denom == 0.0) ? 1.0 : (H[j][j] / denom);
      sn[j] = (denom == 0.0) ? 0.0 : (H[j + 1][j] / denom);
      H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
      H[j + 1][j] = 0.0;
      // Update g
      double tmp = cs[j] * g[j];
      g[j + 1] = -sn[j] * g[j];
      g[j] = tmp;

      double rel = std::abs(g[j + 1]) / bnorm;
      if (rel < rtol)
      {
        j++;
        iter++;
        break;
      }
    }

    // Solve upper-triangular system (least-squares reduction already applied)
    int m = j;
    std::vector<double> y(m, 0.0);
    for (int i = m - 1; i >= 0; --i)
    {
      double s = g[i];
      for (int k = i + 1; k < m; ++k)
        s -= H[i][k] * y[k];
      y[i] = s / H[i][i];
    }
    // Update x = x + Z_m y  (Z_i = M^{-1} V_i)
    for (int i = 0; i < m; i++)
    {
      MInv(V[i], z);
      axpby(y[i], z, 1.0, x);
    }

    Av(x, r);
    axpby(1.0, b, -1.0, r);
    rnorm = nrm2_global(comm, iter, grid, r);
    if (rnorm / std::max(1e-30, bnorm) < rtol)
    {
      st.converged = true;
      break;
    }
  }
  st.iters = iter;
  // Guards against division by 0
  const double denom = (bnorm > 0.0 ? bnorm : 1.0);
  st.relres = rnorm / denom;
  LOGI("gmres", iter_id, "exit iters={} relres={:.3e} converged={}",
     st.iters, st.relres, (int)st.converged);
  return st;
}
