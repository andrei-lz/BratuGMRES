#include "bratu.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

static inline int NX(const Grid &g) { return g.nx + 2; }
static inline int NY(const Grid &g) { return g.ny + 2; }

void apply_F(const Cart2D &cart, Grid &g, double lambda)
{
  exchange_halos(cart, g, g.u);
  const int nxh = NX(g);
  const double h2 = g.h * g.h;
#pragma omp parallel for collapse(2)
  for (int j = 1; j < NY(g) - 1; ++j)
  {
    for (int i = 1; i < nxh - 1; ++i)
    {
      const int k = idx(i, j, nxh);
      const double uc = g.u[k];
      const double lap =
          (-g.u[k + 1] - g.u[k - 1] - g.u[k + nxh] - g.u[k - nxh] + 4.0 * uc) /
          h2;
      g.F[k] = lap - lambda * std::exp(uc);
    }
  }
  // Dirichlet boundary via halos (zeros).
}

void apply_Jv(const Cart2D &cart, Grid &g, const std::vector<double> &v,
              std::vector<double> &y, double lambda)
{
  const int nxh = NX(g);
  const double h2 = g.h * g.h;
  g.tmp = v;

  exchange_halos(cart, g, g.tmp);

  if ((int)y.size() != (int)g.u.size())
    y.assign(g.u.size(), 0.0);

#pragma omp parallel for collapse(2)
  for (int j = 1; j < NY(g) - 1; ++j)
  {
    for (int i = 1; i < nxh - 1; ++i)
    {
      const int k = idx(i, j, nxh);
      const double factor = (4.0 / h2 - lambda * std::exp(g.u[k]));
      y[k] = factor * g.tmp[k] -
             (g.tmp[k + 1] + g.tmp[k - 1] + g.tmp[k + nxh] + g.tmp[k - nxh]) *
                 (1.0 / h2);
    }
  }
}

void update_diagJ(const Cart2D & /*cart*/, Grid &g, double lambda)
{
  const int nxh = NX(g);
  const double h2 = g.h * g.h;

#pragma omp parallel for collapse(2)
  for (int j = 1; j < NY(g) - 1; ++j)
  {
    for (int i = 1; i < nxh - 1; ++i)
    {
      const int k = j * nxh + i;
      g.diagJ[k] = (4.0 / h2 - lambda * std::exp(g.u[k]));
    }
  }
}

void apply_Jacobi(const Grid &g, const std::vector<double> &r,
                  std::vector<double> &z)
{

  if ((int)z.size() != (int)r.size())
    z.resize(r.size());

#pragma omp parallel for
  for (long k = 0; k < r.size(); ++k)
  {
    z[k] = r[k] / g.diagJ[k];
  }
}
