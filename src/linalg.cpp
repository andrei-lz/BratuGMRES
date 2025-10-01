#include "linalg.hpp"
#include <cmath>
#include <omp.h>

double dot(const Grid &g, const std::vector<double> &a,
           const std::vector<double> &b)
{
  const int nxh = g.nx + 2;
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) collapse(2) schedule(static)
  for (int j = 1; j <= g.ny; ++j)
  {
    for (int i = 1; i <= g.nx; ++i)
    {
      const int k = idx(i, j, nxh);
      s += a[k] * b[k];
    }
  }
  return s;
}

double nrm2(const Grid &g, const std::vector<double> &a)
{
  return std::sqrt(dot(g, a, a));
}

void axpby(double a, const std::vector<double> &x, double b,
           std::vector<double> &y)
{
  if (y.size() != x.size())
    y.resize(x.size());
#pragma omp simd
  for (size_t i = 0; i < x.size(); ++i)
    y[i] = a * x[i] + b * y[i];
}
void scal(double a, std::vector<double> &x)
{
#pragma omp simd
  for (size_t i = 0; i < x.size(); ++i)
    x[i] *= a;
}
