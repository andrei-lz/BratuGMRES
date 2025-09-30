#include "newton.hpp"
#include "bratu.hpp"
#include "linalg.hpp"
#include "timing.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <vector>

static double global_norm2(MPI_Comm comm, int iter, const Grid &g,
                           const std::vector<double> &v)
{
  double local = nrm2(g, v);
  double z = local * local;
  MPI_CALL(comm, iter, "MPI_Allreduce (resnorm)",
                 MPI_Allreduce(MPI_IN_PLACE, &z, 1, MPI_DOUBLE, MPI_SUM, comm));
  return std::sqrt(z);
}

NewtonStats newton_solve(const Cart2D &cart, Grid &g, const Options &opt)
{
  NewtonStats st;
  const int n = (int)g.u.size();
  std::vector<double> rhs(n, 0.0), delta(n, 0.0);

  for (int it = 0; it < opt.max_newton; ++it)
  {
    // F(u)
    apply_F(cart, g, opt.lambda);
    double resnorm = global_norm2(cart.comm, it, g, g.F);
    if (cart.rank == 0)
      std::cout << "Newton " << it << " residual " << resnorm << "\n";
    if (resnorm < opt.rtol)
    {
      st.converged = true;
      st.newton_iters = it;
      st.final_res = resnorm;
      return st;
    }

    // rhs = -F
    rhs = g.F;
    scal(-1.0, rhs);
    // Update Jacobi diagonal
    update_diagJ(cart, g, opt.lambda);

    // Callbacks
    auto Av = [&](const std::vector<double> &x, std::vector<double> &y)
    { apply_Jv(cart, g, x, y, opt.lambda); };
    auto MInv = [&](const std::vector<double> &r, std::vector<double> &z)
    { apply_Jacobi(g, r, z); };

    // Solve J δ = -F with GMRES
    delta.assign(n, 0.0);
    (void)gmres_solve(cart.comm, it, n, opt.gmres_restart, opt.max_gmres_it,
                      1e-3, Av, MInv, rhs, delta, g, nullptr);

    // Line search (Armijo)
    double alpha = 1.0;
    const double c1 = opt.ls_c1, beta = opt.ls_beta;
    std::vector<double> u_save = g.u;

    for (int ls = 0; ls < 20; ++ls)
    {
      // u_trial = u + alpha*delta
      g.u = u_save;
      axpby(alpha, delta, 1.0, g.u);
      apply_F(cart, g, opt.lambda);
      double ftrial = global_norm2(cart.comm, it, g, g.F);
      double ftrial_sq = ftrial * ftrial;
      double resnorm_sq = resnorm * resnorm;
      if (ftrial_sq <= resnorm_sq * (1.0 - 2.0 * c1 * alpha)) { break; } // sufficient decrease
      alpha *= beta;
    }
    st.newton_iters = it + 1;
  }

  // Final residual
  apply_F(cart, g, opt.lambda);
  st.final_res = global_norm2(cart.comm, opt.max_newton, g, g.F);
  return st;
}
