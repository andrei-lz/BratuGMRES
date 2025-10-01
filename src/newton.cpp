#include "newton.hpp"
#include "bratu.hpp"
#include "linalg.hpp"
#include "timing.hpp"
#include <cmath>
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

  // For measurements
  double res0 = -1.0;
  double atol = 0.0; 

  for (int it = 0; it < opt.max_newton; ++it)
  {
    // F(u)
    apply_F(cart, g, opt.lambda);
    double resnorm = global_norm2(cart.comm, it, g, g.F);
    if (res0 < 0.0) res0 = std::max(resnorm, 1e-30);
    double rel = resnorm / res0;
    LOGI("newton", it, "||F|| = {:.6e}  (rel={:.6e})", resnorm, rel);

    // Stopping: abs OR relative (robust)
    bool abs_ok = (resnorm <= atol);
    bool rel_ok = (resnorm <= opt.rtol * res0);
    if (abs_ok || rel_ok) {
      st.converged   = true;
      st.newton_iters= it;
      st.final_res   = resnorm;
      LOGI("newton", it, "CONVERGED abs_ok={} rel_ok={}  final ||F||={:.6e}",
           abs_ok, rel_ok, resnorm);                                     // >>> log
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

    // Inexact-Newton coupling: inner tol follows outer progress (cap between 1e-8 and 1e-2*rel)
    double linrtol = std::min(1e-8, std::max(1e-2 * rel, 1e-6));

    GMRESStats gst = gmres_solve(cart.comm, it, n,
                                 opt.gmres_restart, opt.max_gmres_it,
                                 linrtol, Av, MInv, rhs, delta, g, nullptr);
    LOGI("gmres", it, "iters={} relres={:.3e} target={:.1e} conv={}",
         gst.iters, gst.relres, linrtol, (int)gst.converged);

    // Line search (Armijo)
    double alpha = 1.0;
    const double c1 = opt.ls_c1, beta = opt.ls_beta;
    std::vector<double> u_save = g.u;

    int ls_tries = 0;
    for (int ls = 0; ls < 20; ++ls)
    {
      g.u = u_save;
      axpby(alpha, delta, 1.0, g.u);
      apply_F(cart, g, opt.lambda);
      double ftrial = global_norm2(cart.comm, it, g, g.F);
      if (ftrial * ftrial <= resnorm * resnorm * (1.0 - 2.0 * c1 * alpha)) {
        LOGI("linesearch", it, "accept alpha={:.3f} ftrial={:.3e}", alpha, ftrial); // >>>
        break;
      }
      alpha *= beta;
      ++ls_tries;
    }
    if (ls_tries > 0) LOGI("linesearch", it, "backtracks={}", ls_tries);           // >>>
    st.newton_iters = it + 1;
  }

  // Final residual
  apply_F(cart, g, opt.lambda);
  st.final_res = global_norm2(cart.comm, opt.max_newton, g, g.F);
  return st;
}
