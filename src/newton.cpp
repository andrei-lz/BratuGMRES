#include "newton.hpp"
#include "bratu.hpp"
#include "linalg.hpp"
#include "timing.hpp"
#include <cmath>
#include <mpi.h>
#include <vector>

static double global_norm2(MPI_Comm comm, int iter, const Grid &g,
                           const std::vector<double> &v) {
  double local = nrm2(g, v);
  double z = local * local;
  MPI_CALL(comm, iter, "MPI_Allreduce (resnorm)",
           MPI_Allreduce(MPI_IN_PLACE, &z, 1, MPI_DOUBLE, MPI_SUM, comm));
  return std::sqrt(z);
}

NewtonStats newton_solve(const Cart2D &cart, Grid &g, const Options &opt) {
  NewtonStats st;
  const int n = (int)g.u.size();
  std::vector<double> rhs(n, 0.0), delta(n, 0.0);

  // For measurements
  double res0 = -1.0;
  double atol = 0.0; // if you later add opt.atol, wire it here
  const int myrank = rank_of(cart.comm);

  // Whole-solve timer (for strong/weak scaling)
  double run_t0 = MPI_Wtime();

  for (int it = 0; it < opt.max_newton; ++it) {
    // Per-step timer
    double step_t0 = MPI_Wtime();

    // F(u)
    apply_F(cart, g, opt.lambda);
    double resnorm = global_norm2(cart.comm, it, g, g.F);
    if (res0 < 0.0)
      res0 = std::max(resnorm, 1e-30);
    double rel = resnorm / res0;

    LOGI("newton", it, "||F|| = {:.6e}  (rel={:.6e})", resnorm, rel);

    // Log stopping rule + create CSV header once (it == 0, rank 0)
    if (it == 0 && myrank == 0) {
      LOGI("newton", it,
           "stopping rule: ||F|| <= max(atol, rtol*||F0||) with atol={:.3e}, "
           "rtol={:.1e}, ||F0||={:.6e}",
           atol, opt.rtol, res0);

      // create/overwrite CSV with header
      if (std::FILE *fph = std::fopen("bratu_newton_trace.csv", "w")) {
        std::fprintf(
            fph, "it,Fnorm,rel,gmres_it,gmres_rel,linrtol,alpha,step_tmax\n");
        std::fclose(fph);
      }
    }

    // Stopping: abs OR relative (robust)
    const bool abs_ok = (resnorm <= atol);
    const bool rel_ok = (resnorm <= opt.rtol * res0);
    if (abs_ok || rel_ok) {
      st.converged = true;
      st.newton_iters = it;
      st.final_res = resnorm;
      LOGI("newton", it, "CONVERGED abs_ok={} rel_ok={}  final ||F||={:.6e}",
           abs_ok, rel_ok, resnorm);

      // Per-step max time across ranks
      double step_dt = MPI_Wtime() - step_t0;
      double step_dt_max = 0.0;
      MPI_Reduce(&step_dt, &step_dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart.comm);
      if (myrank == 0) {
        LOGI("time", it, "step_time_max={:.6f}s", step_dt_max);
        // Append a CSV row for the convergent step; GMRES/alpha fields are
        // NaN/-1
        if (std::FILE *fp = std::fopen("bratu_newton_trace.csv", "a")) {
          std::fprintf(fp, "%d,%.16e,%.16e,%d,%.16e,%.1e,%.6f,%.6f\n", it,
                       resnorm, rel, -1,
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(), step_dt_max);
          std::fclose(fp);
        }
      }

      // Whole-run time (max across ranks)
      double run_dt = MPI_Wtime() - run_t0;
      double run_dt_max = 0.0;
      MPI_Reduce(&run_dt, &run_dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart.comm);
      if (myrank == 0)
        LOGI("time", -1, "solve_time_max={:.6f}s", run_dt_max);

      return st;
    }

    // Refresh diagonal of J(u) for Jacobi preconditioner
    update_diagJ(cart, g, opt.lambda);

    // Callbacks
    auto Av = [&](const std::vector<double> &x, std::vector<double> &y) {
      apply_Jv(cart, g, x, y, opt.lambda);
    };
    auto MInv = [&](const std::vector<double> &r, std::vector<double> &z) {
      apply_Jacobi(g, r, z);
    };

    // Solve J δ = -F with GMRES
    delta.assign(n, 0.0);

    // Inexact-Newton coupling: inner tol follows outer progress
    // cap between 1e-8 and 1e-2*rel (with a floor of 1e-6)
    double linrtol = std::min(1e-8, std::max(1e-2 * rel, 1e-6));

    // Build RHS: -F(u)
    for (int i = 0; i < n; ++i)
      rhs[i] = -g.F[i];

    // Solve (J) delta = -F with GMRES
    GMRESStats gst =
        gmres_solve(cart.comm, it, n, opt.gmres_restart, opt.max_gmres_it,
                    linrtol, Av, MInv, rhs, delta, g, nullptr);
    LOGI("gmres", it, "iters={} relres={:.3e} target={:.1e} conv={}", gst.iters,
         gst.relres, linrtol, (int)gst.converged);

    // Line search: u <- u + alpha * delta
    const double c1 = 1e-4;
    const double beta = 0.5;
    double alpha = 1.0;
    int ls_tries = 0;

    std::vector<double> u_save = g.u;

    for (int ls = 0; ls < 20; ++ls) {
      g.u = u_save;
      axpby(alpha, delta, 1.0, g.u);

      apply_F(cart, g, opt.lambda);
      double ftrial = global_norm2(cart.comm, it, g, g.F);

      if (ftrial * ftrial <= resnorm * resnorm * (1.0 - 2.0 * c1 * alpha)) {
        LOGI("linesearch", it, "accept alpha={:.3f} ftrial={:.3e}", alpha,
             ftrial);
        break;
      }
      alpha *= beta;
      ++ls_tries;
    }
    if (ls_tries > 0)
      LOGI("linesearch", it, "backtracks={}", ls_tries);

    st.newton_iters = it + 1;

    // Per-step timing (max across ranks) and CSV row (rank 0 only)
    double step_dt = MPI_Wtime() - step_t0;
    double step_dt_max = 0.0;
    MPI_Reduce(&step_dt, &step_dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart.comm);
    if (myrank == 0) {
      LOGI("time", it, "step_time_max={:.6f}s", step_dt_max);
      if (std::FILE *fp = std::fopen("bratu_newton_trace.csv", "a")) {
        std::fprintf(fp, "%d,%.16e,%.16e,%d,%.16e,%.1e,%.6f,%.6f\n", it,
                     resnorm, rel, gst.iters, gst.relres, linrtol, alpha,
                     step_dt_max);
        std::fclose(fp);
      }
    }
  }

  // If we exit via max_newton: record final residual and whole-run time
  apply_F(cart, g, opt.lambda);
  st.final_res = global_norm2(cart.comm, opt.max_newton, g, g.F);

  double run_dt = MPI_Wtime() - run_t0;
  double run_dt_max = 0.0;
  MPI_Reduce(&run_dt, &run_dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart.comm);
  if (myrank == 0)
    LOGI("time", -1, "solve_time_max={:.6f}s", run_dt_max);

  return st;
}
