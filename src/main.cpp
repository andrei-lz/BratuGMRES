#include "grid.hpp"
#include "io.hpp"
#include "newton.hpp"
#include "options.hpp"
#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  auto world = MPI_COMM_WORLD;

  Options opt = parse_options(argc, argv);
  if (opt.omp_threads > 0)
    omp_set_num_threads(opt.omp_threads);

  Cart2D cart = make_cart2d(world);
  Grid g = make_local_grid(cart, opt.N);

  auto stats = newton_solve(cart, g, opt);

  if (cart.rank == 0)
  {
    std::cout << "Converged=" << stats.converged
              << " newton_iters=" << stats.newton_iters
              << " final_res=" << stats.final_res << "\n";
  }

  write_vti_cellcentered(opt.output_prefix + "_rank" + std::to_string(cart.rank) + ".vti", cart, g);

  MPI_Comm_free(&cart.comm);
  MPI_Finalize();
  return 0;
}
