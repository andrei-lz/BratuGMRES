#pragma once
#include <string>

struct Options
{
  int N = 512;
  double lambda = 3.0;
  double rtol = 1e-8;
  int gmres_restart = 50;
  int max_newton = 30;
  int max_gmres_it = 400;
  int omp_threads = 8;
  double ls_c1 = 1e-4;
  double ls_beta = 0.5;
  std::string output_prefix = "data/run";
};

Options parse_options(int argc, char **argv);