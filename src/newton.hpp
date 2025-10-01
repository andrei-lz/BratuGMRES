#pragma once
#include "gmres.hpp"
#include "grid.hpp"
#include "options.hpp"

struct NewtonStats
{
  int newton_iters = 0;
  double final_res = 0.0;
  bool converged = false;
};

NewtonStats newton_solve(const Cart2D &cart, Grid &g, const Options &opt);
