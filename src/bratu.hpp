#pragma once
#include "grid.hpp"

// Apply residual F(u)
void apply_F(const Cart2D &cart, Grid &g, double lambda);

// Apply Jacobian-vector product y = J(u) * v (matrix-free)
void apply_Jv(const Cart2D &cart, Grid &g, const std::vector<double> &v,
              std::vector<double> &y, double lambda);

// Build/refresh Jacobi preconditioner diagJ = diag(J(u))
void update_diagJ(const Cart2D &cart, Grid &g, double lambda);

// Jacobi preconditioner: z = M^{-1} r
void apply_Jacobi(const Grid &g, const std::vector<double> &r,
                  std::vector<double> &z);
