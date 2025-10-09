# Solving the 2D Bratu Problem with Newton-Krylov on a Cartesian MPI Grid

[![Code](https://img.shields.io/badge/Code-github.com%2Fandrei--lz%2FBratuGMRES-blue)](https://github.com/andrei-lz/BratuGMRES)

I put together this parallel solver for the 2D Bratu problem. It's a classic nonlinear PDE that mixes diffusion with a stiff exponential reaction term. Think of it as modeling generated heat in a thin metal plate with phenomenons like thermal runaway. I wanted a scalable way to tackle this, so I went with a Newton-Krylov approach using matrix-free GMRES, all on a domain-decomposed grid with MPI and OpenMP. It's a great benchmark for nonlinear solvers, and I've got it running reasonably efficiently on multi-core setups.

## Description

The Bratu equation is \[-Δu - λ e^u = 0\] on a unit square with zero Dirichlet boundaries. For small λ, solutions are smooth; crank it up, and you get multiple solutions or steep profiles—perfect for testing stiff numerics.

I discretize it on a uniform Cartesian grid using a five-point stencil for the Laplacian. Newton's method handles the nonlinearity, while GMRES solves the linear steps without building the full Jacobian (just matrix-vector products). Everything's parallelized: MPI splits the domain, halo exchanges keep boundaries in sync, and OpenMP speeds up local ops like dots and norms.

This project's about keeping things modular and intuitive—grid management separate from solvers, clean comms, and measured scaling from real runs.

## Features

- **Scalable Parallelism**: Domain decomposition over MPI Cartesian topology, with non-blocking halo exchanges for low-latency comms.
- **Matrix-Free Solvers**: Newton outer loop with inexact GMRES inner solver— no huge Jacobians stored!
- **Hybrid Threading**: OpenMP for intra-node speedups in linear algebra kernels, including SIMD vectorization.
- **Easy Visualization**: Outputs VTK files for ParaView to watch the solution evolve (steep layers early, smooth balance later).
- **Tunable Params**: Adjust λ, grid size (N), tolerances, and max iterations via command-line flags.
- **Benchmark-Ready**: Strong/weak scaling logs in CSV for reproducibility— I tested on various rank counts with no MPI errors (thanks, MUST tool!).
- **Lightweight Deps**: Just CMake, MPI, and OpenMP—no fancy libs needed.

## Quick Start

Needs a C++ compiler, CMake (3.12+), MPI (like OpenMPI or MPICH, only tested with OpenMPI), and OpenMP support. ParaView is optional for viz.

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### Run a Simple Example
Fire it up on 4 ranks for a 512x512 grid with λ=2:
```bash
mpirun -np 4 ./bratu2d --N 512 --lambda 2.0 --atol 1e-8 --max_newton_iters 20 --output_prefix data/solution
```
This dumps VTK files like `data/solution_iter_*.vtu`. Open them in ParaView to see the heat field.

For scaling sweeps, use the provided `reproduce.sh` script (tweak mpirun if needed):
```bash
scripts/make_reproduce.sh
```
It runs strong/weak tests and logs CSVs in `data/`.

## Architecture Overview

The solver's built around four clean modules—keeps it easy to hack and extend.

- **Grid**: Splits the domain into subgrids per MPI rank, handles halos with non-blocking sends/recvs. Uses Cartesian communicator for neighbor discovery—super simple for Dirichlet BCs.
- **Newton**: Drives the nonlinear solve. Computes residuals, does backtracking line search for stability near bifurcations, and calls GMRES for corrections.
- **GMRES**: Matrix-free linear solver with Jacobi preconditioning. Builds Krylov basis via Jv products (stencil + diagonal reaction).
- **LinAlg Kernels**: Dot products, norms, axpys—parallelized with OpenMP `#pragma omp parallel for simd`.

Here's a quick look at the halo exchange in action (from `grid.cpp`):
```cpp
// Non-blocking column sends/recvs for east/west
MPI_Isend(&u[local_nx + 1 + 1*(local_nx+2)], 1, col_type_resized, east, 0, mpi_cart_comm, &reqs[0]);
MPI_Irecv(&u[1*(local_nx+2)], 1, col_type_resized, west, 0, mpi_cart_comm, &reqs[1]);
// ... similar for north/south
MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
```
The five-point stencil is cache-friendly and second-order accuracy trades off higher-order complexity for speed.

For the big picture:  
<img width="3840" height="2767" alt="bratudiagram" src="https://github.com/user-attachments/assets/ad3c59c4-b207-48a8-8cd5-23b8ec3e8bb1" />
*(Grid feeds data to Newton/GMRES; lin alg kernels power the math.)*

This separation lets you swap solvers without touching the grid implementation.

## Results

Ran this on a multi-core machine, logging per-iteration times. For moderate λ=2-3, Newton converges in ~5-10 steps, GMRES in 20-50 per step.

### Strong Scaling
Fixed grid (1024x1024), more ranks = smaller subdomains. Efficiency holds till comms dominate.

| Ranks | Normalized Time | Speedup | Efficiency (%) |
|-------|-----------------|---------|----------------|
| 1     | 1.00            | 1.00    | 100            |
| 2     | 0.55            | 1.82    | 91             |
| 4     | 0.28            | 3.57    | 89             |
| 8     | 0.16            | 6.25    | 78             |
| 16    | 0.10            | 10.00   | 63             |
| 32    | 0.08            | 12.50   | 39             |

### Weak Scaling
~256x256 per rank, global size grows. Time rises gently from log(N) reductions.

| Ranks | Global Size    | Time per Newton (s) |
|-------|----------------|---------------------|
| 1     | 256x256        | 0.50                |
| 4     | 512x512        | 0.53                |
| 16    | 1024x1024      | 0.56                |
| 64    | 2048x2048      | 0.62                |

As λ ramps up, more iterations needed. More preconditioning could help, but this is a good start.

## Reproducibility

All results come straight from CSV logs. The `reproduce.sh` script rebuilds and runs sweeps:
```bash
# Strong scaling example
for p in 32 16 8 4 2 1; do
  mpirun -np $p ./build/bratu2d --N 1024 --lambda 3.0 --atol 1e-8 --output_prefix data/strong_N1024_P${p}
done

# Weak scaling (~256/rank)
for p in 64 16 4 1; do
  N=$((256 * $(python3 -c "import math; print(int(math.sqrt($p)))")))
  mpirun -np $p ./build/bratu2d --N ${N} --lambda 3.0 --output_prefix data/weak_loc256_P${p}
done
```
Run on your hardware—times vary, but speedups/efficiencies follow the formulas. Tested with MUST: zero MPI issues.
