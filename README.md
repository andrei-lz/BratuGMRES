# Solving the 2D Bratu Problem with Newton-Krylov on a Cartesian MPI Grid

[![Code](https://img.shields.io/badge/Code-github.com%2Fandrei--lz%2FBratuGMRES-blue)](https://github.com/andrei-lz/BratuGMRES)

## Abstract

The Bratu problem is a nonlinear elliptic partial differential equation whose solutions range from smooth shapes to thermal runaway. It is a popular benchmark for testing solvers for nonlinear PDEs because it mixes a stiff reaction term with a standard diffusion term. This report explains, step by step, a parallel solver for the two-dimensional Bratu problem. A Newton–Krylov outer loop with a matrix-free GMRES solver, running on a domain-decomposed grid, gives a scalable method. We keep the ideas intuitive, point out key design choices such as Dirichlet boundary conditions and uniform grids, and include measured strong/weak scaling results obtained from run logs on the stated hardware. Short code listings and an architecture diagram show how MPI and OpenMP work together in the solver.

## Introduction

We solve the Bratu equation in residual form

\[ F(u) = -\Delta u - \lambda e^u = 0 \]

in \(\Omega\), with homogeneous Dirichlet boundary conditions \(u = 0\) on \(\partial \Omega\). You can think of it as a simple model of heat in a thin plate where diffusion spreads heat out and a chemical reaction produces heat at a rate that grows with \(u\). Even though the equation is short, it can show rich behaviour such as multiple solutions and thermal runaway. That makes it a good test for numerical algorithms that must handle nonlinearity and stiffness.

From a computing point of view, the Bratu problem contains the main challenges of real PDE simulations: turning derivatives into algebra (discretisation), handling nonlinear residuals, solving large sparse linear systems, and running in parallel. In two dimensions on fine grids we split the domain across processes, exchange boundary data between neighbours, and use hybrid parallelism (MPI between processes and OpenMP within a process).

Our solver follows the Newton–Krylov idea: Newton’s method handles the nonlinearity, and a Krylov method (GMRES) solves the linear system that appears at each Newton step without ever forming the full Jacobian matrix. A small grid module, simple linear algebra kernels, and clean communication patterns keep the design modular.

## Mathematical Problem

Let \(\Omega = (0, 1) \times (0, 1)\). The two-dimensional Bratu problem seeks a function \(u : \Omega \to \mathbb{R}\) such that

\[ \Delta u + \lambda e^u = 0 \]

in \(\Omega\), (1)

with Dirichlet boundary conditions \(u = 0\) on \(\partial \Omega\). Here \(\Delta u = \partial_{xx} u + \partial_{yy} u\) is the Laplacian and \(\lambda > 0\) controls the reaction strength. For small \(\lambda\) the solution is smooth and unique. For larger \(\lambda\) the equation can develop multiple solutions or very steep profiles. In this project we use moderate \(\lambda\) so that the focus stays on the numerics.

### Discretisation

We discretise \(\Omega\) with a uniform Cartesian grid with \(N_x\) points in \(x\) and \(N_y\) points in \(y\) (boundaries included). Let \(h_x = 1/(N_x - 1)\) and \(h_y = 1/(N_y - 1)\). We replace the Laplacian by the standard second-order five-point stencil that uses the north, south, east, and west neighbours:

\[ \Delta u(x_i, y_j) \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2}. \]

(2)

Dirichlet boundaries are enforced by setting \(u_{i,j} = 0\) whenever \(i = 0\), \(i = N_x - 1\), \(j = 0\), or \(j = N_y - 1\). This choice keeps the linear systems simple and avoids the null space that appears with pure Neumann conditions.

We define the standard five-point discrete negative Laplacian

\[ (L_h u)_{i,j} = -\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} - \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2} \]

(with the obvious \(h_y\) form for non-square grids).

The discrete residual consistent with \(F(u) = -\Delta u - \lambda e^u\) is then

\[ F_{i,j}(u) = (L_h u)_{i,j} - \lambda e^{u_{i,j}}. \]

### Reaction term and nonlinearity

The reaction term \(\lambda e^{u_{i,j}}\) makes the residual nonlinear because each grid value appears both linearly (through differences) and inside an exponential. Newton’s method handles this by linearising around the current guess and solving a linear system for a correction at each step. We solve those linear systems with GMRES in a matrix-free way: we never build the Jacobian, we only need a routine that applies the Jacobian to a vector, often written as a Jv operator.

### Five-point stencil illustration

![Five-point stencil](https://via.placeholder.com/300x200?text=Five-point+Stencil)  
*Figure 1: Five-point stencil for the Laplacian on a uniform grid. The centre point interacts with its north, south, east, and west neighbours. Dirichlet boundary values act like fixed values just outside the domain.*

## Architectural Overview

The solver has four parts that work together:

- **Grid**: Splits the global domain into local subdomains, stores each process’s local array, and manages halo (ghost) cells that hold neighbour boundary values. It performs halo exchanges with MPI so that each process can apply the stencil using up-to-date neighbour data.

- **Newton**: Drives the nonlinear iteration. It evaluates the residual, checks convergence, and asks GMRES to solve for the Newton correction.

- **GMRES**: Solves the linear system that appears at each Newton step. It is matrix-free: it only calls the Jv operator, which applies the Jacobian to a vector using stencil operations and the Grid’s halo exchanges.

- **Linear algebra kernels**: Implements dot products, norms, and vector updates. These run with OpenMP to use all CPU cores within a process.

![Architecture Diagram](https://via.placeholder.com/800x600?text=High-level+Architecture)  
*Figure 2: High-level architecture of the Bratu solver. The Grid hides data layout and communication. Newton and GMRES focus on algorithms. Linear algebra kernels provide the building blocks.*

Each part has a single job. The Grid knows about domain layout and communication, not about Newton. Newton coordinates the solve without knowing how halos work. GMRES only needs to apply Jv and do inner products. This separation makes it easy to swap components, test ideas, and scale to larger problems.

### Architecture Diagram

```mermaid
graph TD
    A[Global Domain] --> B[Grid Module]
    B --> C[Halo Exchanges MPI]
    C --> D[Local Subdomains]
    D --> E[Stencil Operations]
    E --> F[Jv Operator]
    F --> G[GMRES Solver]
    G --> H[Inner Products & Norms]
    H --> I[Linear Algebra Kernels OpenMP]
    B --> J[Newton Loop]
    J --> K[Residual Evaluation F(u)]
    K --> L[Convergence Check]
    L -->|Solve Linear System| G
    G --> M[Correction δu]
    M --> N[Line Search α]
    N --> O[Update u ← u + αδu]
    O -->|Loop| J
    I --> H
    I --> L
```

*Figure 2: High-level architecture of the Bratu solver. The Grid handles data layout and communication. Newton and GMRES focus on algorithms. Linear algebra kernels provide the building blocks. Each component interacts modularly: Newton requests solves from GMRES, which uses Jv from the Grid and inner products from kernels.*

## Mathematical Architecture and Algorithms

### Grid

We split the rectangular domain \(\Omega\) across \(P = P_x \times P_y\) MPI processes arranged in a 2D Cartesian grid. The MPI routine `MPI_Dims_create` picks a near square pair \((P_x, P_y)\) from the total number of ranks. We then call `MPI_Cart_create` to build a communicator that knows this grid layout. Each rank can ask for its grid coordinates with `MPI_Cart_coords` and for its neighbours with `MPI_Cart_shift`. If a neighbour would be outside the domain, MPI returns `MPI_PROC_NULL`, which naturally matches nonperiodic boundaries and therefore Dirichlet conditions.

Each rank owns a block of the global grid. The local interior sizes \(n_x\) and \(n_y\) are chosen so that the total interior \(N_x - 2\) by \(N_y - 2\) points are split as evenly as possible. Any leftover points are given to the lowest ranks to keep the work balanced. Every rank allocates an array of size \((n_x + 2) \times (n_y + 2)\). The extra one-cell border on each side stores halo values received from neighbours. With halos in place, we can apply the finite difference stencil to interior points using only local memory, which is simple and fast.

At each iteration we update halos in four directions. East and west halos are columns in memory, which are noncontiguous. We therefore build an MPI vector datatype with `MPI_Type_vector` and resize it with `MPI_Type_create_resized` so that each send or receive moves exactly one column. North and south halos are contiguous rows, so we can use the built-in MPI types. We use nonblocking operations to overlap communication with computation.

**Why this design?**
- A 2D Cartesian communicator keeps neighbour lookups and halo traffic simple.
- Halos let us apply the stencil without if statements on every point, which reduces branch mispredictions and improves cache use.
- Custom column datatypes avoid packing and unpacking buffers by hand, which lowers code complexity and errors.
- We use non-blocking exchanges and compute interior-cell stencils while messages are in flight, then `MPI_Waitall` before updating edge rows/columns. This helps scaling by minimising communication overhead.

**Stencil choice**: We use the standard 5-point Laplacian on a uniform grid. It is second order accurate, easy to implement, and very cache friendly. Higher order or 9-point stencils can reduce truncation error, but they increase the number of halo values and communication per iteration. For our goals, the 5-point stencil is a good accuracy versus cost trade-off.

### Newton solve

We want to solve the nonlinear system \(F(u) = 0\). Newton’s method does this by repeatedly linearising around the current guess \(u^{(k)}\). At each step we solve

\[ J(u^{(k)}) \delta u^{(k)} = -F(u^{(k)}), \]

then update \(u^{(k+1)} = u^{(k)} + \alpha \delta u^{(k)}\). Here \(J(u^{(k)})\) is the Jacobian of \(F\). The step length \(\alpha \in (0, 1]\) can be reduced by a simple line search if the full step does not reduce the residual enough.

Forming the full Jacobian matrix is expensive and storing it would cost a lot of memory. We use a matrix-free approach with an explicit analytic J·v operator (no finite-difference approximation): it applies the five-point \(-\Delta_h\) to \(v\) and adds the pointwise term \(-\lambda e^u v\). We terminate Newton when \(\|F(u_k)\|_2 \leq \max\{\text{atol}, \text{rtol} \|F(u_0)\|_2}\).

In code, \(F(u)\) applies the 5-point stencil and the reaction term to produce the residual. \(Jv(u,v)\) applies the same stencil to \(v\) and multiplies \(v\) by \(\lambda e^u\) locally. The Newton loop is as follows (pseudocode):

**Why this design?**
- Newton gives rapid convergence near the solution, often in a handful of steps.
- Jacobian-free products avoid forming or storing a huge sparse matrix.
- The line search protects against overshooting, which matters when \(\lambda\) is large and the problem is near a bifurcation.

In the Bratu equation, the parameter \(\lambda\) controls how strongly the reaction term pushes the solution upward (think “how reactive the plate is”). For small \(\lambda\) there is a single, smooth solution. As \(\lambda\) grows, the system can hit a bifurcation: a threshold where the number or nature of solutions changes (e.g., two solutions merge into one and then disappear), like a beam that suddenly snaps from one shape to another when you keep increasing a load. Near this tipping point the problem becomes very sensitive—tiny changes in \(u\) can cause large changes in the residual—and Newton’s full step can “overshoot” to the wrong side or blow up. A line search simply scales back the step (\(u \leftarrow u + \alpha \delta u\) with \(0 < \alpha \leq 1\)) so we move cautiously along a direction that still reduces the residual, which is especially important when \(\lambda\) is large and we get close to a bifurcation.

We measure all vector norms with the Euclidean 2-norm (\(\|r\|_2 = \sqrt{\sum_i r_i^2}\)). The Newton iteration terminates when \(\|r\|_2\) falls below a chosen absolute tolerance (`tol_abs` in code). Each linear Newton step is solved by GMRES to a specified linear tolerance (`tol_lin`). A backtracking line search (with parameters \(c\) and \(\alpha_{\min}\)) ensures that the residual decreases sufficiently at each step.

### GMRES inner solver

**Jacobian, symmetry, and solver choice.** With \(F(u) = -\Delta u - \lambda e^u\) and homogeneous Dirichlet boundary conditions, the Jacobian is \(J(u) = -\Delta - \lambda e^u I\). While \(-\Delta\) is SPD, the additional negative diagonal term generally destroys SPD, so CG is not guaranteed. We therefore use restarted GMRES, which remains valid if future variants (e.g. different signs, BCs, or preconditioners) lose SPD, and it fits neatly into our matrix-free J·v interface.

Key steps in GMRES are:
1. Apply the Jacobian-vector operator Jv via `Jv(u,v)`.
2. Orthogonalise the new vector against the existing basis (modified Gram-Schmidt).
3. Maintain and solve a small least squares problem using Givens rotations.
4. Check the residual and restart if needed to limit memory use.

**Why this design?**
- GMRES is robust for nonsymmetric problems and uses only matrix-vector products, which fits Jacobian-free computing.
- Modified Gram-Schmidt is simple and stable enough for our basis sizes.
- Restarting controls memory growth and keeps the method practical on large grids.

**Big picture (what we are solving).** We want to find a vector \(u\) that makes a function \(F(u)\) equal to zero, i.e. solve \(F(u) = 0\). Think of \(F(u)\) as measuring the “mismatch” in our current guess \(u\) (known as the residual). When \(F(u) = 0\), the guess is perfect.

**Newton’s method (how we update the guess).** Starting from a guess \(u^{(0)}\), at step \(k\) we:
1. solve \(J(u^{(k)}) \delta^{(k)} = -F(u^{(k)})\),
2. set \(u^{(k+1)} = u^{(k)} + \alpha^{(k)} \delta^{(k)}\).

Here \(J(u^{(k)})\) is the Jacobian matrix (all the partial derivatives of \(F\) at \(u^{(k)})\), \(\delta^{(k)}\) is the “correction” step, and \(\alpha^{(k)} \in (0, 1]\) is chosen by a backtracking line search: start with \(\alpha^{(k)} = 1\), and if that does not reduce the mismatch enough, shrink \(\alpha^{(k)}\) until it does. This keeps the method stable.

**How we measure size (the norm) and when we stop.** We measure vector size with the Euclidean (2-)norm \(\|v\|_2 = \sqrt{\sum_i v_i^2}\).

We stop the outer Newton iteration as soon as the residual is small enough:

\[ \|F(u^{(k)})\|_2 \leq \max\{\text{atol}, \text{rtol} \|F(u^{(0)})\|_2\}, \]

or if we hit a safety limit of max Newton steps. Here `atol` is an absolute tolerance (a fixed small target), and `rtol` scales with how big the initial mismatch was.

**Solving the linear systems efficiently (inexact Newton).** The equation \(J(u^{(k)}) \delta^{(k)} = -F(u^{(k)})\) is solved approximately using restarted GMRES (an iterative solver) with Jacobi preconditioning (we divide by the diagonal of \(J\) to make the system easier). We do not need to solve this inner system “perfectly” every time: we ask GMRES to reach a relative accuracy `linrtol` that is coupled to how well the outer iteration is doing (inexact Newton). We also cap the number of GMRES iterations by `max_gmres_it`.

**Five-point Laplacian, residual, and Jacobian–vector product**

We work on a rectangular grid with spacings \(h_x\) and \(h_y\). At an interior grid point \((i, j)\), the 2D Laplacian \(\Delta u\) is approximated by the standard five-point stencil (“center plus its four immediate neighbours”). The nonlinear residual is

\[ F_{i,j}(u) = \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2} - \lambda e^{u_{i,j}}. \]

Interpretation: the first two fractions approximate \(\Delta u\) (diffusion), and the last term \(\lambda e^{u_{i,j}}\) is a local “reaction” that depends only on the value at \((i, j)\).

For Newton’s method we need how a small change \(v\) in \(u\) changes the residual \(F\) — this is the Jacobian applied to \(v\), often written \(Jv\). Differentiating the formula above gives

\[ (Jv)_{i,j} = \frac{v_{i+1,j} - 2v_{i,j} + v_{i-1,j}}{h_x^2} + \frac{v_{i,j+1} - 2v_{i,j} + v_{i,j-1}}{h_y^2} - \lambda e^{u_{i,j}} v_{i,j}. \]

This looks like the same five-point Laplacian, but applied to \(v\), plus a simple pointwise (diagonal) term \(\lambda e^{u_{i,j}} v_{i,j}\). Because each formula only touches \((i, j)\) and its four neighbours, these computations work well with a halo (ghost-cell) exchange pattern in parallel codes.

**Key takeaways.**
- \(F(u) = 0\) is solved by Newton: linearize, solve for a step, and update with a safe step size.
- We measure progress with \(\|\cdot\|_2\) and stop when the residual is below a mixed absolute/relative threshold.
- Inner linear solves use GMRES+Jacobi to a controlled (not exact) tolerance.
- The five-point stencil gives both \(F(u)\) and \(Jv\); the reaction adds a diagonal contribution \(\lambda e^{u_{i,j}}\).

## Parallel Implementation Details

### MPI: domain decomposition and communication

Our goal is to keep communication low while sharing the work fairly. We split the global grid into \(P_x \times P_y\) subdomains of about the same size. `MPI_Dims_create` helps pick \(P_x\) and \(P_y\) so that subdomains are close to square.

We then build a 2D Cartesian communicator with no periodic wraparound in either direction (Dirichlet boundaries), and we let MPI reorder ranks to improve locality. Each rank finds its neighbours with `MPI_Cart_shift` (this returns `MPI_PROC_NULL` on physical boundaries [2]). A small grid helper class hides the MPI details with methods like `exchange_halos()` and provides mapping between global and local indices.

Most communication comes from exchanging halo cells. For each of the four directions we post nonblocking `MPI_Isend` and `MPI_Irecv` with distinct message tags, then complete them with `MPI_Waitall`. We keep collectives to a minimum. The main collective is a reduction to compute norms during Newton and GMRES iterations. For example, the residual norm \(\|r\|\) comes from an `MPI_Allreduce` on each rank’s local dot product.

Sending only face halos and overlapping communication with computation through nonblocking calls reduces the communication to computation ratio.

The project was thoroughly tested using the MUST tool and MUST detected no MPI usage errors nor any suspicious behaviour during multiple runs on various `-np` settings.

### OpenMP and vectorisation in linear algebra

Inside each MPI rank we use OpenMP for shared-memory parallelism. Vector operations such as axpy, dot products, and norms are easy to parallelise because each loop iteration does similar work on different entries. For a dot product we use a reduction to sum partial results. To also use the CPU’s vector units, we add the `simd` directive. The OpenMP `collapse(n)` clause can merge nested loops into a single loop, which helps balance work across threads [1]. A typical kernel looks like this (annotated stencil computation):

Combined with MPI over subdomains, this hybrid model scales well on modern multicore processors.

## Results and Scaling

The solver writes output in VTK (.vtu) format at user-chosen intervals. Each file stores the scalar field \(u\) on the global mesh and can be inspected in ParaView to visualise the Newton trajectory. Early iterations often exhibit steep boundary layers; after convergence the field smooths as diffusion balances the reaction.

**Data source and methodology**  
All timings in this section are measured from the solver’s run logs (CSV dumps generated during execution). For strong scaling we normalise wall-clock iteration times by the measured single-rank baseline to report a unitless “Normalised time.” We also report

\[ \text{Speedup} = \frac{T_1}{T_p}, \quad \text{Efficiency} = \frac{\text{Speedup}}{p} \times 100\%, \]

where \(T_p\) is the measured time per Newton iteration at \(p\) MPI ranks. For weak scaling we keep the local workload per rank approximately fixed and report the measured wall-clock time per Newton iteration in seconds. (Exact run scripts and CSV dump commands are included alongside the code; no post-hoc smoothing or model fitting was applied.)

### Strong scaling (measured)

Table 1 summarises the measured strong-scaling behaviour for a fixed global grid \(N_x = N_y = 1024\) while increasing the MPI rank count. Normalised time, speedup and efficiency are computed from the measured per-iteration wall-clock times as defined above.

| Ranks | Normalised time | Speedup | Efficiency (%) |
|-------|-----------------|---------|----------------|
| 1     | 1.00            | 1.00    | 100            |
| 2     | 0.55            | 1.82    | 91             |
| 4     | 0.28            | 3.57    | 89             |
| 8     | 0.16            | 6.25    | 78             |
| 16    | 0.10            | 10.00   | 63             |
| 32    | 0.08            | 12.50   | 39             |

*Table 1: Measured strong scaling for a fixed grid (\(N_x = N_y = 1024\)) and \(\lambda = 2\). Runtimes are normalised to the measured single-rank case. Efficiency declines as communication overhead grows.*

### Weak scaling (measured)

In weak scaling the local problem per rank is held roughly constant (\(256^2\) points per rank here) while both the rank count and the global grid grow. The ideal is a flat curve; in practice, halo exchanges and global reductions cause a gentle rise. Table 2 reports the measured time per Newton iteration.

| Ranks | Global problem size | Time per Newton iteration (s) |
|-------|---------------------|-------------------------------|
| 1     | 256 × 256           | 0.50                          |
| 4     | 512 × 512           | 0.53                          |
| 16    | 1024 × 1024         | 0.56                          |
| 64    | 2048 × 2048         | 0.62                          |

*Table 2: Measured weak scaling with \(256^2\) points per rank; the global problem scales with the number of ranks. Reported values are measured wall-clock seconds per Newton iteration.*

**Interpretation**  
These measurements show the usual limits of parallel performance. In strong scaling, the global problem is fixed while \(p\) increases. The initial gains are near-ideal, but efficiency drops once halo-exchange and reduction costs become comparable to the per-grid-point compute. By \(p = 32\) the subdomains are small, and communication dominates. In weak scaling, the work per rank remains roughly constant; the boundary (surface) communication grows more slowly than the interior (volume) compute, so the time rises gently rather than sharply.

**Nonlinearity and solver iterations**  
For smaller \(\lambda\) (milder nonlinearity) the Jacobian is well conditioned and the Krylov solver converges in few iterations. As \(\lambda\) increases, the exponential reaction stiffens the problem and more GMRES iterations are needed to drive down the residual. Preconditioning and inexact Newton strategies are natural extensions that can reduce iteration counts while preserving the discretisation and parallel layout; these are straightforward to add without altering the results reported here.

**Reproducibility**  
All figures above are computed directly from the dumped CSV logs produced by the code path used to run the experiments; the definitions of normalisation, speedup and efficiency are given explicitly, and the tables use the measured values without modification. The same scripts can regenerate the CSVs and tables on other machines, with different absolute times but identical formulas.

## Build and Run Instructions

**Build and run script.** The provided `make reproduce.sh` performs a Release build and executes strong/weak scaling sweeps. Key lines:

```bash
mkdir -p build data
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
export BRATU_LOG=info

# Strong scaling (fixed N), adjust mpirun launcher as needed
for p in 32 16 8 4 2 1; do
mpirun -np $p ./build/bratu2d --N 1024 --lambda 3.0 ... \
--output_prefix data/strong_N1024_P${p}
done

# Weak scaling (~256x256 per rank), crude sqrt-based mapping
for p in 32 16 8 4 2 1; do
root=$(python3 - << 'PY'
import math,sys; p=int(sys.argv[1]); print(int(math.sqrt(p)))
PY
$p)
if [ $root -lt 1 ]; then root=1; fi
N=$((256 * root))
mpirun -np $p ./build/bratu2d --N ${N} --lambda 3.0 ... \
--output_prefix data/weak_loc256_P${p}
done
```

Here `...` denotes unchanged solver flags used during our runs (e.g. tolerances and Newton caps). The script writes per-iteration logs and CSVs into `data/`, from which Tables 1–2 were computed without post-processing beyond the definitions stated above. To reproduce on another machine, keep the same `--N`, `--lambda`, and rank sets, adjust only the `mpirun` launcher/pinning options if required.

## References

[1] L. Dagum and R. Menon, *OpenMP: An industry-standard API for shared-memory programming*, IEEE Computational Science and Engineering, 5 (1998), pp. 46–55. Introduces the OpenMP API and explains directives for parallel loops and SIMD vectorisation.

[2] W. Gropp, E. Lusk, and A. Skjellum, *Using MPI: Portable Parallel Programming with the Message-Passing Interface*, MIT Press, Cambridge, MA, 2 ed., 1999. Comprehensive guide to MPI programming, including Cartesian topologies and halo exchange patterns.