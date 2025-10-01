#!/usr/bin/env bash
set -euo pipefail

mkdir -p build data
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

export BRATU_LOG=info

# Strong scaling example (adjust mpirun to your launcher/env)
for p in 6 4 2 1; do
  mpirun -np $p ./build/bratu2d --N 1024 --lambda 3.0 --rtol 1e-8 --gmres_restart 100 --max_newton 30 --output_prefix data/strong_N1024_P${p}
done

# Weak scaling example: aim for ~square proc grid; adjust N accordingly
for p in 6 4 2 1; do
  # crude mapping to keep ~256x256 per rank assuming square Px x Py
  # Use integer sqrt; refine as needed.
  root=$(python3 - << 'PY'
import math,sys
p=int(sys.argv[1])
print(int(math.sqrt(p)))
PY
$p)
  if [ $root -lt 1 ]; then root=1; fi
  N=$((256 * root))
  mpirun -np $p ./build/bratu2d --N ${N} --lambda 3.0 --rtol 1e-8 --gmres_restart 100 --max_newton 30 --output_prefix data/weak_loc256_P${p}
done
