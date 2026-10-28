#!/usr/bin/env bash
# Run the compiled Fortran ISOTOPIA on every sample case and store the results
# as references for scripts/run_samples.py --ref.
#
#   scripts/run_fortran_samples.sh                 # all samples
#   scripts/run_fortran_samples.sh p-Mo100-Tc099m  # selected samples
#
# Uses $ISOTOPIA_DIR (the Fortran repository, with bin/isotopia built by
# `make` in source/) and writes to runs/fortran/<case>/. Set ISOTOPIA_BIN to
# use a different executable.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
: "${ISOTOPIA_DIR:?set ISOTOPIA_DIR to the isotopia (Fortran) directory}"
bin=${ISOTOPIA_BIN:-$ISOTOPIA_DIR/bin/isotopia}
[[ -x "$bin" ]] || { echo "no executable at $bin (run make in $ISOTOPIA_DIR/source)" >&2; exit 1; }
export ISOTOPIA_DIR ISOTOPIA_USER="${ISOTOPIA_USER:-Arjan Koning}"

samples=$ISOTOPIA_DIR/samples
out=$here/runs/fortran
if [[ $# -gt 0 ]]; then
  cases=("$@")
else
  cases=()
  for d in "$samples"/*/; do
    [[ -f "$d/org/isotopia.inp" ]] && cases+=("$(basename "$d")")
  done
fi

for c in "${cases[@]}"; do
  tmp=$(mktemp -d)
  cp "$samples/$c/org/isotopia.inp" "$tmp/"
  t0=$(date +%s)
  (cd "$tmp" && "$bin" < isotopia.inp > isotopia.out)
  rm -rf "$out/$c"; mkdir -p "$out/$c"
  cp "$tmp"/* "$out/$c/"; rm -rf "$tmp"
  printf '%-22s %3d s\n' "$c" $(( $(date +%s) - t0 ))
done
echo "references written to $out; compare with: python scripts/run_samples.py --ref runs/fortran"
