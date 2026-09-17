#!/usr/bin/env bash
# test/parity_cmp.sh -- byte-for-byte parity evidence, with the exact cmp commands.
#
#   1. fpm test         (the suite also compares the bytes in-process)
#   2. cmp              of the file the library wrote against the fixture produced
#                       by the OFFICIAL implementation (or by the pure-Python
#                       oracle when the official one is unavailable)
#   3. sha256           of both, so the evidence can be reproduced elsewhere
#
# Usage: bash test/parity_cmp.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# fpm lives either as `fpm` (CI/homebrew) or as `fortran-fpm` (nixpkgs).
FPM="${FPM:-}"
if [ -z "$FPM" ]; then
  if command -v fpm >/dev/null 2>&1; then FPM=fpm; else FPM=fortran-fpm; fi
fi
echo "using fpm: $FPM"

# Python with numpy (+ safetensors, ideally) for the oracle side.
PY="${PYTHON:-python3}"
echo "using python: $PY"

echo "== fpm test =="
"$FPM" test

echo
echo "== byte-for-byte parity (cmp) =="
pairs=(
  "build/st_test/parity_lib.safetensors:test/fixtures/parity_oracle.safetensors"
  "build/st_test/escapes_lib.safetensors:test/fixtures/escapes_oracle.safetensors"
)
for pair in "${pairs[@]}"; do
  a="${pair%%:*}"
  b="${pair##*:}"
  if cmp "$a" "$b"; then
    echo "cmp: IDENTICAL  $a  ==  $b"
  else
    echo "cmp: DIFFERENT  $a  vs  $b"
    exit 1
  fi
done

echo
echo "== sha256 of both sides =="
sha256sum build/st_test/parity_lib.safetensors test/fixtures/parity_oracle.safetensors
sha256sum build/st_test/escapes_lib.safetensors test/fixtures/escapes_oracle.safetensors

echo
echo "== optional: regenerate the fixtures with the official package and re-cmp =="
if "$PY" -c "import safetensors" 2>/dev/null; then
  "$PY" tools/make_fixtures.py --out /tmp/st_fixtures_regen
  cmp /tmp/st_fixtures_regen/parity_oracle.safetensors test/fixtures/parity_oracle.safetensors \
    && echo "cmp: regenerated parity fixture is identical to the committed one"
  cmp /tmp/st_fixtures_regen/escapes_oracle.safetensors test/fixtures/escapes_oracle.safetensors \
    && echo "cmp: regenerated escaping fixture is identical to the committed one"
else
  echo "SKIP: the official 'safetensors' package is not importable in this python ($PY)"
fi

echo
echo "PARITY OK"
