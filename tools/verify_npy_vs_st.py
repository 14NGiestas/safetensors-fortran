#!/usr/bin/env python3
"""verify_npy_vs_st.py -- prove the .safetensors holds the same bytes as the .npy set.

Three-way check, all on RAW BYTES (so NaN/Inf and -0.0 are compared exactly):

  1. for every canonical tensor, the payload in the .safetensors must be
     byte-identical to the payload of its source .npy file;
  2. the FNV-1a 64 of that payload is recomputed here;
  3. optionally (`--inspect-bin`), the Fortran reader is executed on the same file
     and its printed FNV-1a 64 per tensor must match (1) and (2) -- that is the
     end-to-end "Fortran reads what Python wrote, bit for bit" evidence.

Usage:
    python3 tools/verify_npy_vs_st.py CKPT_DIR FILE.safetensors \
        [--inspect-bin path/to/inspect]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import npy_to_safetensors as conv  # noqa: E402
import reference_writer as rw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("st_file")
    ap.add_argument("--inspect-bin", default="")
    args = ap.parse_args()

    arch = conv.read_arch(os.path.join(args.ckpt_dir, "arch.txt"))
    tensors = conv.collect(args.ckpt_dir, arch, include_optimizer=False)
    expected = {n: p for n, _d, _s, p, _f in tensors}
    source = {n: f for n, _d, _s, _p, f in tensors}

    metadata, got = rw.read(args.st_file)
    ok = True
    print("%-10s %-42s %9s  %-16s %s" % ("tensor", "source .npy", "bytes", "fnv1a64", "payload equal"))
    py_hashes = {}
    for name, payload in expected.items():
        if name not in got:
            print("%-10s %-42s %9s  %-16s MISSING IN .safetensors" % (name, source[name], "-", "-"))
            ok = False
            continue
        h = rw.fnv1a64(got[name]["data"])
        py_hashes[name] = h
        same = got[name]["data"] == payload
        ok = ok and same
        print("%-10s %-42s %9d  %016x %s"
              % (name, source[name], len(payload), h, "yes" if same else "NO"))
    print("")
    print("payloads compared: %d, all byte-identical: %s" % (len(expected), ok))
    print("metadata keys    : %d" % len(metadata))
    print("tensor_manifest.sha256 = %s" % metadata.get("tensor_manifest.sha256", "(absent)"))
    print("manifest.sha256        = %s" % (metadata.get("manifest.sha256") or "(absent)"))

    if args.inspect_bin:
        if not os.path.exists(args.inspect_bin):
            print("inspect binary not found: %s" % args.inspect_bin)
            return 2
        out = subprocess.run(
            [args.inspect_bin, args.st_file, "--hash"], capture_output=True, text=True
        )
        if out.returncode != 0:
            print("inspect failed: %s" % out.stderr.strip())
            return 2
        ft_hashes = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) == 6 and parts[0] in py_hashes:
                ft_hashes[parts[0]] = int(parts[-1], 16)
        mismatches = [n for n in py_hashes if ft_hashes.get(n) != py_hashes[n]]
        print("")
        print("fortran reader    : %s tensors hashed by '%s'" % (len(ft_hashes), args.inspect_bin))
        print("fortran vs python : %s" % ("ALL MATCH" if not mismatches else "MISMATCH %s" % mismatches))
        if len(ft_hashes) != len(py_hashes) or mismatches:
            ok = False
    print("")
    print("VERIFY %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
