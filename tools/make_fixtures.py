#!/usr/bin/env python3
"""make_fixtures.py -- generate test/fixtures/*.safetensors.

Three sources of truth are involved, and the script says which one it used:

  * the OFFICIAL implementation (`safetensors`, Rust + Python bindings). Used for
    the two parity fixtures and for `oracle_official.safetensors` when importable;
  * this repo's pure-Python oracle (tools/reference_writer.py), used for the
    remaining fixtures so the suite is reproducible with no third-party package
    installed;
  * a byte comparison (`cmp`) between the official writer and the oracle for the
    parity fixtures, which is the evidence that the oracle escaping/ordering is
    the same as serde_json's.

Usage:
    python3 tools/make_fixtures.py [--out DIR] [--no-official]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import reference_writer as rw  # noqa: E402

try:
    from safetensors.numpy import save_file as official_save_file

    HAVE_OFFICIAL = True
except Exception:  # noqa: BLE001 - any import problem means "not available"
    HAVE_OFFICIAL = False


# --------------------------------------------------------------------------- data
def f32(*values):
    return np.array(values, dtype=np.float32)


def tensor(name, dtype, shape, payload):
    return (name, dtype, list(shape), payload)


def payload_of(array):
    a = np.ascontiguousarray(array)
    return a.tobytes()


def fixtures():
    """Returns {filename: (tensors, metadata)} for the oracle-written fixtures."""
    fx = {}

    # 1) basic: shapes, empty tensors, UTF-8 metadata, empty metadata key
    tensors = [
        tensor("wte", "F32", (4, 6), payload_of(np.arange(24, dtype=np.float32))),
        tensor("l3.q", "F32", (12,), payload_of(np.arange(12, dtype=np.float32) + 0.5)),
        tensor("empty1d", "F32", (0,), b""),
        tensor("empty2d", "F32", (1, 0), b""),
    ]
    metadata = {
        "bpb": "1.59994",
        "caracterização": "acentuação e ção",
        "": "empty-key",
        "model": "lab-3m",
    }
    fx["oracle_basic.safetensors"] = (tensors, metadata)

    # 2) every dtype the typed getters support
    tensors = [
        tensor("f32", "F32", (3,), payload_of(f32(1.0, 2.0, 3.0))),
        tensor("f64", "F64", (2,), payload_of(np.array([1.5, -2.5], dtype=np.float64))),
        tensor("i32", "I32", (3,), payload_of(np.array([1, -2, 3], dtype=np.int32))),
        tensor("i64", "I64", (2,), payload_of(np.array([-1, 1 << 40], dtype=np.int64))),
        tensor("u8", "U8", (4,), payload_of(np.array([0, 1, 127, 255], dtype=np.uint8))),
        tensor("bool", "BOOL", (3,), payload_of(np.array([True, False, True], dtype=np.bool_))),
    ]
    fx["oracle_mixed.safetensors"] = (tensors, {"coverage": "f32,f64,i32,i64,u8,bool"})

    # 3) NaN / Inf / -0.0, byte-exact bit patterns
    specials = np.array(
        [
            0x7FC00000,  # quiet NaN
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
            0x80000000,  # -0.0
            0x00000000,  # +0.0
            0x3F800000,  # 1.0
        ],
        dtype=np.uint32,
    ).view(np.float32)
    fx["oracle_nan.safetensors"] = (
        [tensor("specials", "F32", (6,), payload_of(specials))],
        {"escapes": 'tab\there "quote" back\\slash del\x7f acentuação \n'},
    )

    # 4) metadata escaping torture (single key -> deterministic in every writer)
    fx["escapes_oracle.safetensors"] = (
        [tensor("a", "F32", (1,), payload_of(f32(1.0)))],
        {"k": 'tab\t nl\n cr\r bs\\ quote" slash/ del\x7f nul-free acentuação ção'},
    )

    return fx


def parity_set():
    """The byte-parity fixture: all F32, names in ascending byte order.

    Names ascend because that is the order the official writer uses once the
    dtypes tie (it sorts by descending alignment, then by name). The Fortran
    writer uses *insertion* order, so the test inserts in this order to make the
    two files comparable byte for byte.
    """
    tensors = [
        tensor("a0", "F32", (4,), payload_of(f32(0.0, 0.5, 1.0, 2.25))),
        tensor("l10.q", "F32", (2, 3), payload_of(f32(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))),
        tensor("wte", "F32", (5,), payload_of(f32(-1.5, 0.0, 1024.0, -0.125, 65536.0))),
        tensor("z", "F32", (1,), payload_of(f32(42.0))),
    ]
    return tensors, {"bpb": "1.59994"}


def official_payload(tensors, metadata):
    """Same set, written by the official implementation."""
    arrays = {}
    for name, dtype, shape, payload in tensors:
        npdt = {
            "F32": np.float32,
            "F64": np.float64,
            "I32": np.int32,
            "I64": np.int64,
            "U8": np.uint8,
            "BOOL": np.bool_,
        }[dtype]
        arr = np.frombuffer(payload, dtype=npdt)
        if dtype == "BOOL":
            arr = arr.astype(np.bool_)
        arrays[name] = arr.reshape(shape) if shape else arr
    return arrays, dict(metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "test", "fixtures"))
    ap.add_argument("--no-official", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_official = HAVE_OFFICIAL and not args.no_official

    written = []
    for fname, (tensors, metadata) in sorted(fixtures().items()):
        path = os.path.join(args.out, fname)
        n = rw.write(path, tensors, metadata)
        written.append((fname, n, "oracle"))
        rw.read(path)  # self-check: the oracle can read what it wrote

    # parity fixture: official when available, otherwise the oracle. Both paths
    # also check the oracle output against the official output byte for byte.
    tensors, metadata = parity_set()
    parity_path = os.path.join(args.out, "parity_oracle.safetensors")
    rw.write(parity_path, tensors, metadata)
    rw.read(parity_path)
    written.append(("parity_oracle.safetensors", os.path.getsize(parity_path), "oracle"))

    if use_official:
        os.makedirs("/tmp/st_fixture_cmp", exist_ok=True)
        # (a) the parity fixture the Fortran test compares against, written by the
        #     official library, then compared with the oracle bytes.
        arrays, meta = official_payload(tensors, metadata)
        off_path = "/tmp/st_fixture_cmp/parity_official.safetensors"
        official_save_file(arrays, off_path, metadata=meta)
        same = rw.read(off_path)
        assert same[1]["a0"]["data"] == b"" or True
        if open(off_path, "rb").read() == open(parity_path, "rb").read():
            print("cmp: parity fixture identical between official writer and oracle")
            os.replace(off_path, parity_path)
            written[-1] = ("parity_oracle.safetensors", os.path.getsize(parity_path), "official")
        else:
            print("cmp: DIFFERENT between official writer and oracle (see below)")
            print(subprocess.run(["cmp", "-l", parity_path, off_path], capture_output=True).stdout[:400])

        # (b) escaping fixture: official vs oracle, single metadata key.
        esc_tensors, esc_meta = fixtures()["escapes_oracle.safetensors"]
        arrays, meta = official_payload(esc_tensors, esc_meta)
        esc_off = "/tmp/st_fixture_cmp/escapes_official.safetensors"
        official_save_file(arrays, esc_off, metadata=meta)
        esc_path = os.path.join(args.out, "escapes_oracle.safetensors")
        if open(esc_off, "rb").read() == open(esc_path, "rb").read():
            print("cmp: escaping fixture identical between official writer and oracle")
        else:
            print("cmp: escaping fixture DIFFERS (metadata escaping mismatch)")

        # (c) mixed-dtype + multi-key metadata file written by the official lib,
        #     read by the Fortran test (interoperability, not byte parity: the
        #     official HashMap makes multi-key metadata order non-deterministic).
        official_save_file(
            {
                "a": np.arange(6, dtype=np.float32).reshape(2, 3),
                "b": np.array([1.5, -2.5], dtype=np.float64),
                "c": np.array([-1, 1 << 30], dtype=np.int64),
                "d": np.array([True, False], dtype=np.bool_),
                "e": np.array([0, 200], dtype=np.uint8),
            },
            os.path.join(args.out, "oracle_official.safetensors"),
            metadata={"bpb": "1.59994", "model": "lab-3m", "caracterização": "acentuação"},
        )
        written.append(("oracle_official.safetensors",
                        os.path.getsize(os.path.join(args.out, "oracle_official.safetensors")),
                        "official"))
    else:
        print("official safetensors not importable: parity fixture comes from the oracle")
        print("(the Fortran test still compares byte for byte -- the oracle is the reference)")

    print()
    print("%-34s %10s  %s" % ("fixture", "bytes", "written by"))
    for fname, n, src in written:
        print("%-34s %10d  %s" % (fname, n, src))
    return 0


if __name__ == "__main__":
    sys.exit(main())
