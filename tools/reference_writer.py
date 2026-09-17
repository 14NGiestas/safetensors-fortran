#!/usr/bin/env python3
"""reference_writer.py -- pure-Python oracle for the safetensors format.

Why a second implementation in a language nobody asked for:
byte-for-byte parity between two independent writers is the strongest cheap
evidence that the Fortran library implements the spec and not just some
self-consistent variant of it. This oracle is written from the spec text
(little-endian length, space-padded JSON header, contiguous buffer) with
`struct` + `json` only, so it shares no code with the Fortran side.

It also *reads*, applying the same validation the Fortran reader applies
(overlaps, holes, size mismatches, unknown dtypes, ...), which is what lets the
test suite do "Fortran writes -> Python reads" and "Python writes -> Fortran
reads" without the official Rust implementation being installed.

Canonical layout produced here (identical to the Fortran writer):
  * `__metadata__` is always present (possibly `{}`);
  * metadata keys are sorted byte-wise;
  * tensors are written in insertion order with increasing offsets;
  * the header is padded with 0x20 so that `8 + N` is a multiple of 8;
  * JSON is emitted like serde_json: no spaces, `\\u00xx` (lowercase) for control
    characters, `\\b \\t \\n \\f \\r` for those five, non-ASCII bytes passed raw.

Interoperability check with the official implementation lives in
tools/make_fixtures.py (it compares the bytes produced here against the bytes
produced by `safetensors` when that package is installed).
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys

# name -> bits per element, exactly the table of the official Dtype enum
DTYPE_BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def json_escape(s: str) -> str:
    """Escape exactly like serde_json (what the official writer uses)."""
    out = []
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)  # >= 0x20 (including DEL and non-ASCII) passes raw
    return "".join(out)


def dtype_itemsize(dtype: str) -> int:
    bits = DTYPE_BITS.get(dtype)
    if bits is None:
        raise ValueError("unknown dtype %r" % dtype)
    if bits % 8:
        raise ValueError("sub-byte dtype %r not supported by this oracle" % dtype)
    return bits // 8


def build(tensors, metadata=None):
    """tensors: list of (name, dtype, shape, payload bytes) in insertion order."""
    meta = dict(metadata or {})
    header = {"__metadata__": {}}
    entries = []
    offset = 0
    for name, dtype, shape, payload in tensors:
        itemsize = dtype_itemsize(dtype)
        nelem = 1
        for d in shape:
            if d < 0:
                raise ValueError("negative extent in %r" % (shape,))
            nelem *= d
        expected = nelem * itemsize
        if len(payload) != expected:
            raise ValueError(
                "tensor %r: payload is %d bytes but shape %r with %s needs %d"
                % (name, len(payload), shape, dtype, expected)
            )
        entries.append((name, dtype, list(shape), [offset, offset + len(payload)]))
        offset += len(payload)
    header["__metadata__"] = {k: meta[k] for k in sorted(meta, key=lambda s: s.encode())}
    for name, dtype, shape, offs in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": offs}

    body = _emit(header)
    raw = body.encode("utf-8")
    pad = (-len(raw)) % 8
    raw += b" " * pad
    out = struct.pack("<Q", len(raw)) + raw
    for _, _, _, payload in tensors:
        out += payload
    return out


def _emit(header):
    """Deterministic JSON emission (dict order preserved, serde_json-style escapes)."""
    parts = []
    meta = header["__metadata__"]
    parts.append('{"__metadata__":{')
    parts.append(
        ",".join('"%s":"%s"' % (json_escape(k), json_escape(v)) for k, v in meta.items())
    )
    parts.append("}")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        parts.append(',"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}' % (
            json_escape(name),
            info["dtype"],
            ",".join(str(d) for d in info["shape"]),
            info["data_offsets"][0],
            info["data_offsets"][1],
        ))
    parts.append("}")
    return "".join(parts)


def write(path, tensors, metadata=None):
    data = build(tensors, metadata)
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


# --------------------------------------------------------------------- reading


class SafeTensorError(Exception):
    pass


def read(path):
    """Minimal validating reader: returns (metadata, tensors).

    tensors is a dict name -> {"dtype","shape","data_offsets","data"} with the
    raw payload bytes. Raises SafeTensorError with a human message, mirroring the
    rules the Fortran reader enforces.
    """
    blob = open(path, "rb").read()
    if len(blob) < 8:
        raise SafeTensorError("file smaller than the 8-byte header length prefix")
    (n,) = struct.unpack("<Q", blob[:8])
    if n == 0:
        raise SafeTensorError("header length is 0")
    if n > 100_000_000:
        raise SafeTensorError("header length %d above the 100MB format limit" % n)
    if 8 + n > len(blob):
        raise SafeTensorError(
            "truncated: header declares %d bytes but only %d follow" % (n, len(blob) - 8)
        )
    header = blob[8 : 8 + n]
    if not header.startswith(b"{"):
        raise SafeTensorError("header does not start with '{'")
    try:
        parsed = json.loads(header.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        raise SafeTensorError("invalid header JSON: %s" % exc) from exc
    if not isinstance(parsed, dict):
        raise SafeTensorError("header is not a JSON object")

    metadata = parsed.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise SafeTensorError("__metadata__ must be an object")
    for k, v in metadata.items():
        if not isinstance(v, str):
            raise SafeTensorError("__metadata__[%r] must be a string" % k)

    buf = blob[8 + n :]
    tensors = {}
    order = []
    for name, info in parsed.items():
        if not isinstance(info, dict):
            raise SafeTensorError("entry %r is not an object" % name)
        dtype = info.get("dtype")
        if dtype not in DTYPE_BITS:
            raise SafeTensorError("tensor %r: unknown dtype %r" % (name, dtype))
        shape = info.get("shape")
        if not isinstance(shape, list) or not shape:
            raise SafeTensorError("tensor %r: shape must be a non-empty array" % name)
        if any((not isinstance(d, int)) or d < 0 for d in shape):
            raise SafeTensorError("tensor %r: shape must hold non-negative integers" % name)
        offs = info.get("data_offsets")
        if not isinstance(offs, list) or len(offs) != 2:
            raise SafeTensorError("tensor %r: data_offsets must have 2 integers" % name)
        begin, end = offs
        if begin < 0 or end < begin:
            raise SafeTensorError("tensor %r: bad data_offsets %r" % (name, offs))
        nelem = 1
        for d in shape:
            nelem *= d
        bits = nelem * DTYPE_BITS[dtype]
        if bits % 8:
            raise SafeTensorError("tensor %r: sub-byte dtype not byte aligned" % name)
        if end - begin != bits // 8:
            raise SafeTensorError(
                "tensor %r: data_offsets span %d bytes but shape %r with %s needs %d"
                % (name, end - begin, shape, dtype, bits // 8)
            )
        tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, end],
            "data": buf[begin:end],
        }
        order.append(name)

    cursor = 0
    for name in sorted(order, key=lambda k: tensors[k]["data_offsets"][0]):
        begin, end = tensors[name]["data_offsets"]
        if begin != cursor:
            raise SafeTensorError(
                "tensor %r starts at %d but previous ends at %d (holes/overlaps not allowed)"
                % (name, begin, cursor)
            )
        cursor = end
    if cursor != len(buf):
        raise SafeTensorError(
            "tensors cover %d bytes but the file has %d after the header" % (cursor, len(buf))
        )
    return metadata, tensors


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, path = argv[1], argv[2]
    if cmd in ("--verify", "--dump"):
        metadata, tensors = read(path)
        print("file           : %s" % path)
        print("header+payload : %d bytes (sha256 %s)" % (
            len(open(path, "rb").read()), sha256_file(path)[:16]))
        print("metadata       : %d keys" % len(metadata))
        for k in sorted(metadata):
            print("  meta[%s] = %r" % (k, metadata[k]))
        print("tensors        : %d" % len(tensors))
        for name, info in tensors.items():
            print(
                "  %-28s %-6s %-16s %8d bytes  fnv1a64=%016x"
                % (
                    name,
                    info["dtype"],
                    str(info["shape"]),
                    len(info["data"]),
                    fnv1a64(info["data"]),
                )
            )
        if cmd == "--verify":
            print("VERIFY OK")
        return 0
    print("unknown command %r" % cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
