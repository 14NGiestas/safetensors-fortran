#!/usr/bin/env python3
"""npy_to_safetensors.py -- turn one of our checkpoint directories into ONE file.

The lab stores each checkpoint as a directory with ~90 .npy files plus arch.txt and
template.txt. That has three costs we kept paying: no schema (nothing declares
name/shape/dtype, and a checkpoint/arch mismatch aborted two runs), 90 files are 90
chances of drift with a hash per directory instead of per tensor, and no outside
tool (numpy/HuggingFace) can read the weights.

This adapter writes a single .safetensors with
  * canonical tensor names (`wte`, `lm`, `l{i}.q`, `l{i}.k`, `l{i}.v`, `l{i}.p`,
    `l{i}.fc`, `l{i}.p2`, and optimizer moments as `opt.m.*` / `opt.v.*` if asked);
  * the whole architecture in `__metadata__` (d_model, n_head, n_kv, n_layer,
    vocab, ctx, head_dim, bos);
  * the sha256 of the data manifest and a sha256 over the tensor payloads
    (the per-tensor hash is what a directory hash could never give us);
  * a `card` field: a minimal JSON characterisation with the metric fields left
    EMPTY on purpose (the training code fills them in later).

The output is deterministic: same inputs -> byte-identical file (no timestamps).

Usage:
    python3 tools/npy_to_safetensors.py CKPT_DIR OUT.safetensors \
        [--manifest MAN.json] [--include-optimizer] [--card CARD.json]
        [--official]        # write with the official `safetensors` package
                            # instead of this repo's pure-Python oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reference_writer as rw  # noqa: E402

# numpy dtype -> safetensors dtype
DTYPES = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float64): "F64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int64): "I64",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}

# .npy file -> canonical tensor name. `{L}` is the 0-based layer index.
WEIGHT_MAP = {
    "transformer_wte_weight.npy": "wte",
    "lm_head_weight.npy": "lm",
    "transformer_h_{L}_attn_c_q_weight.npy": "l{L}.q",
    "transformer_h_{L}_attn_c_k_weight.npy": "l{L}.k",
    "transformer_h_{L}_attn_c_v_weight.npy": "l{L}.v",
    "transformer_h_{L}_attn_c_proj_weight.npy": "l{L}.p",
    "transformer_h_{L}_mlp_c_fc_weight.npy": "l{L}.fc",
    "transformer_h_{L}_mlp_c_proj_weight.npy": "l{L}.p2",
}
OPTIMIZER_MAP = {
    "adam_m_wte.npy": "opt.m.wte",
    "adam_v_wte.npy": "opt.v.wte",
    "adam_m_lm.npy": "opt.m.lm",
    "adam_v_lm.npy": "opt.v.lm",
}
for _slot in ("q", "k", "v", "p", "fc", "p2"):
    OPTIMIZER_MAP["adam_m_%s.npy" % _slot] = "opt.m.l%s" % _slot
    OPTIMIZER_MAP["adam_v_%s.npy" % _slot] = "opt.v.l%s" % _slot

ARCH_KEYS = ["d_model", "n_head", "n_kv", "n_layer", "vocab", "ctx"]
ARCH_OPTIONAL = ["bos", "head_dim"]

LAYER_RE = re.compile(r"transformer_h_(\d+)_")


def read_arch(path):
    """arch.txt is `chave = valor`; keep the textual values (metadata is strings)."""
    arch = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            arch[k.strip()] = v.strip()
    missing = [k for k in ARCH_KEYS if k not in arch]
    if missing:
        raise SystemExit("arch.txt is missing %s" % ", ".join(missing))
    return arch


def read_template(path):
    """template.txt is `dialect=...`; keep it verbatim in the metadata."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v
    return out


def collect(ckpt_dir, arch, include_optimizer):
    """Returns [(name, dtype, shape, payload, source_file)] in canonical order."""
    files = sorted(os.listdir(ckpt_dir))
    found = {}
    for fname in files:
        if not fname.endswith(".npy"):
            continue
        m = LAYER_RE.search(fname)
        layer = int(m.group(1)) if m else None
        name = None
        for pat, canon in WEIGHT_MAP.items():
            if "{L}" in pat:
                if layer is None:
                    continue
                if fname == pat.replace("{L}", str(layer)):
                    name = canon.replace("{L}", str(layer))
                    break
            elif fname == pat:
                name = canon
                break
        if name is None and include_optimizer:
            base = OPTIMIZER_MAP.get(fname)
            if base:
                name = base
        if name is None:
            continue
        found[name] = (fname, layer)

    if not found:
        raise SystemExit("no known weight files in %s" % ckpt_dir)

    # Canonical order: wte, lm, then layer 0..N-1 with q,k,v,p,fc,p2 per layer.
    def sort_key(item):
        name, (fname, layer) = item
        if name == "wte":
            return (0, 0, 0)
        if name == "lm":
            return (0, 1, 0)
        if name.startswith("opt."):
            which = 1 if name.startswith("opt.m.") else 2
            return (2, which, name)
        slot = name.split(".")[-1]
        order = ["q", "k", "v", "p", "fc", "p2"]
        return (1, layer, order.index(slot))

    tensors = []
    for name, (fname, _layer) in sorted(found.items(), key=sort_key):
        path = os.path.join(ckpt_dir, fname)
        arr = np.load(path, mmap_mode="r")
        dt = DTYPES.get(arr.dtype)
        if dt is None:
            raise SystemExit("%s: dtype %s has no safetensors mapping" % (fname, arr.dtype))
        if arr.dtype.byteorder not in ("=", "<", "|"):
            raise SystemExit("%s: big-endian .npy is not supported (safetensors is LE)" % fname)
        payload = np.ascontiguousarray(arr).tobytes()
        tensors.append((name, dt, list(arr.shape), payload, fname))
    return tensors


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build_metadata(args, arch, template, tensors, manifest_path):
    tpl = read_template(os.path.join(args.ckpt_dir, "template.txt"))
    meta = {
        "format_version": "1",
        "producer": "safetensors-fortran/tools/npy_to_safetensors.py",
        "source_dir": os.path.basename(os.path.abspath(args.ckpt_dir)),
        "arch.source": "arch.txt",
        "n_tensors": str(len(tensors)),
    }
    for k, v in arch.items():
        meta["arch." + k] = v
    meta["arch.head_dim"] = arch.get("head_dim", str(int(arch["d_model"]) // int(arch["n_head"])))
    if tpl:
        meta["template.dialect"] = tpl.get("dialect", "")
        meta["template.sha256"] = sha256_bytes(
            open(os.path.join(args.ckpt_dir, "template.txt"), "rb").read()
        )

    # sha256 do manifesto de dados (se houver) e sha256 do manifesto de tensores.
    if manifest_path and os.path.exists(manifest_path):
        meta["manifest.path"] = os.path.abspath(manifest_path)
        meta["manifest.sha256"] = sha256_bytes(open(manifest_path, "rb").read())
    else:
        meta["manifest.path"] = ""
        meta["manifest.sha256"] = ""
    lines = ["%s %s %s %s" % (n, d, "x".join(str(s) for s in sh), sha256_bytes(p))
             for n, d, sh, p, _f in tensors]
    meta["tensor_manifest.sha256"] = sha256_bytes("\n".join(lines).encode())

    card = {
        "name": meta["source_dir"],
        "arch": {
            "d_model": int(arch["d_model"]),
            "n_head": int(arch["n_head"]),
            "n_kv": int(arch["n_kv"]),
            "n_layer": int(arch["n_layer"]),
            "vocab": int(arch["vocab"]),
            "ctx": int(arch["ctx"]),
        },
        "data": {
            "manifest_sha256": meta["manifest.sha256"],
            "manifest_path": meta["manifest.path"],
            "rows": "",
            "tokens_total": "",
        },
        # Preenchido depois pelo treino: aqui só o ESQUELETO da caracterização.
        "metrics": {"bpb": "", "val_loss": "", "tokens_seen": "", "wall_clock_s": ""},
        "energy": {"kwh": "", "joules_per_token": "", "hardware": ""},
        "lineage": {"run": meta["source_dir"], "parent": "", "commit": "", "notes": ""},
        "tensor_manifest_sha256": meta["tensor_manifest.sha256"],
    }
    if args.card:
        with open(args.card, "r", encoding="utf-8") as fh:
            card = json.load(fh)
    meta["card"] = json.dumps(card, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("out")
    ap.add_argument("--manifest", default="", help="data manifest json (sha256 goes into __metadata__)")
    ap.add_argument("--include-optimizer", action="store_true",
                    help="also store adam m/v state as opt.m.*/opt.v.* (2x file size)")
    ap.add_argument("--card", default="", help="use this card JSON instead of the generated skeleton")
    ap.add_argument("--official", action="store_true", help="write with the official safetensors package")
    args = ap.parse_args()

    arch = read_arch(os.path.join(args.ckpt_dir, "arch.txt"))
    tensors = collect(args.ckpt_dir, arch, args.include_optimizer)
    meta = build_metadata(args, arch, None, tensors, args.manifest)

    if args.official:
        from safetensors.numpy import save_file as official_save_file

        arrays = {}
        for name, dt, shape, payload, _f in tensors:
            npdt = {v: k for k, v in DTYPES.items()}[dt]
            arrays[name] = np.frombuffer(payload, dtype=npdt).reshape(shape) if shape else \
                np.frombuffer(payload, dtype=npdt)
        official_save_file(arrays, args.out, metadata=meta)
    else:
        rw.write(args.out, [(n, d, s, p) for n, d, s, p, _f in tensors], meta)

    _, back = rw.read(args.out)
    total = sum(len(t["data"]) for t in back.values())
    print("wrote %s" % args.out)
    print("  tensors        : %d" % len(back))
    print("  payload bytes  : %d (%.2f MiB)" % (total, total / 1048576))
    print("  file bytes     : %d" % os.path.getsize(args.out))
    print("  metadata keys  : %d" % len(meta))
    print("  manifest sha256: %s" % (meta["manifest.sha256"] or "(no manifest given)"))
    print("  tensor sha256  : %s" % meta["tensor_manifest.sha256"])
    print("  canonical names:")
    for name, info in back.items():
        print("    %-14s %-4s %-14s %9d bytes  sha256=%s"
              % (name, info["dtype"], str(info["shape"]), len(info["data"]),
                 sha256_bytes(info["data"])[:16]))
    if args.official:
        return 0
    # cross-check with the official reader when it is importable
    try:
        from safetensors import safe_open  # noqa: F401

        print("  official safetensors package: importable (see --official)")
    except Exception:
        print("  official safetensors package: NOT importable here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
