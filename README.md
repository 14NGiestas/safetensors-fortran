# safetensors (Fortran)

[![CI](https://github.com/14NGiestas/safetensors-fortran/actions/workflows/ci.yml/badge.svg)](https://github.com/14NGiestas/safetensors-fortran/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A dependency-free Fortran library for the [safetensors](https://github.com/huggingface/safetensors)
format: a writer, a reader, and header inspection for tensor checkpoints.

```fortran
type(st_writer) :: w
call w%init()
call w%set_meta('format_version', '1')
call w%set_meta('bpb', '1.59994')
call w%set('wte', wte)              ! dtype and rank come from the argument
call w%set('l3.q', q)
call w%write('model.safetensors', stat, msg)
```

## Why this exists

The format is a JSON header (name, dtype, shape, byte offsets, plus a free-form
`__metadata__` string map) followed by one contiguous byte buffer. That fixes a
class of problems that a checkpoint-as-a-directory-of-`.npy`-files cannot:

* **there is a schema** — name/dtype/shape travel with the data, so an
  architecture/checkpoint mismatch is caught instead of producing silent garbage;
* **one file, one hash per tensor** — instead of dozens of files with a single
  directory-level hash;
* **everything else can read it** — numpy, PyTorch, HuggingFace `transformers`,
  `candle`, `llama.cpp`, ... with no pickle and no arbitrary code execution;
* **provenance travels with the weights** — loss, lineage or an energy budget can
  live in `__metadata__`, next to the bytes they describe.

**Who it is for:** Fortran numerical code that has to hand tensors to (or receive
tensors from) the Python ecosystem; people who want one file per checkpoint with a
real schema; anyone who needs to *inspect* a checkpoint header (shapes, dtypes,
offsets, metadata) without loading gigabytes.

## Conformance to the format

The authoritative text used is the **"Format"** section of the official README
(the `docs/spec.md` file that used to be referenced no longer exists on `main`;
see `docs/spec_notes.md` for URLs, commit and per-item notes).

| Spec item | Status |
|---|---|
| 8-byte little-endian header length | **met** (written/read explicitly, not via `transfer`) |
| Header is UTF-8 JSON, starts with `{`, space-padded to a multiple of 8 | **met** (space padding and JSON escaping match the official writer byte-for-byte in the parity fixtures; see below) |
| `data_offsets` relative to the buffer, `begin <= end` | **met** |
| `end - begin == prod(shape) * itemsize` | **met** (validated with clear errors) |
| Offsets non-overlapping, buffer fully covered, increasing | **met** (sorted + contiguity check; rejects holes and trailing bytes) |
| `__metadata__` is string → string | **met** (non-string values rejected) |
| Little-endian on disk | **met**; big-endian hosts get an explicit error, no byteswap |
| NaN / ±Inf / −0.0 preserved | **met** (payload bytes are copied verbatim; tested bit by bit) |
| Empty tensors (`shape: [0]`) | **met** (0-byte payload, collapsed offsets) |
| Rank-0 tensors (`shape: []`) | **not supported** — rejected on both write and read (a scalar in a weight file is a bug, and refusing it keeps the API unambiguous) |
| Duplicate header keys | **met** — rejected (hash-set check, no O(n²) DoS) |
| Header ≤ 100 MB | **met** (`100000000`, same constant as the reference) |
| Dtypes: `F32 F64 I32 I64 U8 BOOL` | **met** — typed getters, round-trip tested |
| Dtypes: `I8 I16 U16 U32 U64 C64 F16 BF16 F4 F6_* F8_*` | **partial** — recognised, size-validated, inspectable and readable as raw bytes (`get_raw`), but no typed getter |
| Sub-byte dtypes (`F4`, `F6_*`) byte alignment rule | **met** (rejected unless the element count is a whole number of bytes) |
| Row-major ('C') layout, no strides | **met** (bytes are stored in memory order; see the rank ≥ 2 note) |
| Tensor order in the buffer | **partial** — this writer uses **insertion order** with increasing offsets; the reference writer sorts by descending dtype alignment then name. Both are valid files: the spec does not mandate an order, so byte-identity holds only when the same order is used (the parity fixtures pass because their insertion order coincides with the sorted one). On a real 74-tensor checkpoint the two writers produce different files — headers of 6624 vs 6568 bytes — that are semantically identical and load in either implementation |
| `__metadata__` omitted when there is no metadata | **partial** — we always emit `__metadata__` (empty object if you set nothing), so that consumers can always rely on the slot existing |

Byte-for-byte parity with the reference implementation is checked by the test
suite: `build/st_test/parity_lib.safetensors` (written by this library) and
`test/fixtures/parity_oracle.safetensors` (written by the official Rust/Python
implementation) have the same SHA-256, and so do the two escaping fixtures:

```
71ea7846eb4f3678cc5a9ef542f4f8dcbcf3f746cd49dafb8ec0673ee4b96ec0  parity
a741175e7362c15087c70f0c1a5682b7b5f41896ff373dbf9c717cdfcfe1e845  metadata escaping
```

## Requirements

* A Fortran 2008+ compiler: **gfortran ≥ 14** (developed on 15.3; 13 hits a compiler bug, see Limitations)
  and Fortran Package Manager (fpm ≥ 0.9).
* Nothing else. No BLAS, no `stdlib`, no system libraries beyond libc.

## Build, test, use

```bash
fpm build                 # library + examples
fpm test                  # the suite; prints "PASS n / FAIL m" and exits non-zero on failure
fpm test --profile debug --flag "-Wall -Wextra -fcheck=all -fbacktrace -finit-real=snan"
fpm run --example write_read
fpm run --example inspect -- model.safetensors --hash
```

Use it as a dependency:

```toml
[dependencies]
safetensors = { git = "https://github.com/14NGiestas/safetensors-fortran" }
```

## API

Everything returns `stat` + `msg`; the library **never** calls `stop` or
`error stop`. `st_ok` is 0; the error codes are exported (`st_err_io`,
`st_err_truncated`, `st_err_header_size`, `st_err_json`, `st_err_schema`,
`st_err_dtype`, `st_err_offsets`, `st_err_missing`, `st_err_type_mismatch`,
`st_err_duplicate`, `st_err_not_open`, `st_err_range`, `st_err_value`,
`st_err_endian`).

### Writing

```fortran
use safetensors, only: st_writer, st_ok

type(st_writer) :: w
real(real32) :: wte(4, 8), q(12)

call w%init()
call w%set_meta('format_version', '1')      ! metadata keys are kept sorted (byte-wise)
call w%set_meta('bpb', '1.59994')
call w%set_meta_int('n_layer', 12_int64)
call w%set('wte', wte)                      ! real32/real64/int32/int64/int8/logical, rank 1 or 2
call w%set('l3.q', q)
call w%write('model.safetensors', stat, msg)
if (stat /= st_ok) print *, 'write failed: ', msg
```

* `set` and `set_meta` take **optional** `stat`/`msg`. If you omit them and
  something is wrong (unsupported type, duplicate name, rank 0), the first error is
  remembered and `write` refuses to produce a file: `call w%error(stat, msg)`
  retrieves it. Nothing fails silently.
* The payload is copied into an internal buffer, so you can deallocate your arrays
  right after `set`. Peak memory = payload + one copy.
* `w%to_bytes(image, stat, msg)` gives the whole file as an `integer(int8)` array
  (handy for sockets/tests); `w%header_text(text, aligned_len, stat, msg)` gives
  just the JSON header.
* Tensor order in the file is insertion order (documented, stable, offsets always
  increasing).

### Reading

```fortran
use safetensors, only: st_reader, st_ok
use iso_fortran_env, only: int64, real32

type(st_reader) :: r
real(real32), pointer :: p2(:, :) => null()
integer(int64), allocatable :: shp(:)
character(len=:), allocatable :: msg, dtype, val
logical :: found

call r%open('model.safetensors', stat, msg)      ! parses and validates the header only
call r%shape('wte', shp, stat, msg)              ! C order, e.g. [4, 8]
call r%dtype('wte', dtype, stat, msg)            ! 'F32' — no bytes are read yet
call r%get('wte', p2, stat, msg)                 ! rank-2 pointer, allocated by the library
call r%meta('bpb', val, found)                   ! metadata is strings
call r%close()
```

* `open` reads **only the header** and validates everything (JSON, dtypes, offsets,
  coverage, duplicates): inspecting a 30 GB checkpoint costs the header.
* `get` is a generic resolved by the pointer's declared type and rank:
  `real(real32)`, `real(real64)`, `integer(int32)`, `integer(int64)`,
  `integer(int8)` (`U8`) and `logical` (`BOOL`). Asking for the wrong dtype is a
  clean `st_err_type_mismatch` that names both sides.
* The returned pointer is allocated by the library and is the **final storage**:
  one positional `read` writes the tensor straight into it (via a `c_f_pointer`
  alias), with no intermediate byte buffer. You own the memory: `deallocate(p)`
  when done, and pass a fresh pointer to each `get`.
* `r%get_raw(name, bytes, stat, msg)` returns the raw `integer(int8)` payload for
  **any** dtype — that is the escape hatch for `F16`, `BF16`, `F8_*`, `I8`, ...
* Other queries: `n_tensors`, `tensor_name(i)`, `rank`, `nbytes`, `tensor_offsets`,
  `has`, `n_meta`, `meta_key(i)`, `file_size`, `header_size`, `buffer_size`.

### Rank ≥ 2 and C order (read this before storing matrices)

Fortran is column-major, the format is row-major. This library never transposes
and never copies to hide that: it keeps the **memory order** and reports the
**extents reversed** on read. Writing a Fortran `a(m, n)` produces a header with
`shape: [m, n]` and the bytes exactly as they sit in `a`; reading it back with the
2-D getter gives `p(n, m)` with `p = reshape(a, [n, m])` — the same bytes, so

```fortran
all(transfer(p, [-1.0], size(p)) == transfer(a, [-1.0], size(a)))   ! .true.
```

because `transfer` flattens in memory order. In Python the same tensor reads as
`numpy_array.T` when compared with the Fortran array's logical values.

Practical advice: **store flat rank-1 arrays** when interop matters (that is what
the adapter in `tools/` does with our own checkpoints), or use the flat getters and
reshape yourself. If you must move a matrix across, use `transpose` explicitly —
being surprised by a silent transpose is worse than typing it.

## Interoperability with Python

Writing with Fortran and reading with the official library:

```python
import numpy as np
from safetensors import safe_open

with safe_open("model.safetensors", framework="numpy") as f:
    print(len(list(f.keys())), "tensors")
    print(f.metadata()["format_version"])                  # '1'
    wte = f.get_tensor("wte")
    print(wte.shape, wte.dtype)
```

```python
# and the other direction: a file written by Python, read by Fortran
from safetensors.numpy import save_file
import numpy as np

save_file({"x": np.arange(6, dtype=np.float32).reshape(2, 3)},
          "from_python.safetensors", metadata={"origin": "python"})
```

```fortran
call r%open('from_python.safetensors', stat, msg)
call r%get('x', x, stat, msg)      ! real(real32), pointer, rank-2: extents (3, 2)
call r%meta('origin', val, found)  ! 'python'
```

Both directions and both sides of the metadata escaping are covered by `fpm test`
itself: the suite shells out to `tools/reference_writer.py`, and one of the
committed fixtures was written by the official package. On a real 74-tensor
transformer checkpoint (11 MB) the official implementation read back every tensor
and each payload hash equalled that of the source array.

## Limitations (honest list)

* **Little-endian only.** A big-endian host is detected at runtime and every
  operation fails with `st_err_endian` instead of writing byte-swapped data. No
  byteswap support.
* **Rank ≥ 2 semantics** — see the section above. Memory order is preserved,
  extents are reversed; the format's row-major convention is not "fixed up"
  for you.
* **Rank-0 tensors are rejected** (write and read). Rank > 8 in a header is
  rejected. `get_*_2d` refuses rank > 2 (the flat getter still works).
* **Typed getters only for F32/F64/I32/I64/U8/BOOL.** Everything else (F16, BF16,
  FP8 variants, I8/I16/U16/U32/U64/C64, sub-byte F4/F6) can be inspected
  (`dtype`, `shape`, `nbytes`) and read raw with `get_raw`, but you have to
  interpret the bytes yourself.
* **No mmap.** Tensors are read with positional stream I/O into freshly allocated
  memory. Per-tensor lazy loading is supported (that is what `get` is); *partial*
  loading of a single tensor is not.
* **Optional `stat`/`msg`** exist for `set`/`set_meta`; a writer with a pending
  error refuses to write (so the convenience cannot hide a failure).
* **Duplicated `set` names are an error**, and duplicate `set_meta` overwrites.
* **The writer copies** the payload once (see above); only one copy, not two.
* **`get_bool` converts** to the default `logical` kind (4 bytes in gfortran) and
  rejects bytes other than 0/1, matching the reference implementation.
* **gfortran 13 does not compile this library** — a compiler ICE in
  `gfc_conv_procedure_call`. gfortran 14 and 15 build and pass the suite; treat
  **gfortran ≥ 14** as the requirement.
* **Not verified:** macOS (the CI job fails during the build), files > 4 GB,
  big-endian hosts.
* `fpm test` also inherits fpm's working directory assumption; the suite looks for
  fixtures in `test/fixtures`, `../test/fixtures` and `fixtures`, and writes its
  scratch output to `build/st_test/`.

## Repository layout

```
src/safetensors.f90        writer + reader + validation (public module `safetensors`)
src/safetensors_json.f90   minimal JSON parser/emitter (flat-arena DOM, no deps)
test/test_safetensors.f90  the suite: 106 checks, PASS/FAIL counter, non-zero exit
test/fixtures/             reference files (official package and/or pure-Python oracle)
test/parity_cmp.sh         runs the suite and shows the cmp/sha256 parity evidence
examples/write_read.f90    end-to-end write/read with error handling
examples/inspect.f90       header dump + per-tensor FNV-1a 64 of the raw payload
tools/reference_writer.py  pure-Python writer/reader oracle (struct + json only)
tools/make_fixtures.py     generates test/fixtures, cross-checking the official writer
tools/npy_to_safetensors.py  directory of .npy tensors -> one .safetensors
tools/verify_npy_vs_st.py  proves the .safetensors bytes equal the .npy bytes
docs/spec_notes.md         URLs, commit and per-item conformance notes
```

## Helpers in `tools/`

Generic utilities, useful beyond this library:

* `npy_to_safetensors.py` — packs a directory of `.npy` tensors into one
  `.safetensors`, with a name mapping you can edit (the built-in mapping targets
  one particular transformer checkpoint layout) and optional JSON metadata.
* `verify_npy_vs_st.py` — proves the payloads in a `.safetensors` equal the bytes
  of the original `.npy` files.
* `reference_writer.py` — a ~40-line pure-Python writer/reader used as an
  independent oracle by the test suite.
* `make_fixtures.py` — regenerates `test/fixtures/`, cross-checking the official
  package when it is installed.

## License

MIT — see [LICENSE](LICENSE).
