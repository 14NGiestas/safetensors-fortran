# Format notes: what the spec says, where it was read, what was verified

Read on 2026-09-17 against `huggingface/safetensors` at commit
`0accbb8ed807590a0116c70c02907b505e69a1ff` (`main`), fetched over HTTP. Every
format rule this library implements is listed with its source and with the test
or output that confirms it.

## Sources

| # | URL | What was taken from it |
|---|-----|------------------------|
| S1 | <https://github.com/huggingface/safetensors/blob/main/README.md> (section "Format") | file layout: 8-byte LE size, UTF-8 JSON header starting with `{`, whitespace padding, `data_offsets` relative to the buffer, `__metadata__` as string→string, buffer without holes, C/row-major order, little-endian, empty and rank-0 tensors allowed, duplicate keys forbidden, 100 MB header limit, NaN/±Inf not checked |
| S2 | <https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs> | `const MAX_HEADER_SIZE: usize = 100_000_000;` (line 10); the `Dtype` enum with its alignment order (line ~811); `bitsize()`; `Metadata::validate()` (`s != start \|\| e < s`, i.e. contiguous offsets from 0), `nbits = nelements * bitsize`, `nbits % 8 != 0` ⇒ error; `prepare()`/`serialize()` for the padding (`next_multiple_of(N_LEN)` filled with `b' '`) and the ordering `right.dtype().cmp(&left.dtype()).then(lname.cmp(rname))` (decreasing alignment, then name) |
| S3 | <https://huggingface.co/docs/safetensors/index> | overview and the `safe_open`/`save_file` examples used in the README's interoperability section |
| S4 | <https://github.com/huggingface/safetensors/blob/main/docs/safetensors.schema.json> | the published header schema (exists and is valid; not used as the source of truth in the code) |

### Note on `docs/spec.md`

That file no longer exists on `main`: the GitHub contents API on 2026-09-17 lists
only `safetensors.schema.json` and `source/` (`_toctree.yml`, `api/`,
`convert-weights.md`, `index.mdx`, `metadata_parsing.mdx`, `speed.mdx`,
`torch_shared_tensors.mdx`) under `docs/`. The normative description of the format
is today the **"Format"** section of the README (S1), which is what this library
follows. S1 was read in full; the table below is its operational translation.

## Item by item

| Spec item | Source | Implementation | Verified by |
|---|---|---|---|
| `N` = 8 bytes LE, unsigned | S1 | `le_from_i64`/`i64_from_le` (explicit, no reliance on `transfer`) | byte-parity test and `cmp` in `test/parity_cmp.sh` |
| header = `N` bytes of UTF-8 JSON starting with `{` | S1 | `rd_open` goes through the parser, which rejects non-objects | test "header that is not an object" (stat 6) |
| header padded with spaces to a multiple of 8 | S2 (`prepare`) | `wr_header_text` computes `pad = 8 - (len mod 8)` and writes `0x20` | byte-for-byte parity: different padding would fail `cmp` |
| `data_offsets` relative to the buffer, `begin < end` | S1 | positional read at `dstart + begin + 1` | tests 1, 4, 5 |
| `end - begin == prod(shape) * itemsize` | S1, S2 | explicit check in `rd_parse_header` | test "offsets inconsistent with shape" (stat 8) |
| offsets non-overlapping, buffer fully covered, increasing | S1, S2 (`validate`) | `rd_check_coverage`: sorts by `begin` (heapsort) and requires `begin_i == cursor` and `cursor_final == buffer_size` | tests "overlapping" and "hole" (stat 8) |
| `__metadata__` is a string→string map | S1 | `rd_read_metadata` rejects non-string values | test "non-string metadata value" (stat 6) |
| dtypes F32/F64/I64/I32/U8/BOOL | S1, S2 | full table of the 22 official dtypes in `dt_names`/`dt_bits`; typed getters for those six | tests 1, 5, 6 |
| little-endian always | S1 | explicit LE read/write; a big-endian host gets `st_err_endian` instead of corrupted data | `st_host_is_little_endian()` (checked at runtime) |
| NaN/Inf/−0.0 neither checked nor altered | S1 | no arithmetic on payload values: bytes are copied verbatim | test 7 (24 identical bytes, including `0x80000000`) |
| empty tensors (`[0]`) and rank-0 allowed | S1 | empty: collapsed offsets, 0 bytes; rank-0: **rejected** (see below) | test 2 (`[0]`, `[1,0]`) |
| duplicate keys forbidden | S1 | hash set (FNV-1a, open addressing) per object | test "duplicate key" (stat 5) |
| header ≤ 100 MB | S1, S2 | `st_max_header_bytes = 100000000` | test "header length over 100MB" (stat 4) |
| C / row-major order | S1 | bytes preserved in memory order; see the rank ≥ 2 note in the README | test 1 (2-D) and test 4 (numpy fixture) |
| official writer orders tensors by decreasing alignment, then name | S2 (`prepare`) | this library writes in **insertion order** (documented); the parity test inserts in the official order so the two files can be compared | parity test |
| `__metadata__` omitted when empty (official writer) | S2 (`HashMetadata`, `skip_serializing_if`) | **difference**: this library always emits `__metadata__` (possibly `{}`) so consumers can rely on the slot existing | parity test uses metadata, so the byte comparison is meaningful |

## Not implemented, and why

* **rank-0 (scalar)**: the format allows it (`shape: []`), this API does not. A
  scalar in a weight file is almost always a mistake by the writer, and rejecting
  it with a message beats inventing empty-shape semantics. Reading files that
  contain rank-0 tensors is rejected too (`st_max_rank` is at least 1).
* **sub-byte dtypes (F4, F6_*)**: present in the table (they show up in error
  messages and in the `nbits % 8` validation), but there is no typed getter; raw
  bytes are available through `get_raw`.
* **byteswap for big-endian hosts**: an explicit error instead of support.
* **official tensor ordering on write**: see the corresponding row above.
