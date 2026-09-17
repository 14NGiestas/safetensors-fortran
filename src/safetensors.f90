! safetensors.f90 — Fortran implementation of the safetensors format (writer + reader).
!
! WHY THIS EXISTS (lab context, not theory):
! the lab stores every checkpoint as a directory with ~90 .npy files plus a parallel
! arch.txt. Three problems we actually paid for: (a) there is no schema — nothing
! declares name/shape/dtype, and two runs were already aborted by checkpoint/arch
! mismatches; (b) 90 files are 90 chances of drift and the hash is per directory,
! not per tensor; (c) no outside tool (Python/HF) can read the weights.
! safetensors fixes all three: a JSON header with name/dtype/shape/offsets, one
! contiguous buffer, and the official `__metadata__` field — the model's
! characterisation (bpb, metrics, lineage, cost) travels WITH the weights.
!
! LIBRARY CONTRACTS:
!   * zero external dependencies (not even stdlib);
!   * NEVER calls stop/error stop: every failure comes back as stat + msg;
!   * little-endian on disk (that is what the spec mandates). The host is tested at
!     runtime (st_host_is_little_endian) and a big-endian host gets an explicit error
!     instead of silently writing swapped bytes. There is no byteswap: this is a
!     declared limitation, not a hidden assumption;
!   * offsets are int64 and the payload is assembled byte by byte, with no arithmetic
!     on the values: NaN/Inf/-0.0 survive the round-trip bit for bit (compare bytes,
!     never `==` on floats — NaN != NaN);
!   * tensor order when writing: INSERTION ORDER (documented), with strictly
!     increasing offsets; `__metadata__` is always present, with keys in byte-wise
!     lexicographic order (stable, for diffing and reproducibility).
!
! DECLARED LIMITATIONS (see the README, "Honest limitations"):
!   * a big-endian host is not supported (it errors out, it does not corrupt);
!   * typed `get` exists for F32/F64/I32/I64/U8/BOOL; other dtypes (F16, BF16, FP8,
!     I8, U16...) can be read RAW with `get_raw` and inspected through
!     shape/dtype/nbytes;
!   * header shapes up to rank 8; `get_*_2d` only accepts rank <= 2;
!   * the writer COPIES the payload into an internal buffer (the caller may free
!     the arrays right after `set`); the reader does NOT copy: it reads the file
!     straight into the memory of the array it returns (one read, no intermediate
!     buffer);
!   * no mmap (the file is read with plain positional I/O);
!   * a writer with no `set_meta` emits `__metadata__:{}`; the official writer
!     OMITS the key when there is no metadata. The only known byte difference.
module safetensors
  use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only: c_f_pointer, c_loc, c_ptr
  use safetensors_json, only: json_writer, json_doc, json_parse, json_ok, &
                              json_object, json_array, json_string, json_number, &
                              json_bool, sort_indices
  implicit none
  private

  character(*), parameter, public :: st_version = '0.1.1'
  ! Header ceiling, same as the official implementation (safetensors README:
  ! "a limit on the size of the header of 100MB"): it stops a giant header from
  ! killing memory before any validation can run.
  integer(int64), parameter, public :: st_max_header_bytes = 100000000_int64
  ! Maximum rank accepted in a header. The spec imposes no limit; 8 is slack over
  ! any real tensor and keeps the shape record a fixed size.
  integer, parameter, public :: st_max_rank = 8
  ! Dtypes this version can DESCRIBE (same table as the official Dtype enum).
  integer, parameter, public :: st_ndtypes = 22

  ! -------------------------------------------------------------- error codes
  integer, parameter, public :: st_ok = 0
  integer, parameter, public :: st_err_io = 1            ! OS open/read/write
  integer, parameter, public :: st_err_not_found = 2     ! no such file
  integer, parameter, public :: st_err_truncated = 3     ! file smaller than the header
  integer, parameter, public :: st_err_header_size = 4   ! empty/absurd header
  integer, parameter, public :: st_err_json = 5          ! malformed JSON
  integer, parameter, public :: st_err_schema = 6        ! valid JSON, wrong header
  integer, parameter, public :: st_err_dtype = 7         ! unknown/unsupported dtype
  integer, parameter, public :: st_err_offsets = 8       ! overlap/hole/size mismatch
  integer, parameter, public :: st_err_missing = 9       ! missing tensor
  integer, parameter, public :: st_err_type_mismatch = 10! tensor dtype != requested type
  integer, parameter, public :: st_err_duplicate = 11    ! repeated name
  integer, parameter, public :: st_err_not_open = 12     ! reader not open
  integer, parameter, public :: st_err_range = 13        ! rank/item outside the supported range
  integer, parameter, public :: st_err_value = 14        ! invalid payload value (bool)
  integer, parameter, public :: st_err_endian = 15       ! big-endian host

  ! ------------------------------------------------------------------ data types
  integer, parameter :: dt_bool = 1, dt_f4 = 2, dt_f6_e2m3 = 3, dt_f6_e3m2 = 4, &
                        dt_u8 = 5, dt_i8 = 6, dt_f8_e5m2 = 7, dt_f8_e4m3 = 8, &
                        dt_f8_e8m0 = 9, dt_f8_e4m3fnuz = 10, dt_f8_e5m2fnuz = 11, &
                        dt_i16 = 12, dt_u16 = 13, dt_f16 = 14, dt_bf16 = 15, &
                        dt_i32 = 16, dt_u32 = 17, dt_f32 = 18, dt_c64 = 19, &
                        dt_f64 = 20, dt_i64 = 21, dt_u64 = 22

  character(len=12), parameter :: dt_names(st_ndtypes) = [character(len=12) :: &
    'BOOL', 'F4', 'F6_E2M3', 'F6_E3M2', 'U8', 'I8', 'F8_E5M2', 'F8_E4M3', &
    'F8_E8M0', 'F8_E4M3FNUZ', 'F8_E5M2FNUZ', 'I16', 'U16', 'F16', 'BF16', &
    'I32', 'U32', 'F32', 'C64', 'F64', 'I64', 'U64']
  ! bits per element; the table is the official Dtype enum's (safetensors/src/tensor.rs)
  integer, parameter :: dt_bits(st_ndtypes) = [8, 4, 6, 6, 8, 8, 8, 8, 8, 8, 8, &
                                              16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64]

  public :: st_dtype_bits, st_dtype_name, st_host_is_little_endian, st_writer, st_reader

  ! ------------------------------------------------------------------- types
  type :: st_kv
    character(len=:), allocatable :: k, v
  end type st_kv

  type :: st_tensor
    character(len=:), allocatable :: name
    integer :: dt = 0
    integer :: rank = 0
    integer(int64) :: shape(st_max_rank) = 0_int64
    integer(int64) :: nelem = 0
    integer(int64) :: nbytes = 0
    integer(int64) :: begin = 0
    integer(int64) :: finish = 0
  end type st_tensor

  type, public :: st_writer
    private
    type(st_tensor), allocatable :: t(:)
    integer :: nt = 0
    integer(int64) :: payload = 0
    integer(int8), allocatable :: buf(:)
    type(st_kv), allocatable :: md(:)
    integer :: nm = 0
    integer :: estat = st_ok          ! first pending error (see wr_fail)
    character(len=:), allocatable :: emsg
  contains
    procedure :: init => wr_init
    procedure :: reset => wr_init
    procedure :: set => wr_set
    procedure :: set_meta => wr_set_meta
    procedure :: set_meta_int => wr_set_meta_int
    procedure :: n_tensors => wr_n_tensors
    procedure :: tensor_name => wr_tensor_name
    procedure :: n_meta => wr_n_meta
    procedure :: meta_key => wr_meta_key
    procedure :: meta_value => wr_meta_value
    procedure :: payload_size => wr_payload_size
    procedure :: write => wr_write
    procedure :: to_bytes => wr_to_bytes
    procedure :: header_text => wr_header_text
    procedure :: error => wr_error
  end type st_writer

  type, public :: st_reader
    private
    logical :: opened = .false.
    character(len=:), allocatable :: path
    integer(int64) :: fsize = 0, hlen = 0, dstart = 0, bsize = 0
    type(st_tensor), allocatable :: t(:)
    integer :: nt = 0
    type(st_kv), allocatable :: md(:)
    integer :: nm = 0
  contains
    procedure :: open => rd_open
    procedure :: close => rd_close
    procedure :: n_tensors => rd_n_tensors
    procedure :: tensor_name => rd_tensor_name
    procedure :: shape => rd_shape
    procedure :: rank => rd_rank
    procedure :: dtype => rd_dtype
    procedure :: nbytes => rd_nbytes
    procedure :: tensor_offsets => rd_offsets
    procedure :: has => rd_has
    procedure :: n_meta => rd_n_meta
    procedure :: meta_key => rd_meta_key
    procedure :: meta => rd_meta
    procedure :: file_size => rd_file_size
    procedure :: header_size => rd_header_size
    procedure :: buffer_size => rd_buffer_size
    ! Every getter is a private binding; `get` is the generic that resolves type
    ! and rank from the pointer the caller declared (zero ambiguity).
    procedure, private :: rd_get_r32_1, rd_get_r32_2, rd_get_r64_1, rd_get_r64_2, &
      rd_get_i32_1, rd_get_i32_2, rd_get_i64_1, rd_get_i64_2, rd_get_u8_1, &
      rd_get_u8_2, rd_get_bool_1
    generic, public :: get => rd_get_r32_1, rd_get_r32_2, rd_get_r64_1, rd_get_r64_2, &
      rd_get_i32_1, rd_get_i32_2, rd_get_i64_1, rd_get_i64_2, rd_get_u8_1, &
      rd_get_u8_2, rd_get_bool_1
    procedure :: get_raw => rd_get_raw
  end type st_reader

contains

  ! ============================================================ common helpers
  pure function i2s(v) result(s)
    integer(int64), intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i2s

  pure function st_dtype_name(dt) result(s)
    integer, intent(in) :: dt
    character(len=:), allocatable :: s
    if (dt >= 1 .and. dt <= st_ndtypes) then
      s = trim(dt_names(dt))
    else
      s = '???'
    end if
  end function st_dtype_name

  pure function st_dtype_bits(dt) result(b)
    integer, intent(in) :: dt
    integer :: b
    if (dt >= 1 .and. dt <= st_ndtypes) then
      b = dt_bits(dt)
    else
      b = 0
    end if
  end function st_dtype_bits

  pure function dtype_lookup(name) result(dt)
    character(*), intent(in) :: name
    integer :: dt
    integer :: i
    dt = 0
    do i = 1, st_ndtypes
      if (trim(dt_names(i)) == name) then
        dt = i
        return
      end if
    end do
  end function dtype_lookup

  ! Host endianness. Tested, not assumed: writing little-endian on a big-endian
  ! host would produce a file that only looks right.
  pure function st_host_is_little_endian() result(le)
    logical :: le
    integer(int32) :: x
    integer(int8) :: b(4)
    x = 1_int32
    b = transfer(x, b)
    le = (b(1) == 1_int8)
  end function st_host_is_little_endian

  pure function le_from_i64(v) result(b)
    integer(int64), intent(in) :: v
    integer(int8) :: b(8)
    integer(int64) :: x
    integer :: i
    x = v
    do i = 1, 8
      b(i) = int(iand(x, 255_int64), int8)
      x = ishft(x, -8)
    end do
  end function le_from_i64

  pure function i64_from_le(b) result(v)
    integer(int8), intent(in) :: b(8)
    integer(int64) :: v
    integer :: i
    v = 0_int64
    do i = 8, 1, -1
      v = ior(ishft(v, 8), iand(int(b(i), int64), 255_int64))
    end do
  end function i64_from_le

  ! Byte-by-byte string comparison (byte-wise lexicographic order, not the
  ! processor's collating sequence). Used to sort the `__metadata__` keys.
  pure function str_less(a, b) result(r)
    character(*), intent(in) :: a, b
    logical :: r
    integer :: i, la, lb, ca, cb
    la = len(a)
    lb = len(b)
    do i = 1, min(la, lb)
      ca = iachar(a(i:i))
      cb = iachar(b(i:i))
      if (ca /= cb) then
        r = ca < cb
        return
      end if
    end do
    r = la < lb
  end function str_less

  subroutine grow_kv(a, n)
    type(st_kv), allocatable, intent(inout) :: a(:)
    integer, intent(in) :: n
    type(st_kv), allocatable :: tmp(:)
    integer :: cap
    cap = 0
    if (allocated(a)) cap = size(a)
    if (n <= cap) return
    allocate (tmp(max(8, 2*cap)))
    if (cap > 0) tmp(1:cap) = a(1:cap)
    call move_alloc(tmp, a)
  end subroutine grow_kv

  ! =============================================================== WRITER
  subroutine wr_init(self)
    class(st_writer), intent(inout) :: self
    self%nt = 0
    self%nm = 0
    self%payload = 0_int64
    self%estat = st_ok
    self%emsg = ''
    if (allocated(self%t)) deallocate (self%t)
    if (allocated(self%md)) deallocate (self%md)
    if (allocated(self%buf)) deallocate (self%buf)
  end subroutine wr_init

  function wr_n_tensors(self) result(n)
    class(st_writer), intent(in) :: self
    integer :: n
    n = self%nt
  end function wr_n_tensors

  function wr_n_meta(self) result(n)
    class(st_writer), intent(in) :: self
    integer :: n
    n = self%nm
  end function wr_n_meta

  function wr_payload_size(self) result(n)
    class(st_writer), intent(in) :: self
    integer(int64) :: n
    n = self%payload
  end function wr_payload_size

  subroutine wr_tensor_name(self, i, name, stat, msg)
    class(st_writer), intent(in) :: self
    integer, intent(in) :: i
    character(len=:), allocatable, intent(out) :: name
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    if (i < 1 .or. i > self%nt) then
      stat = st_err_range
      msg = 'tensor index '//trim(i2s(int(i, int64)))//' out of range 1..'//trim(i2s(int(self%nt, int64)))
      name = ''
      return
    end if
    name = self%t(i)%name
    stat = st_ok
    msg = ''
  end subroutine wr_tensor_name

  subroutine wr_meta_key(self, i, key, stat, msg)
    class(st_writer), intent(in) :: self
    integer, intent(in) :: i
    character(len=:), allocatable, intent(out) :: key
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    if (i < 1 .or. i > self%nm) then
      stat = st_err_range
      msg = 'metadata index out of range'
      key = ''
      return
    end if
    key = self%md(i)%k
    stat = st_ok
    msg = ''
  end subroutine wr_meta_key

  subroutine wr_meta_value(self, key, val, found)
    class(st_writer), intent(in) :: self
    character(*), intent(in) :: key
    character(len=:), allocatable, intent(out) :: val
    logical, intent(out) :: found
    integer :: i
    val = ''
    found = .false.
    do i = 1, self%nm
      if (self%md(i)%k == key) then
        val = self%md(i)%v
        found = .true.
        return
      end if
    end do
  end subroutine wr_meta_value

  ! Insert keeping lexicographic order (linear search: metadata holds dozens of
  ! keys, not millions). A repeated key OVERWRITES — `set` semantics.
  subroutine wr_set_meta(self, key, val, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: key, val
    integer, intent(out), optional :: stat
    character(len=:), allocatable, intent(out), optional :: msg
    integer :: i, pos
    if (present(stat)) stat = st_ok
    if (present(msg)) msg = ''
    pos = self%nm + 1
    do i = 1, self%nm
      if (self%md(i)%k == key) then
        self%md(i)%v = val
        return
      end if
      if (str_less(key, self%md(i)%k)) then
        pos = i
        exit
      end if
    end do
    call grow_kv(self%md, self%nm + 1)
    do i = self%nm, pos, -1
      self%md(i + 1) = self%md(i)
    end do
    self%md(pos)%k = key
    self%md(pos)%v = val
    self%nm = self%nm + 1
  end subroutine wr_set_meta

  subroutine wr_set_meta_int(self, key, val, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: key
    integer(int64), intent(in) :: val
    integer, intent(out), optional :: stat
    character(len=:), allocatable, intent(out), optional :: msg
    integer :: lstat
    character(len=:), allocatable :: lmsg
    call wr_set_meta(self, key, i2s(val), lstat, lmsg)
    if (present(stat)) stat = lstat
    if (present(msg)) msg = lmsg
  end subroutine wr_set_meta_int

  ! Generic `set` through assumed-rank (`class(*), intent(in) :: a(..)`): the type
  ! and rank come from the argument, so `call w%set("wte", wte)` works for
  ! real32/real64/int32/int64/int8/logical, 1-D or 2-D. An unsupported type
  ! returns an error listing what is supported — it never writes wrong bytes.
  subroutine wr_set(self, name, a, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: name
    class(*), intent(in) :: a(..)
    integer, intent(out), optional :: stat
    character(len=:), allocatable, intent(out), optional :: msg
    integer :: lstat
    character(len=:), allocatable :: lmsg

    lstat = st_ok
    lmsg = ''
    call wr_set_impl(self, name, a, lstat, lmsg)
    call wr_report(self, lstat, lmsg, stat, msg)
  end subroutine wr_set

  subroutine wr_report(self, lstat, lmsg, stat, msg)
    class(st_writer), intent(inout) :: self
    integer, intent(in) :: lstat
    character(*), intent(in) :: lmsg
    integer, intent(out), optional :: stat
    character(len=:), allocatable, intent(out), optional :: msg
    if (lstat /= st_ok) call wr_fail(self, lstat, lmsg)
    if (present(stat)) stat = lstat
    if (present(msg)) msg = lmsg
  end subroutine wr_report

  subroutine wr_set_impl(self, name, a, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: name
    class(*), intent(in) :: a(..)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg

    stat = st_ok
    msg = ''
    select rank (a)
    rank (0)
      stat = st_err_range
      msg = 'tensor '''//trim(name)//''': rank-0 (scalar) tensors are not supported; '// &
            'pass an array (wrap the scalar in a 1-element array)'
    rank (1)
      select type (x => a)
      type is (real(real32))
        call wr_add1(self, name, dt_f32, [int(size(x), int64)], x, 4_int64, stat, msg)
      type is (real(real64))
        call wr_add1(self, name, dt_f64, [int(size(x), int64)], x, 8_int64, stat, msg)
      type is (integer(int32))
        call wr_add1(self, name, dt_i32, [int(size(x), int64)], x, 4_int64, stat, msg)
      type is (integer(int64))
        call wr_add1(self, name, dt_i64, [int(size(x), int64)], x, 8_int64, stat, msg)
      type is (integer(int8))
        call wr_add1(self, name, dt_u8, [int(size(x), int64)], x, 1_int64, stat, msg)
      type is (logical)
        block
          integer(int8), allocatable :: tmp(:)
          integer :: j
          allocate (tmp(size(x)))
          do j = 1, size(x)
            tmp(j) = merge(1_int8, 0_int8, x(j))
          end do
          call wr_add1(self, name, dt_bool, [int(size(x), int64)], tmp, 1_int64, stat, msg)
        end block
      class default
        call wr_unsupported(name, stat, msg)
      end select
    rank (2)
      select type (x => a)
      type is (real(real32))
        call wr_add2(self, name, dt_f32, [int(size(x, 1), int64), int(size(x, 2), int64)], x, 4_int64, stat, msg)
      type is (real(real64))
        call wr_add2(self, name, dt_f64, [int(size(x, 1), int64), int(size(x, 2), int64)], x, 8_int64, stat, msg)
      type is (integer(int32))
        call wr_add2(self, name, dt_i32, [int(size(x, 1), int64), int(size(x, 2), int64)], x, 4_int64, stat, msg)
      type is (integer(int64))
        call wr_add2(self, name, dt_i64, [int(size(x, 1), int64), int(size(x, 2), int64)], x, 8_int64, stat, msg)
      type is (integer(int8))
        call wr_add2(self, name, dt_u8, [int(size(x, 1), int64), int(size(x, 2), int64)], x, 1_int64, stat, msg)
      type is (logical)
        block
          integer(int8), allocatable :: tmp(:, :)
          integer :: j, k
          allocate (tmp(size(x, 1), size(x, 2)))
          do k = 1, size(x, 2)
            do j = 1, size(x, 1)
              tmp(j, k) = merge(1_int8, 0_int8, x(j, k))
            end do
          end do
          call wr_add2(self, name, dt_bool, [int(size(x, 1), int64), int(size(x, 2), int64)], tmp, 1_int64, stat, msg)
        end block
      class default
        call wr_unsupported(name, stat, msg)
      end select
    rank default
      stat = st_err_range
      msg = 'tensor '''//trim(name)//''': only rank 1 and rank 2 arrays are supported'
    end select
  end subroutine wr_set_impl

  subroutine wr_unsupported(name, stat, msg)
    character(*), intent(in) :: name
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    stat = st_err_dtype
    msg = 'tensor '''//trim(name)//''': unsupported Fortran type. Supported: '// &
          'real(real32), real(real64), integer(int32), integer(int64), '// &
          'integer(int8) (written as U8) and logical (written as BOOL)'
  end subroutine wr_unsupported

  subroutine wr_add1(self, name, dt, shape, src, isize, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dt
    integer(int64), intent(in) :: shape(:)
    class(*), intent(in) :: src(:)
    integer(int64), intent(in) :: isize
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer(int64) :: nb
    nb = int(size(src), int64)*isize
    call wr_commit(self, name, dt, shape, nb, stat, msg)
    if (stat /= st_ok) return
    if (nb > 0) call wr_put_bytes(self, transfer(src, 0_int8, nb))
  end subroutine wr_add1

  subroutine wr_add2(self, name, dt, shape, src, isize, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dt
    integer(int64), intent(in) :: shape(:)
    class(*), intent(in) :: src(:, :)
    integer(int64), intent(in) :: isize
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer(int64) :: nb
    nb = int(size(src), int64)*isize
    call wr_commit(self, name, dt, shape, nb, stat, msg)
    if (stat /= st_ok) return
    if (nb > 0) call wr_put_bytes(self, transfer(src, 0_int8, nb))
  end subroutine wr_add2

  ! Registers the tensor (name/dtype/shape/offsets) and reserves payload space.
  ! No arithmetic on the VALUES: the payload is raw bytes, which is why NaN/Inf/-0.0
  ! travel through unchanged.
  subroutine wr_commit(self, name, dt, shape, nb, stat, msg)
    class(st_writer), intent(inout) :: self
    character(*), intent(in) :: name
    integer, intent(in) :: dt
    integer(int64), intent(in) :: shape(:)
    integer(int64), intent(in) :: nb
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    type(st_tensor), allocatable :: tmp(:)
    integer :: i, r, ncap

    stat = st_ok
    msg = ''
    if (.not. st_host_is_little_endian()) then
      stat = st_err_endian
      msg = 'this host is big-endian; safetensors is little-endian and this build '// &
            'does not implement byte swapping'
      return
    end if
    if (len(name) == 0) then
      stat = st_err_schema
      msg = 'empty tensor name is not allowed'
      return
    end if
    if (name == '__metadata__') then
      stat = st_err_schema
      msg = 'tensor name ''__metadata__'' is reserved by the format'
      return
    end if
    ! Duplicate-name detection: linear search. A real model has O(10^2..10^3)
    ! tensors; the alternative (a hash) would only pay off at O(10^4) and is not
    ! worth the complexity here (the READER already has a hash set for hostile input).
    do i = 1, self%nt
      if (self%t(i)%name == name) then
        stat = st_err_duplicate
        msg = 'tensor '''//trim(name)//''' was already added to this writer'
        return
      end if
    end do
    if (size(shape) < 1 .or. size(shape) > st_max_rank) then
      stat = st_err_range
      msg = 'tensor '''//trim(name)//''': rank must be 1..'//trim(i2s(int(st_max_rank, int64)))
      return
    end if
    ncap = 0
    if (allocated(self%t)) ncap = size(self%t)
    if (self%nt + 1 > ncap) then
      allocate (tmp(max(16, 2*ncap)))
      if (self%nt > 0) tmp(1:self%nt) = self%t(1:self%nt)
      call move_alloc(tmp, self%t)
    end if
    self%nt = self%nt + 1
    associate (e => self%t(self%nt))
      e%name = name
      e%dt = dt
      e%rank = size(shape)
      e%shape = 0_int64
      do r = 1, size(shape)
        if (shape(r) < 0) then
          stat = st_err_range
          msg = 'tensor '''//trim(name)//''': negative extent in shape'
          self%nt = self%nt - 1
          return
        end if
        e%shape(r) = shape(r)
      end do
      e%nelem = 1_int64
      do r = 1, e%rank
        if (e%shape(r) > 0 .and. e%nelem > huge(0_int64)/max(1_int64, e%shape(r))) then
          stat = st_err_range
          msg = 'tensor '''//trim(name)//''': shape product overflows int64'
          self%nt = self%nt - 1
          return
        end if
        e%nelem = e%nelem*e%shape(r)
      end do
      e%nbytes = nb
      e%begin = self%payload
      e%finish = self%payload + nb
    end associate
    self%payload = self%payload + nb
  end subroutine wr_commit

  subroutine wr_put_bytes(self, bytes)
    class(st_writer), intent(inout) :: self
    integer(int8), intent(in) :: bytes(:)
    integer(int8), allocatable :: tmp(:)
    integer(int64) :: need
    need = self%payload
    if (.not. allocated(self%buf)) then
      allocate (self%buf(max(1024_int64, need)))
    else if (need > size(self%buf, kind=int64)) then
      allocate (tmp(max(2_int64*size(self%buf, kind=int64), need)))
      if (size(self%buf, kind=int64) > 0) tmp(1:size(self%buf)) = self%buf
      call move_alloc(tmp, self%buf)
    end if
    self%buf(need - size(bytes, kind=int64) + 1:need) = bytes
  end subroutine wr_put_bytes

  ! Builds the header text (no padding) and the length already aligned to 8.
  subroutine wr_header_text(self, htext, aligned_len, stat, msg)
    class(st_writer), intent(in) :: self
    character(len=:), allocatable, intent(out) :: htext
    integer(int64), intent(out) :: aligned_len
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    type(json_writer) :: jw
    integer :: i, r
    integer(int64) :: pad

    call jw%reset()
    call jw%raw('{"__metadata__":{')
    do i = 1, self%nm
      if (i > 1) call jw%raw(',')
      call jw%string(self%md(i)%k)
      call jw%raw(':')
      call jw%string(self%md(i)%v)
    end do
    call jw%raw('}')
    do i = 1, self%nt
      call jw%raw(',')
      call jw%string(self%t(i)%name)
      call jw%raw(':{"dtype":')
      call jw%string(trim(dt_names(self%t(i)%dt)))
      call jw%raw(',"shape":[')
      do r = 1, self%t(i)%rank
        if (r > 1) call jw%raw(',')
        call jw%integer(self%t(i)%shape(r))
      end do
      call jw%raw('],"data_offsets":[')
      call jw%integer(self%t(i)%begin)
      call jw%raw(',')
      call jw%integer(self%t(i)%finish)
      call jw%raw(']}')
    end do
    call jw%raw('}')
    htext = jw%finish()
    pad = mod(8_int64 - mod(int(len(htext), int64), 8_int64), 8_int64)
    aligned_len = int(len(htext), int64) + pad
    stat = st_ok
    msg = ''
  end subroutine wr_header_text

  ! Pending writer error. `set`/`set_meta` take OPTIONAL stat/msg (for the common
  ! case of someone who only wants to write), but nothing is lost silently: the
  ! first error is kept and `write` refuses to run while it is pending.
  subroutine wr_error(self, stat, msg)
    class(st_writer), intent(in) :: self
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    stat = self%estat
    msg = self%emsg
  end subroutine wr_error

  subroutine wr_fail(self, stat, msg)
    class(st_writer), intent(inout) :: self
    integer, intent(in) :: stat
    character(*), intent(in) :: msg
    if (self%estat == st_ok) then
      self%estat = stat
      self%emsg = msg
    end if
  end subroutine wr_fail

  subroutine wr_write(self, path, stat, msg)
    class(st_writer), intent(in) :: self
    character(*), intent(in) :: path
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    character(len=:), allocatable :: htext
    character(len=:), allocatable :: hpad
    integer(int64) :: alen
    integer(int8) :: n8(8)
    integer :: u, ios
    character(len=256) :: iomsg

    if (self%estat /= st_ok) then
      stat = self%estat
      msg = 'refusing to write: an earlier set/set_meta call failed: '//self%emsg
      return
    end if
    call wr_header_text(self, htext, alen, stat, msg)
    if (stat /= st_ok) return
    if (alen > st_max_header_bytes) then
      stat = st_err_header_size
      msg = 'header would be '//trim(i2s(alen))//' bytes, above the '// &
            trim(i2s(st_max_header_bytes))//'-byte limit of the format'
      return
    end if
    hpad = htext//repeat(' ', int(alen - int(len(htext), int64)))
    n8 = le_from_i64(alen)
    iomsg = ''
    open (newunit=u, file=path, access='stream', form='unformatted', &
          status='replace', action='write', iostat=ios, iomsg=iomsg)
    if (ios /= 0) then
      call io_err('cannot open '''//trim(path)//''' for writing', iomsg, stat, msg)
      return
    end if
    write (u, iostat=ios, iomsg=iomsg) n8
    if (ios == 0) write (u, iostat=ios, iomsg=iomsg) hpad
    if (ios == 0 .and. self%payload > 0) write (u, iostat=ios, iomsg=iomsg) self%buf(1:self%payload)
    close (u, iostat=ios)
    if (ios /= 0) then
      call io_err('write failed for '''//trim(path)//'''', iomsg, stat, msg)
      return
    end if
    stat = st_ok
    msg = ''
  end subroutine wr_write

  ! The whole file image in memory. It exists so the byte-for-byte parity test
  ! does not need a temporary file (and for anyone who wants to send the file
  ! over a network/socket). Here there IS an extra copy of the payload.
  subroutine wr_to_bytes(self, image, stat, msg)
    class(st_writer), intent(in) :: self
    integer(int8), allocatable, intent(out) :: image(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    character(len=:), allocatable :: htext
    integer(int64) :: alen, total
    integer :: i

    if (self%estat /= st_ok) then
      stat = self%estat
      msg = 'refusing to build the image: an earlier set/set_meta call failed: '//self%emsg
      return
    end if
    call wr_header_text(self, htext, alen, stat, msg)
    if (stat /= st_ok) return
    if (alen > st_max_header_bytes) then
      stat = st_err_header_size
      msg = 'header too large'
      return
    end if
    total = 8_int64 + alen + self%payload
    allocate (image(total))
    image(1:8) = le_from_i64(alen)
    do i = 1, int(alen)
      if (i <= len(htext)) then
        image(8 + i) = int(iachar(htext(i:i)), int8)
      else
        image(8 + i) = 32_int8                       ! padding with a space (0x20)
      end if
    end do
    if (self%payload > 0) image(9 + alen:total) = self%buf(1:self%payload)
    stat = st_ok
    msg = ''
  end subroutine wr_to_bytes

  subroutine io_err(what, iomsg, stat, msg)
    character(*), intent(in) :: what, iomsg
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    stat = st_err_io
    if (len_trim(iomsg) > 0) then
      msg = what//': '//trim(iomsg)
    else
      msg = what
    end if
  end subroutine io_err

  ! =============================================================== READER
  subroutine rd_close(self)
    class(st_reader), intent(inout) :: self
    self%opened = .false.
    self%nt = 0
    self%nm = 0
    self%fsize = 0
    self%hlen = 0
    self%dstart = 0
    self%bsize = 0
    if (allocated(self%path)) deallocate (self%path)
    if (allocated(self%t)) deallocate (self%t)
    if (allocated(self%md)) deallocate (self%md)
  end subroutine rd_close

  function rd_file_size(self) result(n)
    class(st_reader), intent(in) :: self
    integer(int64) :: n
    n = self%fsize
  end function rd_file_size

  function rd_header_size(self) result(n)
    class(st_reader), intent(in) :: self
    integer(int64) :: n
    n = self%hlen
  end function rd_header_size

  function rd_buffer_size(self) result(n)
    class(st_reader), intent(in) :: self
    integer(int64) :: n
    n = self%bsize
  end function rd_buffer_size

  function rd_n_tensors(self) result(n)
    class(st_reader), intent(in) :: self
    integer :: n
    n = self%nt
  end function rd_n_tensors

  function rd_n_meta(self) result(n)
    class(st_reader), intent(in) :: self
    integer :: n
    n = self%nm
  end function rd_n_meta

  ! Opens and VALIDATES the whole header (offsets, dtypes, buffer coverage)
  ! without materialising a single tensor: that is what makes it possible to
  ! query the shape/dtype of a 30 GB file for the price of the header.
  subroutine rd_open(self, path, stat, msg)
    class(st_reader), intent(inout) :: self
    character(*), intent(in) :: path
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    character(len=:), allocatable :: htext, jmsg
    integer(int8) :: n8(8)
    integer :: u, ios, i, j, idx, jstat, nkeys
    integer(int64) :: fsz
    logical :: ex
    character(len=256) :: iomsg

    call rd_close(self)
    stat = st_ok
    msg = ''
    if (.not. st_host_is_little_endian()) then
      stat = st_err_endian
      msg = 'this host is big-endian; safetensors is little-endian and this build '// &
            'does not implement byte swapping'
      return
    end if
    inquire (file=path, exist=ex, size=fsz, iostat=ios)
    if (.not. ex) then
      stat = st_err_not_found
      msg = 'file not found: '//trim(path)
      return
    end if
    if (fsz < 8) then
      stat = st_err_truncated
      msg = 'file '''//trim(path)//''' has '//trim(i2s(fsz))// &
            ' bytes: a safetensors file needs at least the 8 bytes of the header length'
      return
    end if
    iomsg = ''
    open (newunit=u, file=path, access='stream', form='unformatted', &
          status='old', action='read', iostat=ios, iomsg=iomsg)
    if (ios /= 0) then
      call io_err('cannot open '''//trim(path)//'''', iomsg, stat, msg)
      return
    end if
    read (u, pos=1, iostat=ios, iomsg=iomsg) n8
    if (ios /= 0) then
      call io_err('cannot read the header length of '''//trim(path)//'''', iomsg, stat, msg)
      close (u)
      return
    end if
    self%hlen = i64_from_le(n8)
    if (self%hlen < 0) then
      stat = st_err_header_size
      msg = 'header length '//trim(i2s(self%hlen))//' is negative (only the sign bit'// &
            ' of the 8-byte length is set): not a safetensors file'
      close (u)
      return
    end if
    if (self%hlen == 0) then
      stat = st_err_header_size
      msg = 'header length is 0: the header must contain at least a JSON object'
      close (u)
      return
    end if
    if (self%hlen > st_max_header_bytes) then
      stat = st_err_header_size
      msg = 'header claims '//trim(i2s(self%hlen))//' bytes, above the format limit of '// &
            trim(i2s(st_max_header_bytes))//' bytes'
      close (u)
      return
    end if
    if (8_int64 + self%hlen > fsz) then
      stat = st_err_truncated
      msg = 'truncated file: the header declares '//trim(i2s(self%hlen))// &
            ' bytes but only '//trim(i2s(fsz - 8_int64))//' bytes follow the length'
      close (u)
      return
    end if
    allocate (character(len=int(self%hlen)) :: htext)
    read (u, pos=9, iostat=ios, iomsg=iomsg) htext
    close (u, iostat=ios)
    if (ios /= 0) then
      stat = st_err_io
      msg = 'cannot read the '//trim(i2s(self%hlen))//'-byte header of '''//trim(path)//''''
      return
    end if
    ! Sizes BEFORE parsing: the coverage validation needs to know how many bytes
    ! exist after the header.
    self%fsize = fsz
    self%dstart = 8_int64 + self%hlen
    self%bsize = fsz - self%dstart
    call rd_parse_header(self, htext, stat, msg)
    if (stat /= st_ok) then
      call rd_close(self)
      return
    end if
    self%path = path
    self%opened = .true.
  end subroutine rd_open

  subroutine rd_parse_header(self, htext, stat, msg)
    class(st_reader), intent(inout) :: self
    character(*), intent(in) :: htext
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    type(json_doc) :: doc
    character(len=:), allocatable :: jmsg, tname, dtxt
    integer :: jstat, i, j, idx, nkeys, nten, dt, r, root, inode, fnode
    integer(int64) :: nelem, bits, nbytes, b0, b1

    call json_parse(htext, doc, jstat, jmsg)
    if (jstat /= json_ok) then
      stat = st_err_json
      msg = 'invalid header JSON: '//trim(jmsg)
      return
    end if
    root = doc%root()
    if (root < 1) then
      stat = st_err_json
      msg = 'invalid header JSON: no value could be parsed'
      return
    end if
    if (doc%kind_of(root) /= json_object) then
      stat = st_err_schema
      msg = 'the header must be a JSON object (got '//trim(kind_name(doc%kind_of(root)))//')'
      return
    end if
    call rd_read_metadata(self, doc, root, stat, msg)
    if (stat /= st_ok) return

    nkeys = doc%count(root)
    idx = doc%member(root, '__metadata__')
    nten = nkeys - merge(1, 0, idx > 0)
    allocate (self%t(max(1, nten)))
    self%nt = 0
    do i = 1, nkeys
      if (doc%child(root, i) == idx .and. idx > 0) cycle
      tname = doc%key(root, i)
      inode = doc%child(root, i)
      if (doc%kind_of(inode) /= json_object) then
        stat = st_err_schema
        msg = 'entry '''//tname//''' must be a JSON object with dtype/shape/data_offsets'
        return
      end if
      self%nt = self%nt + 1
      associate (e => self%t(self%nt))
        e%name = tname
        ! ---- dtype
        fnode = doc%member(inode, 'dtype')
        if (fnode == 0) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''' has no ''dtype'' field'
          return
        end if
        if (doc%kind_of(fnode) /= json_string) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''': ''dtype'' must be a string'
          return
        end if
        dtxt = doc%str_of(fnode)
        dt = dtype_lookup(dtxt)
        if (dt == 0) then
          stat = st_err_dtype
          msg = 'tensor '''//tname//''': unknown dtype '''//dtxt//'''. Known dtypes: '// &
                'BOOL, U8, I8, I16, U16, F16, BF16, I32, U32, F32, C64, F64, I64, U64, '// &
                'F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ, F4, F6_E2M3, F6_E3M2'
          return
        end if
        e%dt = dt
        ! ---- shape
        fnode = doc%member(inode, 'shape')
        if (fnode == 0) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''' has no ''shape'' field'
          return
        end if
        if (doc%kind_of(fnode) /= json_array) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''': ''shape'' must be an array of non-negative integers'
          return
        end if
        e%rank = doc%count(fnode)
        if (e%rank < 1 .or. e%rank > st_max_rank) then
          stat = st_err_range
          msg = 'tensor '''//tname//''': shape has '//trim(i2s(int(e%rank, int64)))// &
                ' dimensions; this library supports 1..'//trim(i2s(int(st_max_rank, int64)))
          return
        end if
        e%shape = 0_int64
        nelem = 1_int64
        do r = 1, e%rank
          if (doc%kind_of(doc%child(fnode, r)) /= json_number) then
            stat = st_err_schema
            msg = 'tensor '''//tname//''': shape element '//trim(i2s(int(r, int64)))// &
                  ' is not an integer'
            return
          end if
          e%shape(r) = doc%int_of(doc%child(fnode, r))
          if (e%shape(r) < 0) then
            stat = st_err_schema
            msg = 'tensor '''//tname//''': negative extent in shape'
            return
          end if
          ! No short-circuit in `.and.`: a 0 divisor would SIGFPE on integer
          ! division. A zero extent zeroes the product (as arithmetic demands).
          if (e%shape(r) == 0) then
            nelem = 0_int64
          else
            if (nelem > huge(0_int64)/e%shape(r)) then
              stat = st_err_schema
              msg = 'tensor '''//tname//''': shape product overflows int64'
              return
            end if
            nelem = nelem*e%shape(r)
          end if
        end do
        e%nelem = nelem
        bits = nelem*int(st_dtype_bits(dt), int64)
        if (mod(bits, 8_int64) /= 0) then
          stat = st_err_offsets
          msg = 'tensor '''//tname//''': dtype '//trim(dt_names(dt))//' with '// &
                trim(i2s(nelem))//' elements is not a whole number of bytes'
          return
        end if
        e%nbytes = bits/8_int64
        ! ---- data_offsets
        fnode = doc%member(inode, 'data_offsets')
        if (fnode == 0) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''' has no ''data_offsets'' field'
          return
        end if
        if (doc%kind_of(fnode) /= json_array .or. doc%count(fnode) /= 2) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''': ''data_offsets'' must be an array of exactly 2 integers'
          return
        end if
        if (doc%kind_of(doc%child(fnode, 1)) /= json_number .or. &
            doc%kind_of(doc%child(fnode, 2)) /= json_number) then
          stat = st_err_schema
          msg = 'tensor '''//tname//''': ''data_offsets'' elements must be integers'
          return
        end if
        b0 = doc%int_of(doc%child(fnode, 1))
        b1 = doc%int_of(doc%child(fnode, 2))
        if (b0 < 0 .or. b1 < 0) then
          stat = st_err_offsets
          msg = 'tensor '''//tname//''': negative data offset'
          return
        end if
        if (b1 < b0) then
          stat = st_err_offsets
          msg = 'tensor '''//tname//''': data_offsets ['//trim(i2s(b0))//','//trim(i2s(b1))// &
                '] has end < begin'
          return
        end if
        if (b1 - b0 /= e%nbytes) then
          stat = st_err_offsets
          msg = 'tensor '''//tname//''': data_offsets span '//trim(i2s(b1 - b0))// &
                ' bytes but shape '//shape_str(e)//' with dtype '//trim(dt_names(dt))// &
                ' needs '//trim(i2s(e%nbytes))//' bytes'
          return
        end if
        e%begin = b0
        e%finish = b1
      end associate
    end do
    call rd_check_coverage(self, stat, msg)
  end subroutine rd_parse_header

  ! Sorts by start offset and demands CONTIGUOUS and EXACT coverage of the buffer:
  ! no overlap and no hole. That is the rule that blocks a polyglot file (a file
  ! that is safetensors AND something else at the same time), and it is the same
  ! rule the official implementation enforces.
  subroutine rd_check_coverage(self, stat, msg)
    class(st_reader), intent(inout) :: self
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer(int64), allocatable :: keys(:)
    integer, allocatable :: idx(:)
    type(st_tensor), allocatable :: tmp(:)
    integer(int64) :: cursor
    integer :: i

    stat = st_ok
    msg = ''
    if (self%nt == 0) return
    allocate (keys(self%nt))
    do i = 1, self%nt
      keys(i) = self%t(i)%begin
    end do
    call sort_indices(keys, idx)
    allocate (tmp(self%nt))
    do i = 1, self%nt
      tmp(i) = self%t(idx(i))
    end do
    call move_alloc(tmp, self%t)
    cursor = 0_int64
    do i = 1, self%nt
      if (self%t(i)%begin /= cursor) then
        stat = st_err_offsets
        if (self%t(i)%begin < cursor) then
          msg = 'tensor '''//self%t(i)%name//''' starts at byte '//trim(i2s(self%t(i)%begin))// &
                ' but the previous tensor already ends at byte '//trim(i2s(cursor))// &
                ' (overlapping data_offsets are not allowed)'
        else
          msg = 'tensor '''//self%t(i)%name//''' starts at byte '//trim(i2s(self%t(i)%begin))// &
                ' but the previous tensor ends at byte '//trim(i2s(cursor))// &
                ' (holes in the byte buffer are not allowed)'
        end if
        return
      end if
      cursor = self%t(i)%finish
    end do
    if (self%bsize >= 0 .and. cursor /= self%bsize) then
      stat = st_err_offsets
      if (cursor > self%bsize) then
        msg = 'tensors need '//trim(i2s(cursor))//' bytes but the file only has '// &
              trim(i2s(self%bsize))//' after the header (truncated payload)'
      else
        msg = 'tensors cover '//trim(i2s(cursor))//' bytes but the file has '// &
              trim(i2s(self%bsize))//' after the header (trailing bytes are not allowed)'
      end if
    end if
  end subroutine rd_check_coverage

  subroutine rd_read_metadata(self, doc, root, stat, msg)
    class(st_reader), intent(inout) :: self
    type(json_doc), intent(in) :: doc
    integer, intent(in) :: root
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: idx, i, cn

    stat = st_ok
    msg = ''
    idx = doc%member(root, '__metadata__')
    if (idx == 0) return
    if (doc%kind_of(idx) /= json_object) then
      stat = st_err_schema
      msg = '''__metadata__'' must be a JSON object mapping strings to strings'
      return
    end if
    self%nm = doc%count(idx)
    if (self%nm == 0) return
    allocate (self%md(self%nm))
    do i = 1, self%nm
      self%md(i)%k = doc%key(idx, i)
      cn = doc%child(idx, i)
      if (doc%kind_of(cn) /= json_string) then
        stat = st_err_schema
        msg = '''__metadata__['''//self%md(i)%k//'''] must be a string (arbitrary JSON '// &
              'is not allowed by the format)'
        self%nm = 0
        return
      end if
      self%md(i)%v = doc%str_of(cn)
    end do
  end subroutine rd_read_metadata

  function kind_name(k) result(s)
    integer, intent(in) :: k
    character(len=:), allocatable :: s
    select case (k)
    case (json_object); s = 'object'
    case (json_array); s = 'array'
    case (json_string); s = 'string'
    case (json_number); s = 'number'
    case (json_bool); s = 'boolean'
    case default; s = 'null'
    end select
  end function kind_name

  function shape_str(e) result(s)
    type(st_tensor), intent(in) :: e
    character(len=:), allocatable :: s
    integer :: r
    s = '['
    do r = 1, e%rank
      if (r > 1) s = s//','
      s = s//trim(i2s(e%shape(r)))
    end do
    s = s//']'
  end function shape_str

  ! -------------------------------------------------------------------- queries
  subroutine rd_find(self, name, idx, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer, intent(out) :: idx
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    idx = 0
    stat = st_ok
    msg = ''
    if (.not. self%opened) then
      stat = st_err_not_open
      msg = 'reader is not open: call r%open(path, stat, msg) first'
      return
    end if
    do i = 1, self%nt
      if (self%t(i)%name == name) then
        idx = i
        return
      end if
    end do
    stat = st_err_missing
    msg = 'tensor '''//trim(name)//''' not found in '''//trim(self%path)//''' ('// &
          trim(i2s(int(self%nt, int64)))//' tensors'
    if (self%nt > 0 .and. self%nt <= 24) then
      msg = msg//': '
      do i = 1, self%nt
        if (i > 1) msg = msg//', '
        msg = msg//self%t(i)%name
      end do
    end if
    msg = msg//')'
  end subroutine rd_find

  function rd_has(self, name) result(found)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    logical :: found
    integer :: i
    found = .false.
    if (.not. self%opened) return
    do i = 1, self%nt
      if (self%t(i)%name == name) then
        found = .true.
        return
      end if
    end do
  end function rd_has

  subroutine require_open(self, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    if (self%opened) then
      stat = st_ok
      msg = ''
    else
      stat = st_err_not_open
      msg = 'reader is not open: call r%open(path, stat, msg) first'
    end if
  end subroutine require_open

  subroutine rd_tensor_name(self, i, name, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(in) :: i
    character(len=:), allocatable, intent(out) :: name
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    name = ''
    call require_open(self, stat, msg)
    if (stat /= st_ok) return
    if (i < 1 .or. i > self%nt) then
      stat = st_err_range
      msg = 'tensor index out of range 1..'//trim(i2s(int(self%nt, int64)))
      return
    end if
    name = self%t(i)%name
    stat = st_ok
    msg = ''
  end subroutine rd_tensor_name

  ! Shape in C ORDER (the same as the header and numpy/torch). For the Fortran
  ! array the memory is the same one; see the get_*_2d comment.
  subroutine rd_shape(self, name, shp, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer(int64), allocatable, intent(out) :: shp(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i, r
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) then
      allocate (shp(0))
      return
    end if
    allocate (shp(self%t(i)%rank))
    do r = 1, self%t(i)%rank
      shp(r) = self%t(i)%shape(r)
    end do
  end subroutine rd_shape

  subroutine rd_rank(self, name, r, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer, intent(out) :: r
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    r = 0
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) return
    r = self%t(i)%rank
  end subroutine rd_rank

  subroutine rd_dtype(self, name, dtxt, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    character(len=:), allocatable, intent(out) :: dtxt
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    dtxt = ''
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) return
    dtxt = trim(dt_names(self%t(i)%dt))
  end subroutine rd_dtype

  subroutine rd_nbytes(self, name, nb, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer(int64), intent(out) :: nb
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    nb = 0
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) return
    nb = self%t(i)%nbytes
  end subroutine rd_nbytes

  subroutine rd_offsets(self, name, b0, b1, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer(int64), intent(out) :: b0, b1
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    b0 = 0
    b1 = 0
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) return
    b0 = self%t(i)%begin
    b1 = self%t(i)%finish
  end subroutine rd_offsets

  subroutine rd_meta_key(self, i, key, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(in) :: i
    character(len=:), allocatable, intent(out) :: key
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    key = ''
    call require_open(self, stat, msg)
    if (stat /= st_ok) return
    if (i < 1 .or. i > self%nm) then
      stat = st_err_range
      msg = 'metadata index out of range 1..'//trim(i2s(int(self%nm, int64)))
      return
    end if
    key = self%md(i)%k
  end subroutine rd_meta_key

  ! Signature from the task sketch: `call r%meta("bpb", val, found)`.
  subroutine rd_meta(self, key, val, found)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: key
    character(len=:), allocatable, intent(out) :: val
    logical, intent(out) :: found
    integer :: i
    val = ''
    found = .false.
    do i = 1, self%nm
      if (self%md(i)%k == key) then
        val = self%md(i)%v
        found = .true.
        return
      end if
    end do
  end subroutine rd_meta

  ! -------------------------------------------------------------------- reading
  ! Reads the tensor bytes straight into the memory of the returned array: one
  ! positional read, no intermediate buffer (c_f_pointer sees the pointer target
  ! as int8). No arithmetic is done on the values, so -0.0/NaN/Inf
  ! travel through intact.
  subroutine read_at(self, i, dst, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(in) :: i
    type(c_ptr), intent(in) :: dst
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer(int8), pointer :: raw(:)
    integer :: u, ios
    character(len=256) :: iomsg
    integer(int64) :: nb

    nb = self%t(i)%nbytes
    if (nb <= 0) then
      stat = st_ok
      msg = ''
      return
    end if
    call c_f_pointer(dst, raw, [nb])
    iomsg = ''
    open (newunit=u, file=self%path, access='stream', form='unformatted', &
          status='old', action='read', iostat=ios, iomsg=iomsg)
    if (ios /= 0) then
      call io_err('cannot reopen '''//trim(self%path)//'''', iomsg, stat, msg)
      return
    end if
    read (u, pos=self%dstart + self%t(i)%begin + 1_int64, iostat=ios, iomsg=iomsg) raw
    close (u)
    if (ios /= 0) then
      call io_err('short read for tensor '''//self%t(i)%name//'''', iomsg, stat, msg)
      return
    end if
    stat = st_ok
    msg = ''
  end subroutine read_at

  ! Checks whether the tensor dtype matches the requested Fortran type, with a
  ! message that names BOTH sides (not a bare "type error").
  subroutine check_dt(self, i, want, asked, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(in) :: i, want
    character(*), intent(in) :: asked
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    if (self%t(i)%dt /= want) then
      stat = st_err_type_mismatch
      msg = 'tensor '''//self%t(i)%name//''' has dtype '//trim(dt_names(self%t(i)%dt))// &
            ' but it was requested as '//trim(asked)//'; use the matching getter '// &
            '(or get_raw for the bytes)'
    else
      stat = st_ok
      msg = ''
    end if
  end subroutine check_dt

  subroutine rd_get_raw(self, name, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: name
    integer(int8), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_find(self, name, i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nbytes))
    call read_at(self, i, c_loc(p(1)), stat, msg)
    if (stat /= st_ok) then
      deallocate (p)
      p => null()
    end if
  end subroutine rd_get_raw

! --------------------------------------------------------------------- getters
! They all follow the same design: `rd_prepare` finds the tensor and checks its
! dtype against the requested Fortran type (the message names BOTH), the array is
! allocated at its final size, and the positional read writes straight into that
! memory. Zero intermediate copy: that is what "no extra copy" means here.
!
! About rank: a rank-N tensor in the header is, in memory, a block in C order
! (row-major). Fortran is column-major, so `get_*_1d` returns the flattened block
! and `get_*_2d` returns the extents reversed:
!     header shape [d0, d1]  ->  Fortran array (d1, d0)   [a(1,1) = (0,0) in C]
!     header shape [n]       ->  Fortran array (1, n)
! That way nothing is transposed or copied, and `sum(a)` matches on both sides.

  ! Fortran extents (column-major) of the 2-D array matching a C-order shape.
  subroutine fortran_dims2(e, d1, d2)
    type(st_tensor), intent(in) :: e
    integer(int64), intent(out) :: d1, d2
    if (e%rank == 1) then
      d1 = 1_int64
      d2 = e%shape(1)
    else
      d1 = e%shape(2)
      d2 = e%shape(1)
    end if
  end subroutine fortran_dims2

  ! Lookup + dtype check in a single call (used by every getter).
  subroutine rd_prepare(self, tname, want_dt, asked, i, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer, intent(in) :: want_dt
    character(*), intent(in) :: asked
    integer, intent(out) :: i
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    call rd_find(self, tname, i, stat, msg)
    if (stat /= st_ok) return
    call check_dt(self, i, want_dt, asked, stat, msg)
  end subroutine rd_prepare

  subroutine check_rank2(self, i, stat, msg)
    class(st_reader), intent(in) :: self
    integer, intent(in) :: i
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    if (self%t(i)%rank > 2) then
      stat = st_err_range
      msg = 'tensor '''//self%t(i)%name//''' has rank '//trim(i2s(int(self%t(i)%rank, int64)))// &
            '; the 2-D getters accept rank 1 or 2 (use the flat getter or get_raw)'
    else
      stat = st_ok
      msg = ''
    end if
  end subroutine check_rank2

  subroutine rd_get_r32_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    real(real32), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_prepare(self, tname, dt_f32, 'real(real32)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_r32_1

  subroutine rd_get_r32_2(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    real(real32), pointer, intent(out) :: p(:, :)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    integer(int64) :: d1, d2
    p => null()
    call rd_prepare(self, tname, dt_f32, 'real(real32)', i, stat, msg)
    if (stat /= st_ok) return
    call check_rank2(self, i, stat, msg)
    if (stat /= st_ok) return
    call fortran_dims2(self%t(i), d1, d2)
    allocate (p(d1, d2))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1, 1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_r32_2

  subroutine rd_get_r64_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    real(real64), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_prepare(self, tname, dt_f64, 'real(real64)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_r64_1

  subroutine rd_get_r64_2(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    real(real64), pointer, intent(out) :: p(:, :)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    integer(int64) :: d1, d2
    p => null()
    call rd_prepare(self, tname, dt_f64, 'real(real64)', i, stat, msg)
    if (stat /= st_ok) return
    call check_rank2(self, i, stat, msg)
    if (stat /= st_ok) return
    call fortran_dims2(self%t(i), d1, d2)
    allocate (p(d1, d2))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1, 1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_r64_2

  subroutine rd_get_i32_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int32), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_prepare(self, tname, dt_i32, 'integer(int32)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_i32_1

  subroutine rd_get_i32_2(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int32), pointer, intent(out) :: p(:, :)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    integer(int64) :: d1, d2
    p => null()
    call rd_prepare(self, tname, dt_i32, 'integer(int32)', i, stat, msg)
    if (stat /= st_ok) return
    call check_rank2(self, i, stat, msg)
    if (stat /= st_ok) return
    call fortran_dims2(self%t(i), d1, d2)
    allocate (p(d1, d2))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1, 1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_i32_2

  subroutine rd_get_i64_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int64), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_prepare(self, tname, dt_i64, 'integer(int64)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_i64_1

  subroutine rd_get_i64_2(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int64), pointer, intent(out) :: p(:, :)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    integer(int64) :: d1, d2
    p => null()
    call rd_prepare(self, tname, dt_i64, 'integer(int64)', i, stat, msg)
    if (stat /= st_ok) return
    call check_rank2(self, i, stat, msg)
    if (stat /= st_ok) return
    call fortran_dims2(self%t(i), d1, d2)
    allocate (p(d1, d2))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1, 1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_i64_2

  subroutine rd_get_u8_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int8), pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    p => null()
    call rd_prepare(self, tname, dt_u8, 'integer(int8) (U8)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_u8_1

  subroutine rd_get_u8_2(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    integer(int8), pointer, intent(out) :: p(:, :)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer :: i
    integer(int64) :: d1, d2
    p => null()
    call rd_prepare(self, tname, dt_u8, 'integer(int8) (U8)', i, stat, msg)
    if (stat /= st_ok) return
    call check_rank2(self, i, stat, msg)
    if (stat /= st_ok) return
    call fortran_dims2(self%t(i), d1, d2)
    allocate (p(d1, d2))
    if (self%t(i)%nbytes > 0) then
      call read_at(self, i, c_loc(p(1, 1)), stat, msg)
      if (stat /= st_ok) then
        deallocate (p)
        p => null()
      end if
    end if
  end subroutine rd_get_u8_2

  ! BOOL needs one extra pass: on disk it is 1 byte (0 or 1) while `logical` takes
  ! 4 bytes in gfortran. This is also where an invalid byte (say 7) becomes an
  ! ERROR instead of a silent `.true.` — the official implementation rejects it too.
  subroutine rd_get_bool_1(self, tname, p, stat, msg)
    class(st_reader), intent(in) :: self
    character(*), intent(in) :: tname
    logical, pointer, intent(out) :: p(:)
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    integer(int8), allocatable, target :: raw(:)
    integer :: i
    integer(int64) :: k
    p => null()
    call rd_prepare(self, tname, dt_bool, 'logical (BOOL)', i, stat, msg)
    if (stat /= st_ok) return
    allocate (p(self%t(i)%nelem))
    if (self%t(i)%nbytes == 0) return
    allocate (raw(self%t(i)%nelem))
    call read_at(self, i, c_loc(raw(1)), stat, msg)
    if (stat /= st_ok) then
      deallocate (p, raw)
      p => null()
      return
    end if
    do k = 1, int(self%t(i)%nelem, int64)
      if (raw(k) /= 0_int8 .and. raw(k) /= 1_int8) then
        stat = st_err_value
        msg = 'BOOLEAN tensor '''//self%t(i)%name//''' has byte value '// &
              trim(i2s(int(raw(k), int64)))//' at element '//trim(i2s(k))// &
              ' (only 0 and 1 are valid)'
        deallocate (p, raw)
        p => null()
        return
      end if
      p(k) = (raw(k) == 1_int8)
    end do
    deallocate (raw)
    stat = st_ok
    msg = ''
  end subroutine rd_get_bool_1

end module safetensors
