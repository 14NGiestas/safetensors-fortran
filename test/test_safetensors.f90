! test_safetensors.f90 - test suite for the safetensors Fortran library.
!
!   fpm test        -> prints "PASS n / FAIL m" and exits non-zero on any failure
!
! What is covered and why (each block says what would break silently otherwise):
!   1. in-memory round-trip of every supported dtype/rank, including the
!      degenerate shapes [0] and [1,0] -- an off-by-one in the offset arithmetic
!      shows up here;
!   2. BYTE-FOR-BYTE parity against test/fixtures/parity_oracle.safetensors, which
!      was produced by the official implementation (or by the pure-Python oracle
!      when the official one is not installed);
!   3. reading fixtures produced outside Fortran (pure oracle and official library);
!   4. NaN/+Inf/-Inf/-0.0 preserved bit by bit -- compared as BYTES, never as floats;
!   5. metadata with accents, JSON escapes and an empty key;
!   6. malformed headers rejected with a stat and a message that says what is
!      wrong (13 cases; the task asked for at least 6);
!   7. a 1024x1024 tensor, timed, to catch quadratic copying;
!   8. cross-language round-trip: Fortran -> Python (the oracle validates our file)
!      and Python -> Fortran (we read the oracle's file).
!
program test_safetensors
  use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
  use safetensors, only: st_writer, st_reader, st_ok, st_version, &
    st_err_truncated, st_err_header_size, st_err_json, st_err_schema, st_err_dtype, &
    st_err_offsets, st_err_missing, st_err_type_mismatch, st_err_not_open, &
    st_err_range, st_err_value, st_max_header_bytes
  implicit none

  integer :: npass = 0, nfail = 0
  character(len=:), allocatable :: fixdir, outdir

  fixdir = resolve_dir([character(len=64) :: 'test/fixtures', '../test/fixtures', 'fixtures'])
  outdir = 'build/st_test'
  call execute_command_line('mkdir -p '//outdir)

  write (*, '(A)') '== safetensors (Fortran) '//trim(st_version)//' test suite =='
  write (*, '(A,A)') 'fixtures dir: ', fixdir
  write (*, '(A,A)') 'output dir  : ', outdir

  call test_roundtrip_all_dtypes()
  call test_degenerate_shapes()
  call test_byte_parity_vs_oracle()
  call test_read_oracle_basic()
  call test_read_oracle_mixed()
  call test_read_official_fixture()
  call test_specials_bitwise()
  call test_metadata_utf8_and_empty_key()
  call test_malformed()
  call test_rank3_and_get2d()
  call test_large_tensor()
  call test_cross_language()

  write (*, '(A)') ''
  write (*, '(A,I0,A,I0)') 'PASS ', npass, ' / FAIL ', nfail
  if (nfail /= 0) then
    write (*, '(A)') 'RESULT: FAIL'
    stop 1
  end if
  write (*, '(A)') 'RESULT: OK'
  stop 0

contains

  ! ------------------------------------------------------------ test utilities
  subroutine ok(cond, label)
    logical, intent(in) :: cond
    character(*), intent(in) :: label
    if (cond) then
      npass = npass + 1
      write (*, '(A)') '  PASS  '//label
    else
      nfail = nfail + 1
      write (*, '(A)') '  FAIL  '//label
    end if
  end subroutine ok

  subroutine ok_stat(got, want, needle, label, msg)
    integer, intent(in) :: got, want
    character(*), intent(in) :: needle, label, msg
    logical :: has
    has = index(msg, needle) > 0
    if (got == want .and. has) then
      npass = npass + 1
      write (*, '(A)') '  PASS  '//label//'  [stat='//trim(i2s(int(got, int64)))//', msg="'// &
        trim(first_line(msg))//'"]'
    else
      nfail = nfail + 1
      write (*, '(A,I0,A,L1,A)') '  FAIL  '//label//': expected stat=', want, ' got=', got, &
        ' msg_has_needle=', has, ' msg="'//trim(first_line(msg))//'"'
    end if
  end subroutine ok_stat

  subroutine section(title)
    character(*), intent(in) :: title
    write (*, '(A)') ''
    write (*, '(A)') '-- '//title
  end subroutine section

  function first_line(s) result(t)
    character(*), intent(in) :: s
    character(len=:), allocatable :: t
    integer :: p
    p = index(s, achar(10))
    if (p > 0) then
      t = s(1:p - 1)
    else
      t = s
    end if
    if (len(t) > 160) t = t(1:160)//'...'
  end function first_line

  pure function i2s(v) result(s)
    integer(int64), intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i2s

  function fmt1(x) result(s)
    real(real64), intent(in) :: x
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(F8.3)') x
    s = trim(adjustl(b))
  end function fmt1

  function resolve_dir(cands) result(d)
    character(*), intent(in) :: cands(:)
    character(len=:), allocatable :: d
    logical :: ex
    integer :: i
    d = ''
    do i = 1, size(cands)
      inquire (file=trim(cands(i))//'/parity_oracle.safetensors', exist=ex)
      if (ex) then
        d = trim(cands(i))
        return
      end if
      inquire (file=trim(cands(i))//'/oracle_basic.safetensors', exist=ex)
      if (ex) then
        d = trim(cands(i))
        return
      end if
    end do
    d = 'test/fixtures'
  end function resolve_dir

  function fixture(name) result(p)
    character(*), intent(in) :: name
    character(len=:), allocatable :: p
    p = fixdir//'/'//name
  end function fixture

  function out(name) result(p)
    character(*), intent(in) :: name
    character(len=:), allocatable :: p
    p = outdir//'/'//name
  end function out

  ! Bytes of a character string as they sit in memory (used to build raw files).
  pure function bytes_of(s) result(b)
    character(*), intent(in) :: s
    integer(int8) :: b(len(s))
    integer :: i
    do i = 1, len(s)
      b(i) = int(iachar(s(i:i)), int8)
    end do
  end function bytes_of

  pure function le64(v) result(b)
    integer(int64), intent(in) :: v
    integer(int8) :: b(8)
    integer(int64) :: x
    integer :: i
    x = v
    do i = 1, 8
      b(i) = int(iand(x, 255_int64), int8)
      x = ishft(x, -8)
    end do
  end function le64

  subroutine write_bytes(path, b, stat)
    character(*), intent(in) :: path
    integer(int8), intent(in) :: b(:)
    integer, intent(out) :: stat
    integer :: u
    open (newunit=u, file=path, access='stream', form='unformatted', status='replace', &
          iostat=stat)
    if (stat /= 0) return
    if (size(b) > 0) write (u, iostat=stat) b
    close (u)
  end subroutine write_bytes

  ! A "raw" file: length prefix + header text + payload, bypassing the library.
  subroutine raw_file(path, hdr, payload, stat)
    character(*), intent(in) :: path, hdr
    integer(int8), intent(in) :: payload(:)
    integer, intent(out) :: stat
    integer(int8), allocatable :: all(:)
    integer(int8) :: hb(len(hdr))
    integer(int64) :: n
    hb = bytes_of(hdr)
    n = int(len(hdr), int64) + int(size(payload), int64)
    allocate (all(8 + n))
    all(1:8) = le64(int(len(hdr), int64))
    if (len(hdr) > 0) all(9:8 + len(hdr)) = hb
    if (size(payload) > 0) all(int(9 + len(hdr), int64):8 + n) = payload
    call write_bytes(path, all, stat)
  end subroutine raw_file

  ! Igualdade byte a byte de dois arrays de real32 (documenta que a leitura 2-D
  ! preserva a MEMÓRIA e inverte as extensões).
  pure function same_bytes(a, b) result(r)
    real(real32), intent(in) :: a(:), b(:)
    logical :: r
    integer(int8), allocatable :: xa(:), xb(:)
    xa = transfer(a, 0_int8, int(size(a, kind=int64), int64)*4_int64)
    xb = transfer(b, 0_int8, int(size(b, kind=int64), int64)*4_int64)
    r = size(xa) == size(xb)
    if (r) r = all(xa == xb)
  end function same_bytes

  subroutine read_file_bytes(path, b, nb, stat)
    character(*), intent(in) :: path
    integer(int8), allocatable, intent(out) :: b(:)
    integer(int64), intent(out) :: nb
    integer, intent(out) :: stat
    integer :: u, sz
    inquire (file=path, size=sz, iostat=stat)
    if (stat /= 0 .or. sz < 0) then
      allocate (b(0))
      nb = 0_int64
      stat = 1
      return
    end if
    allocate (b(sz))
    nb = int(sz, int64)
    stat = 0
    open (newunit=u, file=path, access='stream', form='unformatted', status='old', iostat=stat)
    if (stat /= 0) return
    if (sz > 0) read (u, iostat=stat) b
    close (u)
  end subroutine read_file_bytes

  ! ==================================================================== 1
  subroutine test_roundtrip_all_dtypes()
    type(st_writer) :: w
    type(st_reader) :: r
    real(real32) :: a32(3) = [1.0, -2.5, 3.25]
    real(real32) :: m32(2, 3)
    real(real64) :: a64(2) = [1.5d0, -2.5d0]
    integer(int32) :: a32i(3) = [1, -2, 3]
    integer(int32) :: m32i(2, 2)
    integer(int64) :: a64i(2) = [-1_int64, 1099511627776_int64]
    integer(int8) :: a8(4) = [0_int8, 1_int8, 127_int8, -1_int8]
    logical :: ab(3) = [.true., .false., .true.]
    real(real32), pointer :: p1(:) => null(), pm(:, :) => null()
    real(real64), pointer :: q1(:) => null()
    integer(int32), pointer :: j1(:) => null(), jm(:, :) => null()
    integer(int64), pointer :: k1(:) => null()
    integer(int8), pointer :: u1(:) => null()
    logical, pointer :: b1(:) => null()
    integer(int64), allocatable :: shp(:)
    character(len=:), allocatable :: msg, dtype
    integer :: stat
    integer(int64) :: nb

    call section('1. round-trip in memory (every supported dtype, 1-D and 2-D)')
    m32 = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    m32i = reshape([1, 2, 3, 4], [2, 2])

    call w%init()
    call w%set_meta('format_version', '1')
    call w%set('a32', a32)
    call w%set('m32', m32)
    call w%set('a64', a64)
    call w%set('a32i', a32i)
    call w%set('m32i', m32i)
    call w%set('a64i', a64i)
    call w%set('a8', a8)
    call w%set('ab', ab)
    call w%write(out('roundtrip.safetensors'), stat, msg)
    call ok(stat == st_ok, 'write of 8 tensors returns st_ok')
    if (stat /= st_ok) return
    call ok(w%n_tensors() == 8, 'writer knows it holds 8 tensors')
    call ok(w%payload_size() == 4*3 + 4*6 + 8*2 + 4*3 + 4*4 + 8*2 + 4 + 3, &
            'payload_size is the sum of the tensor sizes (43 bytes)')

    call r%open(out('roundtrip.safetensors'), stat, msg)
    call ok(stat == st_ok, 'open of our own file returns st_ok')
    if (stat /= st_ok) return
    call ok(r%n_tensors() == 8, 'n_tensors == 8')

    call r%get('a32', p1, stat, msg)
    call ok(stat == st_ok .and. all(p1 == a32), 'F32 1-D values identical')
    call r%get('m32', pm, stat, msg)
    call ok(stat == st_ok .and. size(pm, 1) == 3 .and. size(pm, 2) == 2 .and. &
            all(pm == reshape(m32, [3, 2])) .and. same_bytes(reshape(pm, [6]), reshape(m32, [6])), &
            'F32 2-D: identical bytes, extents reversed (2,3)->(3,2)')
    call r%get('a64', q1, stat, msg)
    call ok(stat == st_ok .and. all(q1 == a64), 'F64 1-D values identical')
    call r%get('a32i', j1, stat, msg)
    call ok(stat == st_ok .and. all(j1 == a32i), 'I32 1-D values identical')
    call r%get('m32i', jm, stat, msg)
    call ok(stat == st_ok .and. all(jm == reshape(m32i, [2, 2])), &
            'I32 2-D: identical values, extents reversed')
    call r%get('a64i', k1, stat, msg)
    call ok(stat == st_ok .and. all(k1 == a64i), 'I64 1-D values identical')
    call r%get('a8', u1, stat, msg)
    call ok(stat == st_ok .and. all(u1 == a8), 'U8 bytes identical')
    call r%get('ab', b1, stat, msg)
    call ok(stat == st_ok .and. all(b1 .eqv. ab), 'BOOL values identical')

    call r%dtype('m32', dtype, stat, msg)
    call r%shape('m32', shp, stat, msg)
    call r%nbytes('m32', nb, stat, msg)
    call ok(dtype == 'F32' .and. size(shp) == 2 .and. shp(1) == 2 .and. shp(2) == 3 .and. &
            nb == 24_int64, 'dtype/shape/nbytes answerable without reading the tensor')

    ! Failure paths: same pointers never reused, so nothing leaks here.
    call r%get('a64', p1, stat, msg)
    call ok_stat(stat, st_err_type_mismatch, 'has dtype F64', &
                 'asking for F64 as real(real32) fails cleanly', msg)
    call r%get('nope', p1, stat, msg)
    call ok_stat(stat, st_err_missing, 'not found', 'missing tensor reports st_err_missing', msg)
    call r%dtype('nope', dtype, stat, msg)
    call ok(stat == st_err_missing, 'dtype query for a missing tensor fails the same way')
  end subroutine test_roundtrip_all_dtypes

  ! ==================================================================== 2
  subroutine test_degenerate_shapes()
    type(st_writer) :: w
    type(st_reader) :: r
    real(real32) :: e1(0)
    real(real32) :: e2(1, 0)
    real(real32), pointer :: p1(:) => null()
    real(real32), pointer :: p2(:, :) => null()
    integer(int64), allocatable :: shp(:)
    integer(int64) :: nb, b0, b1
    character(len=:), allocatable :: msg
    integer :: stat

    call section('2. degenerate shapes: [0] and [1,0]')
    call w%init()
    call w%set('empty1d', e1)
    call w%set('empty2d', e2)
    call w%write(out('empty.safetensors'), stat, msg)
    call ok(stat == st_ok, 'writing zero-element tensors returns st_ok')
    call r%open(out('empty.safetensors'), stat, msg)
    call ok(stat == st_ok, 'a file whose whole payload is empty opens')
    if (stat /= st_ok) return
    call r%nbytes('empty1d', nb, stat, msg)
    call r%tensor_offsets('empty1d', b0, b1, stat, msg)
    call r%shape('empty1d', shp, stat, msg)
    call ok(nb == 0_int64 .and. b0 == b1 .and. size(shp) == 1 .and. shp(1) == 0, &
            '[0] tensor: 0 bytes, collapsed offsets, shape [0]')
    call r%get('empty1d', p1, stat, msg)
    call ok(stat == st_ok .and. size(p1) == 0, 'get on a [0] tensor returns a 0-size pointer')
    call r%get('empty2d', p2, stat, msg)
    call r%shape('empty2d', shp, stat, msg)
    call ok(stat == st_ok .and. size(p2, 1) == 0 .and. size(p2, 2) == 1 .and. shp(1) == 1 .and. &
            shp(2) == 0, 'get on a [1,0] tensor returns (0,1) with shape [1,0]')
    call ok(r%buffer_size() == 0_int64, 'buffer_size is 0 for this file')
  end subroutine test_degenerate_shapes

  ! ==================================================================== 3
  subroutine test_byte_parity_vs_oracle()
    type(st_writer) :: w
    real(real32) :: a0(4) = [0.0, 0.5, 1.0, 2.25]
    real(real32) :: lq(2, 3) = reshape([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])
    real(real32) :: wte(5) = [-1.5, 0.0, 1024.0, -0.125, 65536.0]
    real(real32) :: z(1) = [42.0]
    real(real32) :: one(1) = [1.0]
    integer(int8), allocatable :: mine(:), theirs(:), image(:)
    character(len=:), allocatable :: msg
    integer :: stat
    integer(int64) :: nb
    logical :: same

    call section('3. byte-for-byte parity with the external oracle')
    ! Insertion order matches the official writer's order (alignment desc, name
    ! asc). This writer keeps insertion order, which the README documents.
    call w%init()
    call w%set_meta('bpb', '1.59994')
    call w%set('a0', a0)
    call w%set('l10.q', lq)
    call w%set('wte', wte)
    call w%set('z', z)
    call w%write(out('parity_lib.safetensors'), stat, msg)
    call ok(stat == st_ok, 'wrote parity_lib.safetensors')
    call read_file_bytes(out('parity_lib.safetensors'), mine, nb, stat)
    call ok(stat == 0 .and. nb > 0, 'read our own bytes back')
    call read_file_bytes(fixture('parity_oracle.safetensors'), theirs, nb, stat)
    call ok(stat == 0 .and. nb > 0, 'read the oracle fixture bytes')
    same = (size(mine) == size(theirs))
    if (same) same = all(mine == theirs)
    call ok(same, 'cmp: parity_lib.safetensors == parity_oracle.safetensors ('// &
            trim(i2s(int(size(mine), int64)))//' bytes, '//trim(i2s(int(size(theirs), int64)))// &
            ' in the oracle)')

    call w%to_bytes(image, stat, msg)
    call ok(stat == st_ok .and. size(image) == size(theirs) .and. all(image == theirs), &
            'the in-memory to_bytes() image is byte-identical to the oracle file')

    ! Metadata escaping torture, single key (deterministic in every writer).
    call w%init()
    call w%set_meta('k', 'tab'//achar(9)//' nl'//achar(10)//' cr'//achar(13)//' bs\ quote"'// &
                    ' slash/ del'//achar(127)//' nul-free acentuação ção')
    call w%set('a', one)
    call w%write(out('escapes_lib.safetensors'), stat, msg)
    call ok(stat == st_ok, 'wrote escapes_lib.safetensors')
    call read_file_bytes(out('escapes_lib.safetensors'), mine, nb, stat)
    call read_file_bytes(fixture('escapes_oracle.safetensors'), theirs, nb, stat)
    same = (size(mine) == size(theirs))
    if (same) same = all(mine == theirs)
    call ok(same, 'cmp: JSON escaping (\\t \\n \\r \\" \\\\, DEL, UTF-8) matches the oracle '// &
            'byte for byte')
  end subroutine test_byte_parity_vs_oracle

  ! ==================================================================== 4
  subroutine test_read_oracle_basic()
    type(st_reader) :: r
    real(real32), pointer :: p1(:) => null(), p2(:, :) => null()
    real(real32), pointer :: e1(:) => null()
    integer(int64), allocatable :: shp(:)
    character(len=:), allocatable :: msg, val, key
    integer :: stat, i
    logical :: found, saw_utf8, saw_empty

    call section('4. reading the oracle fixture (shapes, empty tensors, metadata)')
    call r%open(fixture('oracle_basic.safetensors'), stat, msg)
    call ok(stat == st_ok, 'oracle_basic opens')
    if (stat /= st_ok) return
    call ok(r%n_tensors() == 4, 'oracle_basic has 4 tensors')
    call r%get('wte', p2, stat, msg)
    call ok(stat == st_ok .and. size(p2, 1) == 6 .and. size(p2, 2) == 4 .and. p2(1, 1) == 0.0 .and. &
            p2(6, 4) == 23.0, 'wte [4,6] arange reads back with the right values')
    call r%get('l3.q', p1, stat, msg)
    call ok(stat == st_ok .and. size(p1) == 12 .and. p1(1) == 0.5 .and. p1(12) == 11.5, &
            'l3.q [12] values match')
    call r%get('empty1d', e1, stat, msg)
    call ok(stat == st_ok .and. size(e1) == 0, 'empty1d is empty')

    call r%meta('bpb', val, found)
    call ok(found .and. val == '1.59994', 'meta bpb round-trips')
    call r%meta('caracterização', val, found)
    call ok(found .and. val == 'acentuação e ção', 'UTF-8 metadata value round-trips byte-wise')
    call r%meta('', val, found)
    call ok(found .and. val == 'empty-key', 'empty metadata key is allowed and found')
    call r%meta('nao_existe', val, found)
    call ok(.not. found .and. len(val) == 0, 'absent key reports found=.false.')
    saw_utf8 = .false.
    saw_empty = .false.
    do i = 1, r%n_meta()
      call r%meta_key(i, key, stat, msg)
      if (key == 'caracterização') saw_utf8 = .true.
      if (len(key) == 0) saw_empty = .true.
    end do
    call ok(saw_utf8 .and. saw_empty .and. r%n_meta() == 4, &
            'metadata enumeration has 4 keys, including the UTF-8 and the empty one')
    call r%shape('wte', shp, stat, msg)
    call ok(size(shp) == 2 .and. shp(1) == 4 .and. shp(2) == 6, &
            'shape is reported in C order, as numpy wrote it')
  end subroutine test_read_oracle_basic

  ! ==================================================================== 5
  subroutine test_read_oracle_mixed()
    type(st_reader) :: r
    real(real32), pointer :: f32p(:) => null()
    real(real64), pointer :: f64p(:) => null()
    integer(int32), pointer :: i32p(:) => null()
    integer(int64), pointer :: i64p(:) => null()
    integer(int8), pointer :: u8p(:) => null()
    logical, pointer :: bp(:) => null()
    integer(int8), pointer :: raw(:) => null()
    character(len=:), allocatable :: msg, dtype
    integer :: stat

    call section('5. reading a fixture with every supported dtype')
    call r%open(fixture('oracle_mixed.safetensors'), stat, msg)
    call ok(stat == st_ok, 'oracle_mixed opens')
    if (stat /= st_ok) return
    call r%get('f32', f32p, stat, msg)
    call ok(stat == st_ok .and. all(f32p == [1.0, 2.0, 3.0]), 'F32')
    call r%get('f64', f64p, stat, msg)
    call ok(stat == st_ok .and. all(f64p == [1.5d0, -2.5d0]), 'F64')
    call r%get('i32', i32p, stat, msg)
    call ok(stat == st_ok .and. all(i32p == [1, -2, 3]), 'I32')
    call r%get('i64', i64p, stat, msg)
    call ok(stat == st_ok .and. i64p(1) == -1_int64 .and. i64p(2) == 1099511627776_int64, 'I64')
    call r%get('u8', u8p, stat, msg)
    call ok(stat == st_ok .and. u8p(1) == 0_int8 .and. u8p(2) == 1_int8 .and. &
            u8p(3) == 127_int8 .and. u8p(4) == -1_int8, 'U8 gives back raw bytes (255 -> -1_int8)')
    call r%get('bool', bp, stat, msg)
    call ok(stat == st_ok .and. bp(1) .and. (.not. bp(2)) .and. bp(3), 'BOOL')
    call r%dtype('i32', dtype, stat, msg)
    call ok(dtype == 'I32', 'dtype of i32 is I32')
    call r%get_raw('f64', raw, stat, msg)
    call ok(stat == st_ok .and. size(raw) == 16, 'get_raw returns the 16 raw bytes of an F64 tensor')
  end subroutine test_read_oracle_mixed

  ! ==================================================================== 6
  subroutine test_read_official_fixture()
    type(st_reader) :: r
    real(real32), pointer :: a(:, :) => null()
    real(real64), pointer :: b(:) => null()
    integer(int64), pointer :: c(:) => null()
    logical, pointer :: d(:) => null()
    integer(int8), pointer :: e(:) => null()
    character(len=:), allocatable :: msg, val
    integer :: stat
    integer(int64) :: off_a, off_b
    logical :: found, ex

    call section('6. reading a file written by the OFFICIAL Rust/Python library')
    inquire (file=fixture('oracle_official.safetensors'), exist=ex)
    if (.not. ex) then
      call ok(.true., 'SKIP: oracle_official.safetensors was not generated (the official '// &
              'safetensors package was not importable when the fixtures were made)')
      return
    end if
    call r%open(fixture('oracle_official.safetensors'), stat, msg)
    call ok(stat == st_ok, 'the official file opens')
    if (stat /= st_ok) return
    call r%get('a', a, stat, msg)
    call ok(stat == st_ok .and. size(a, 1) == 3 .and. size(a, 2) == 2 .and. a(1, 1) == 0.0 .and. &
            a(3, 1) == 2.0, 'official F32 [2,3] arange reads back')
    call r%get('b', b, stat, msg)
    call ok(stat == st_ok .and. all(b == [1.5d0, -2.5d0]), 'official F64')
    call r%get('c', c, stat, msg)
    call ok(stat == st_ok .and. c(1) == -1_int64 .and. c(2) == 1073741824_int64, 'official I64')
    call r%get('d', d, stat, msg)
    call ok(stat == st_ok .and. d(1) .and. (.not. d(2)), 'official BOOL')
    call r%get('e', e, stat, msg)
    call ok(stat == st_ok .and. e(1) == 0_int8 .and. e(2) == -56_int8, 'official U8 (200 -> -56_int8)')
    call r%meta('caracterização', val, found)
    call ok(found .and. val == 'acentuação', 'official UTF-8 metadata value reads back')
    call r%meta('bpb', val, found)
    call ok(found .and. val == '1.59994', 'official metadata bpb reads back')
    call r%dtype('c', val, stat, msg)
    call ok(val == 'I64', 'mixed dtypes in one official file are handled')
    ! the official writer lays tensors out by descending dtype alignment
    call r%tensor_name(1, val, stat, msg)
    call r%tensor_offsets(val, off_a, off_b, stat, msg)
    call ok(off_a == 0_int64, 'first tensor in the buffer is the I64 one (alignment order)')
  end subroutine test_read_official_fixture

  ! ==================================================================== 7
  subroutine test_specials_bitwise()
    type(st_writer) :: w
    type(st_reader) :: r
    integer(int32) :: bits(6)
    real(real32) :: v(6)
    real(real32), pointer :: p(:) => null()
    integer(int8), pointer :: raw(:) => null()
    integer(int8) :: want(24)
    character(len=:), allocatable :: msg, val
    integer :: stat
    logical :: found

    call section('7. NaN / +Inf / -Inf / -0.0 survive (compared as BYTES, never as floats)')
    bits = [int(z'7FC00000'), int(z'7F800000'), int(z'FF800000'), int(z'80000000'), &
            int(z'00000000'), int(z'3F800000')]
    v = transfer(bits, v)
    want = transfer(v, want)
    call ok(.not. (v(1) == v(1)), 'sanity: element 1 really is NaN')

    call w%init()
    call w%set('specials', v)
    call w%write(out('specials.safetensors'), stat, msg)
    call ok(stat == st_ok, 'wrote the special-values tensor')
    call r%open(out('specials.safetensors'), stat, msg)
    call r%get('specials', p, stat, msg)
    call ok(stat == st_ok, 'read it back')
    if (stat /= st_ok) return
    call r%get_raw('specials', raw, stat, msg)
    call ok(stat == st_ok .and. size(raw) == 24 .and. all(raw == want), &
            'all 24 payload bytes identical: NaN/Inf/-0.0 bit patterns preserved')
    block
      integer(int32) :: bits4
      bits4 = transfer(p(4), bits4)                  ! sem 1/0.0: isso levantaria
      call ok(p(4) == 0.0 .and. bits4 == int(z'80000000', int32), &
              '-0.0 keeps its sign bit (0x80000000), not just == 0.0')
    end block

    call r%open(fixture('oracle_nan.safetensors'), stat, msg)
    call ok(stat == st_ok, 'oracle_nan (built in Python from uint32 bit patterns) opens')
    if (stat /= st_ok) return
    call r%get_raw('specials', raw, stat, msg)
    call ok(stat == st_ok .and. all(raw == want), 'oracle_nan payload bytes == our own bytes')
    call r%meta('escapes', val, found)
    call ok(found .and. index(val, 'tab'//achar(9)//'here') > 0 .and. index(val, '"quote"') > 0 .and. &
            index(val, 'back\slash') > 0 .and. index(val, 'acentuação') > 0, &
            'escaped metadata decodes back to the original bytes')
  end subroutine test_specials_bitwise

  ! ==================================================================== 8
  subroutine test_metadata_utf8_and_empty_key()
    type(st_writer) :: w
    type(st_reader) :: r
    real(real32) :: a(2) = [1.0, 2.0]
    integer(int8), allocatable :: b(:)
    character(len=:), allocatable :: msg, val, hdr
    integer :: stat, i
    integer(int64) :: nb
    logical :: found

    call section('8. metadata: UTF-8, byte-wise sorted keys, empty key, overwrite')
    call w%init()
    call w%set_meta('zeta', 'último')            ! inserted out of order on purpose
    call w%set_meta('alpha', 'primeiro')
    call w%set_meta('caracterização', 'acentuação e ção')
    call w%set_meta('', 'empty-key')
    call w%set_meta('alpha', 'sobrescrito')      ! same key again: replaces
    call w%set('a', a)
    call w%write(out('meta.safetensors'), stat, msg)
    call ok(stat == st_ok, 'write with 4 metadata keys (one repeated)')
    call read_file_bytes(out('meta.safetensors'), b, nb, stat)
    hdr = ''
    do i = 9, min(int(nb), 220)
      hdr = hdr//achar(iand(int(b(i), int32), 255))
    end do
    call ok(index(hdr, '{"__metadata__":{"":"empty-key","alpha":"sobrescrito",'// &
            '"caracterização":"acentuação e ção","zeta":"último"}') == 1, &
            'raw header bytes: metadata keys sorted byte-wise, UTF-8 emitted raw')
    call r%open(out('meta.safetensors'), stat, msg)
    call r%meta('alpha', val, found)
    call ok(found .and. val == 'sobrescrito', 'setting the same key overwrites it')
    call r%meta('caracterização', val, found)
    call ok(found .and. val == 'acentuação e ção', 'accented value survives')
    call r%meta('', val, found)
    call ok(found .and. val == 'empty-key', 'empty key survives (documented: allowed)')
    call ok(r%n_meta() == 4, 'n_meta == 4')
  end subroutine test_metadata_utf8_and_empty_key

  ! ==================================================================== 9
  subroutine test_malformed()
    type(st_reader) :: r
    character(len=:), allocatable :: msg
    integer :: stat, i
    integer(int8) :: pay(16)
    integer(int8) :: b16(16)
    integer(int64), allocatable :: shp_scratch(:)
    logical, pointer :: bool_scratch(:) => null()

    pay = [(int(i, int8), i=1, 16)]
    call section('9..18. malformed input must be rejected with stat + a real message')

    ! (1) fewer than 8 bytes
    call write_bytes(out('bad_short.bin'), bytes_of('1234'), stat)
    call r%open(out('bad_short.bin'), stat, msg)
    call ok_stat(stat, st_err_truncated, '8 bytes', 'file shorter than the length prefix', msg)

    ! (2) header length 0
    call write_bytes(out('bad_zero.bin'), le64(0_int64), stat)
    call r%open(out('bad_zero.bin'), stat, msg)
    call ok_stat(stat, st_err_header_size, 'header length is 0', 'zero header length', msg)

    ! (3) header length beyond EOF
    b16 = 0_int8
    b16(1:8) = le64(500_int64)
    b16(9:13) = bytes_of('{"a":')
    call write_bytes(out('bad_trunc.bin'), b16, stat)
    call r%open(out('bad_trunc.bin'), stat, msg)
    call ok_stat(stat, st_err_truncated, 'truncated', 'header length beyond EOF', msg)

    ! (4) header length above the format's 100MB limit
    call write_bytes(out('bad_huge.bin'), le64(st_max_header_bytes + 1_int64), stat)
    call r%open(out('bad_huge.bin'), stat, msg)
    call ok_stat(stat, st_err_header_size, 'above the format limit', 'header length over 100MB', msg)

    ! (5) header length with the sign bit set: not a safetensors file at all
    b16 = 0_int8
    b16(8) = int(z'80', int8)                ! 0x8000000000000000 => negative
    call write_bytes(out('bad_sign.bin'), b16, stat)
    call r%open(out('bad_sign.bin'), stat, msg)
    call ok_stat(stat, st_err_header_size, 'negative', 'header length with the sign bit set', msg)

    ! (6) invalid JSON (a value is missing after ':')
    call raw_file(out('bad_json.bin'), '{"a":}', pay, stat)
    call r%open(out('bad_json.bin'), stat, msg)
    call ok_stat(stat, st_err_json, 'invalid header JSON', 'invalid JSON object', msg)

    ! (7) valid JSON but not an object
    call raw_file(out('bad_notobj.bin'), '[1,2,3]        ', pay, stat)
    call r%open(out('bad_notobj.bin'), stat, msg)
    call ok_stat(stat, st_err_schema, 'JSON object', 'header that is not an object', msg)

    ! (8) unknown dtype
    call raw_file(out('bad_dtype.bin'), &
                  '{"a":{"dtype":"F99","shape":[1],"data_offsets":[0,4]}}', pay(1:4), stat)
    call r%open(out('bad_dtype.bin'), stat, msg)
    call ok_stat(stat, st_err_dtype, 'unknown dtype', 'unknown dtype', msg)

    ! (9) end-begin inconsistent with shape x itemsize
    call raw_file(out('bad_size.bin'), &
                  '{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}', pay(1:8), stat)
    call r%open(out('bad_size.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'needs 16 bytes', 'offsets inconsistent with shape', msg)

    ! (10) overlapping offsets
    call raw_file(out('bad_overlap.bin'), &
                  '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'// &
                  '"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}', pay(1:12), stat)
    call r%open(out('bad_overlap.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'overlapping', 'overlapping data_offsets', msg)

    ! (11) hole in the buffer
    call raw_file(out('bad_hole.bin'), &
                  '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'// &
                  '"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}', pay(1:12), stat)
    call r%open(out('bad_hole.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'holes', 'hole in the byte buffer', msg)

    ! (12) trailing bytes after the last tensor
    call raw_file(out('bad_trail.bin'), &
                  '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', pay(1:8), stat)
    call r%open(out('bad_trail.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'trailing', 'trailing bytes after the buffer', msg)

    ! (13) duplicate key in the header
    call raw_file(out('bad_dup.bin'), &
                  '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'// &
                  '"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}', pay(1:8), stat)
    call r%open(out('bad_dup.bin'), stat, msg)
    call ok_stat(stat, st_err_json, 'duplicate key', 'duplicate key in the header', msg)

    ! (14) metadata value that is not a string
    call raw_file(out('bad_meta.bin'), &
                  '{"__metadata__":{"k":5},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', &
                  pay(1:4), stat)
    call r%open(out('bad_meta.bin'), stat, msg)
    call ok_stat(stat, st_err_schema, 'must be a string', 'non-string metadata value', msg)

    ! (15) negative data offset
    call raw_file(out('bad_neg.bin'), &
                  '{"a":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}', pay(1:4), stat)
    call r%open(out('bad_neg.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'negative', 'negative data offset', msg)

    ! (16) sub-byte dtype that is not byte aligned (F4 x 3 = 12 bits)
    call raw_file(out('bad_subbyte.bin'), &
                  '{"a":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}', pay(1:2), stat)
    call r%open(out('bad_subbyte.bin'), stat, msg)
    call ok_stat(stat, st_err_offsets, 'whole number of bytes', 'sub-byte dtype misaligned', msg)

    ! (17) float in the JSON (the format only uses integers)
    call raw_file(out('bad_float.bin'), &
                  '{"a":{"dtype":"F32","shape":[2.5],"data_offsets":[0,8]}}', pay(1:8), stat)
    call r%open(out('bad_float.bin'), stat, msg)
    call ok_stat(stat, st_err_json, 'non-integer', 'float in the JSON is rejected', msg)

    ! (18) truncated JSON
    call raw_file(out('bad_str.bin'), '{"a":    ', pay(1:4), stat)
    call r%open(out('bad_str.bin'), stat, msg)
    call ok_stat(stat, st_err_json, 'invalid header JSON', 'truncated JSON', msg)

    ! (19) a closed reader must say so instead of crashing
    call r%close()
    call r%shape('a', shp_scratch, stat, msg)
    call ok_stat(stat, st_err_not_open, 'not open', 'using a closed reader is an error', msg)

    ! (20) BOOL tensor with a byte outside {0,1}
    call raw_file(out('bad_bool.bin'), &
                  '{"a":{"dtype":"BOOL","shape":[2],"data_offsets":[0,2]}}', [1_int8, 7_int8], stat)
    call r%open(out('bad_bool.bin'), stat, msg)
    call r%get('a', bool_scratch, stat, msg)
    call ok_stat(stat, st_err_value, 'only 0 and 1', 'BOOL byte other than 0/1', msg)
  end subroutine test_malformed

  ! ==================================================================== 10
  subroutine test_rank3_and_get2d()
    type(st_reader) :: r
    real(real32), pointer :: flat(:) => null()
    real(real32), pointer :: m2(:, :) => null()
    integer(int64), allocatable :: shp(:)
    character(len=:), allocatable :: msg
    integer :: stat, i
    integer(int8) :: pay(32)

    call section('19. rank > 2: shape/dtype are readable, the 2-D getter refuses cleanly')
    pay = [(int(i, int8), i=1, 32)]
    call raw_file(out('rank3.bin'), '{"a":{"dtype":"F32","shape":[2,2,2],"data_offsets":[0,32]}}', &
                  pay, stat)
    call r%open(out('rank3.bin'), stat, msg)
    call ok(stat == st_ok, 'a rank-3 tensor file opens')
    call r%shape('a', shp, stat, msg)
    call ok(size(shp) == 3 .and. shp(1) == 2 .and. shp(2) == 2 .and. shp(3) == 2, &
            'rank-3 shape is reported')
    call r%get('a', flat, stat, msg)
    call ok(stat == st_ok .and. size(flat) == 8, 'the flat getter works for rank-3')
    call r%get('a', m2, stat, msg)
    call ok_stat(stat, st_err_range, 'rank 3', 'the 2-D getter refuses rank-3', msg)
  end subroutine test_rank3_and_get2d

  ! ==================================================================== 11
  subroutine test_large_tensor()
    type(st_writer) :: w
    type(st_reader) :: r
    integer, parameter :: n = 1024
    real(real32), allocatable :: big(:, :)
    real(real32), pointer :: p(:, :) => null()
    character(len=:), allocatable :: msg
    integer :: stat, i, j
    integer(int64) :: rate, t0, t1, nb
    real(real64) :: secs
    real(real32) :: checksum, checksum_in

    call section('20. 1024x1024 F32 (4 MiB), timed: catches quadratic copying')
    allocate (big(n, n))
    do j = 1, n
      do i = 1, n
        big(i, j) = real(i, real32)*0.5 + real(j, real32)
      end do
    end do
    checksum = sum(big)
    call system_clock(t0, rate)
    call w%init()
    call w%set_meta('size', '1024x1024')
    call w%set('big', big)
    call w%write(out('big.safetensors'), stat, msg)
    call ok(stat == st_ok, 'wrote a 4 MiB tensor')
    call r%open(out('big.safetensors'), stat, msg)
    call r%get('big', p, stat, msg)
    call system_clock(t1)
    secs = real(t1 - t0, real64)/real(rate, real64)
    checksum_in = sum(p)
    call r%nbytes('big', nb, stat, msg)
    call ok(stat == st_ok .and. size(p, 1) == n .and. size(p, 2) == n, &
            'read it back with the right extents')
    call ok(nb == int(n, int64)*int(n, int64)*4_int64, 'nbytes == 4 MiB')
    call ok(abs(checksum_in - checksum) <= 1.0e-3*abs(checksum), 'checksum matches')
    call ok(secs < 10.0d0, 'write+read of 4 MiB took '//trim(fmt1(secs))// &
            ' s (< 10 s; a quadratic copy would blow this up)')
    deallocate (big)
  end subroutine test_large_tensor

  ! ==================================================================== 12
  subroutine test_cross_language()
    type(st_writer) :: w
    type(st_reader) :: r
    real(real32) :: x(3) = [0.25, -1.5, 1.0e6]
    character(len=:), allocatable :: msg, cmd, py
    character(len=256) :: pybuf
    integer :: stat, cstat, est
    logical :: ex

    call section('21. cross-language round-trip with the Python oracle')
    call w%init()
    call w%set_meta('origin', 'fortran')
    call w%set('x', x)
    call w%write(out('to_python.safetensors'), stat, msg)
    call ok(stat == st_ok, 'wrote to_python.safetensors')

    pybuf = ''
    call get_environment_variable('ST_PYTHON', pybuf)
    py = trim(pybuf)
    if (len(py) == 0) py = 'python3'
    inquire (file='tools/reference_writer.py', exist=ex)
    if (ex) then
      cmd = trim(py)//' tools/reference_writer.py --verify '//out('to_python.safetensors')
    else
      cmd = trim(py)//' ../tools/reference_writer.py --verify '//out('to_python.safetensors')
    end if
    call execute_command_line(trim(cmd)//' >/dev/null 2>&1', exitstat=est, cmdstat=cstat)
    if (cstat /= 0 .or. est == 127) then
      call ok(.true., 'SKIP: no usable Python interpreter ('//trim(py)// &
              ') to read our file from the test')
    else
      call ok(est == 0, 'fortran -> python: the Python oracle validated our file: '//trim(cmd))
    end if

    ! Python -> Fortran is the other half; it is also covered by the fixtures,
    ! but keeping both directions in one place makes the round-trip explicit.
    call r%open(fixture('oracle_mixed.safetensors'), stat, msg)
    call ok(stat == st_ok, 'python -> fortran: the oracle-written fixture opens here')
  end subroutine test_cross_language

end program test_safetensors
