! inspect.f90 - print the header of a .safetensors file; optionally hash the payload.
!
! Build and run:
!   fpm run --example inspect -- model.safetensors [--hash]
!
! This is the "what is actually inside this checkpoint?" tool: it reads only the
! header (shape/dtype/offsets/metadata) and, with --hash, streams every tensor
! through FNV-1a 64 so a foreign file can be compared against what Python reads.
! The hash is over the RAW BYTES, so it is comparable across languages and is
! unaffected by NaN vs NaN comparisons.
program inspect
  use, intrinsic :: iso_fortran_env, only: int8, int64, real32, real64
  use safetensors, only: st_reader, st_ok, st_dtype_bits
  implicit none

  type(st_reader) :: r
  character(len=:), allocatable :: path, msg, name, val, dtype
  integer(int64), allocatable :: shp(:)
  integer(int64) :: b0, b1, nb, h
  integer(int8), pointer :: raw(:) => null()
  integer :: stat, i, j
  logical :: do_hash, found
  character(len=32) :: arg

  do_hash = .false.
  path = ''
  do i = 1, command_argument_count()
    call get_command_argument(i, arg)
    if (trim(arg) == '--hash') then
      do_hash = .true.
    else
      path = trim(arg)
    end if
  end do
  if (len(path) == 0) then
    write (*, '(A)') 'usage: inspect FILE.safetensors [--hash]'
    stop 0
  end if

  call r%open(path, stat, msg)
  if (stat /= st_ok) then
    write (*, '(A)') 'error: '//msg
    stop 1
  end if

  write (*, '(A,A)') 'file           : ', path
  write (*, '(A,I0)') 'file bytes     : ', r%file_size()
  write (*, '(A,I0)') 'header bytes   : ', r%header_size()
  write (*, '(A,I0)') 'payload bytes  : ', r%buffer_size()
  write (*, '(A,I0)') 'tensors        : ', r%n_tensors()
  write (*, '(A,I0)') 'metadata keys  : ', r%n_meta()
  do i = 1, r%n_meta()
    call r%meta_key(i, name, stat, msg)
    call r%meta(name, val, found)
    write (*, '(A,A,A,A,A)') '  meta[', name, '] = ', val, ''
  end do
  write (*, '(A)') ''
  write (*, '(A)') 'name                             dtype  shape                     bytes     offsets           fnv1a64'
  do i = 1, r%n_tensors()
    call r%tensor_name(i, name, stat, msg)
    call r%dtype(name, dtype, stat, msg)
    call r%shape(name, shp, stat, msg)
    call r%nbytes(name, nb, stat, msg)
    call r%tensor_offsets(name, b0, b1, stat, msg)
    write (*, '(A28,1X,A6,1X,A24,1X,I9,1X,A17)', advance='no') name, dtype, shape_txt(shp), nb, &
      '['//trim(i2s(b0))//','//trim(i2s(b1))//']'
    if (do_hash) then
      call r%get_raw(name, raw, stat, msg)
      if (stat /= st_ok) then
        write (*, '(A)') ' hash-failed: '//msg
        cycle
      end if
      h = fnv1a(raw)
      write (*, '(A,Z16)') ' ', h
    else
      write (*, '(A)') ''
    end if
  end do
  call r%close()

contains

  function shape_txt(s) result(t)
    integer(int64), intent(in) :: s(:)
    character(len=:), allocatable :: t
    integer :: k
    t = '['
    do k = 1, size(s)
      if (k > 1) t = t//','
      t = t//trim(i2s(s(k)))
    end do
    t = t//']'
  end function shape_txt

  pure function i2s(v) result(s)
    integer(int64), intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: b
    write (b, '(I0)') v
    s = trim(b)
  end function i2s

  ! FNV-1a 64 over raw bytes: cheap, dependency-free, and identical in Python
  ! (tools/verify_npy_vs_st.py hashes the .npy payloads the same way).
  pure function fnv1a(bytes) result(h)
    integer(int8), intent(in) :: bytes(:)
    integer(int64) :: h
    integer(int64) :: k
    h = -3750763034362895579_int64
    do k = 1, size(bytes, kind=int64)
      h = ieor(h, iand(int(bytes(k), int64), 255_int64))
      h = h*1099511628211_int64
    end do
  end function fnv1a

end program inspect
