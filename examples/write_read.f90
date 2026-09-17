! write_read.f90 - minimal end-to-end example: write a checkpoint, read it back.
!
! Build and run:
!   fpm run --example write_read
!
! The example deliberately shows the two properties that matter in practice:
!   1. everything that can fail returns (stat, msg) -- nothing aborts the process;
!   2. shape/dtype can be inspected without materialising the tensor.
program write_read
  use, intrinsic :: iso_fortran_env, only: int32, int64, real32
  use safetensors, only: st_writer, st_reader, st_ok
  implicit none

  type(st_writer) :: w
  type(st_reader) :: r
  real(real32), allocatable :: wte(:, :), q(:)
  real(real32), pointer :: p(:, :) => null()
  real(real32), pointer :: pq(:) => null()
  integer(int64), allocatable :: shp(:)
  character(len=:), allocatable :: msg, val, dtype
  integer :: stat, k
  logical :: found

  ! ---------------------------------------------------------------- write side
  allocate (wte(4, 8), q(12))
  wte = reshape([(real(k, real32), k=1, 32)], [4, 8])
  q = [(real(k, real32)*0.5, k=1, 12)]

  call w%init()
  call w%set_meta('format_version', '1')
  call w%set_meta('bpb', '1.59994')                      ! characterisation travels with the weights
  call w%set_meta('caracterização', 'acentos ok')        ! UTF-8 metadata is fine
  call w%set_meta_int('n_layer', 12_int64)
  call w%set('wte', wte)                                 ! dtype and rank come from the argument
  call w%set('l3.q', q)
  call w%write('example.safetensors', stat, msg)
  if (stat /= st_ok) then
    write (*, '(A)') 'write failed: '//msg
    stop 1
  end if
  write (*, '(A,I0,A)') 'wrote example.safetensors (', file_size('example.safetensors'), ' bytes)'

  ! ---------------------------------------------------------------- read side
  call r%open('example.safetensors', stat, msg)
  if (stat /= st_ok) then
    write (*, '(A)') 'open failed: '//msg
    stop 1
  end if

  write (*, '(A,I0,A,I0,A,I0,A)') 'tensors: ', r%n_tensors(), '  (header ', r%header_size(), &
    ' bytes, buffer ', r%buffer_size(), ' bytes)'
  call r%dtype('wte', dtype, stat, msg)                  ! no bytes are read here
  call r%shape('wte', shp, stat, msg)
  write (*, '(A,A,A,2(I0,1X),A)') 'wte: dtype=', dtype, ' shape=[', shp, ']'
  call r%meta('bpb', val, found)
  if (found) write (*, '(A,A)') 'meta bpb = ', val

  call r%get('wte', p, stat, msg)                        ! rank-2 pointer, no extra copy
  if (stat /= st_ok) then
    write (*, '(A)') 'get failed: '//msg
    stop 1
  end if
  ! Header shape [4, 8] is C order; Fortran is column-major, so the array that keeps
  ! the same memory (no copy, no transpose) has the extents reversed: (8, 4) and
  ! p == reshape(wte, [8, 4]). See the README section "Rank >= 2 and C order".
  write (*, '(A,2(I0,1X),A,L1)') 'wte Fortran extents: (', shape(p), ')  same bytes: ', &
    all(p == reshape(wte, [size(p, 1), size(p, 2)]))

  call r%get('l3.q', pq, stat, msg)
  write (*, '(A,L1)') 'l3.q round-trip equal: ', all(pq == q)

  call r%close()

  ! --------------------------------------------------- error paths are values
  call r%open('does_not_exist.safetensors', stat, msg)
  write (*, '(A,I0,A)') 'expected failure: stat=', stat, ' msg='//msg

contains

  integer(int64) function file_size(path)
    character(*), intent(in) :: path
    integer :: sz
    inquire (file=path, size=sz)
    file_size = int(sz, int64)
  end function file_size

end program write_read
