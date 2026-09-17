! safetensors_json.f90 — minimal JSON parser and emitter, no external dependency.
!
! WHY A JSON LAYER OF OUR OWN, and not stdlib/json-fortran:
!   (1) the library promises zero dependencies (the lab builds this next to BLAS
!       kernels and does not want to drag a package manager along);
!   (2) the subset safetensors uses is tiny (object, string, integer, array of
!       integers). A parser this size is auditable end to end — which matters
!       because the header is UNTRUSTED data coming from outside the lab;
!   (3) full control of the messages: every rejection says what was wrong and at
!       which byte, not just "invalid JSON".
!
! WHY A "FLAT ARENA" AND NOT A TREE OF DERIVED TYPES (a real trap):
! the first version used a recursive type (`type :: json_value` containing
! `type(json_value), allocatable :: items(:)`) — the "obvious" design. With
! gfortran 15.3 that CRASHES (free(): invalid pointer / SIGSEGV) when a deep copy
! of such a value is deallocated; reproduced in 10 isolated lines
! (the case is written up in REPORT.md). Instead of depending on a compiler bug,
! the nodes live in flat vectors of intrinsic types and the links between them
! are integer INDICES:
!     nkind(i)          kind of node i (json_object, json_string, ...)
!     sbeg(i), slen(i)  span of node i's string inside `text`
!     efirst(i),ecount(i)  contiguous block of child edges of node i
!     echild(e)         child node of edge e
!     kbeg(e), klen(e)  span of the key (objects) or 0 (arrays)
! Side benefits: no deep copies, no recursive finalisation, contiguous memory,
! and spans that stay valid forever (the text buffer grows through move_alloc and
! is never reallocated under the spans).
!
! What is deliberately NOT supported (rejected with a message, never ignored):
! floats in JSON, duplicate keys in the same object, nesting deeper than 16
! levels, invalid escapes, raw control characters inside a string. The format
! only uses integers — accepting floats would be free bug surface.
!
! Important: the emission has to match serde_json (the official writer) byte for
! byte, otherwise the byte-for-byte parity with the reference breaks:
!   `"` -> `\"`; `\` -> `\\`; 0x08/09/0A/0C/0D -> `\b \t \n \f \r`;
!   other bytes < 0x20 -> `\u00xx` (LOWERCASE hex); `/` is NOT escaped; 0x7F and
!   bytes >= 0x80 pass through raw (UTF-8 preserved).
module safetensors_json
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private

  ! ------------------------------------------------------------- error codes
  integer, parameter, public :: json_ok = 0
  integer, parameter, public :: json_err_syntax = 1
  integer, parameter, public :: json_err_truncated = 2
  integer, parameter, public :: json_err_depth = 3
  integer, parameter, public :: json_err_duplicate = 4
  integer, parameter, public :: json_err_overflow = 5
  integer, parameter, public :: json_err_trailing = 6

  ! --------------------------------------------------------------- node kinds
  integer, parameter, public :: json_object = 1
  integer, parameter, public :: json_array = 2
  integer, parameter, public :: json_string = 3
  integer, parameter, public :: json_number = 4
  integer, parameter, public :: json_bool = 5
  integer, parameter, public :: json_null = 6

  ! The real format is 2 levels deep; 16 is slack and cuts short a hostile "[[[[[[...".
  integer, parameter :: max_depth = 16

  ! -------------------------------------------------------------------- types
  type, public :: json_doc
    private
    character(len=:), allocatable :: text      ! every string, concatenated
    integer(int64) :: tlen = 0
    integer, allocatable :: nkind(:)
    integer(int64), allocatable :: nnum(:)
    logical, allocatable :: nflag(:)
    integer(int64), allocatable :: sbeg(:), slen(:)
    integer, allocatable :: efirst(:), ecount(:)
    integer, allocatable :: echild(:)
    integer(int64), allocatable :: kbeg(:), klen(:)
    integer :: nn = 0
    integer :: ne = 0
    integer :: nroot = 0
  contains
    procedure :: root => jd_root
    procedure :: kind_of => jd_kind_of
    procedure :: count => jd_count
    procedure :: child => jd_child
    procedure :: key => jd_key
    procedure :: member => jd_member
    procedure :: str_of => jd_str_of
    procedure :: int_of => jd_int_of
    procedure :: bool_of => jd_bool_of
  end type json_doc

  ! Output buffer with geometric growth. Naive concatenation
  ! (`s = s // c`) is O(n^2), and the header of a 90-tensor model with a card of
  ! metadata is several KiB; the 1024x1024 test exists to catch a quadratic
  ! regression, so this one must not be quadratic either.
  type, public :: json_writer
    private
    character(len=:), allocatable :: buf
    integer(int64) :: n = 0
  contains
    procedure :: reset => jw_reset
    procedure :: raw => jw_raw
    procedure :: string => jw_string
    procedure :: integer => jw_integer
    procedure :: finish => jw_finish
  end type json_writer

  type :: json_parser
    character(len=:), allocatable :: src
    integer(int64) :: pos = 1
    integer :: stat = json_ok
    character(len=:), allocatable :: msg
  end type json_parser

  public :: json_parse, json_escape, sort_indices

contains

  ! =============================================================== queries
  pure function jd_root(self) result(i)
    class(json_doc), intent(in) :: self
    integer :: i
    i = self%nroot
  end function jd_root

  pure function jd_kind_of(self, i) result(k)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    integer :: k
    k = json_null
    if (i >= 1 .and. i <= self%nn) k = self%nkind(i)
  end function jd_kind_of

  pure function jd_count(self, i) result(n)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    integer :: n
    n = 0
    if (i >= 1 .and. i <= self%nn) n = self%ecount(i)
  end function jd_count

  pure function jd_child(self, i, j) result(c)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i, j
    integer :: c
    c = 0
    if (i < 1 .or. i > self%nn) return
    if (j < 1 .or. j > self%ecount(i)) return
    c = self%echild(self%efirst(i) + j - 1)
  end function jd_child

  function jd_key(self, i, j) result(s)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i, j
    character(len=:), allocatable :: s
    integer :: e
    s = ''
    if (i < 1 .or. i > self%nn) return
    if (j < 1 .or. j > self%ecount(i)) return
    e = self%efirst(i) + j - 1
    if (self%klen(e) > 0) s = self%text(self%kbeg(e):self%kbeg(e) + self%klen(e) - 1)
  end function jd_key

  function jd_str_of(self, i) result(s)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    character(len=:), allocatable :: s
    s = ''
    if (i < 1 .or. i > self%nn) return
    if (self%nkind(i) /= json_string) return
    if (self%slen(i) > 0) s = self%text(self%sbeg(i):self%sbeg(i) + self%slen(i) - 1)
  end function jd_str_of

  pure function jd_int_of(self, i) result(v)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    integer(int64) :: v
    v = 0
    if (i >= 1 .and. i <= self%nn) then
      if (self%nkind(i) == json_number) v = self%nnum(i)
    end if
  end function jd_int_of

  pure function jd_bool_of(self, i) result(v)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    logical :: v
    v = .false.
    if (i >= 1 .and. i <= self%nn) then
      if (self%nkind(i) == json_bool) v = self%nflag(i)
    end if
  end function jd_bool_of

  ! Index of the child with key `key` (0 when absent). Compared span by span,
  ! without materialising the keys.
  function jd_member(self, i, key) result(c)
    class(json_doc), intent(in) :: self
    integer, intent(in) :: i
    character(*), intent(in) :: key
    integer :: c
    integer :: j
    integer(int64) :: b, l
    c = 0
    if (i < 1 .or. i > self%nn) return
    if (self%nkind(i) /= json_object) return
    do j = 1, self%ecount(i)
      b = self%kbeg(self%efirst(i) + j - 1)
      l = self%klen(self%efirst(i) + j - 1)
      if (l == int(len(key), int64)) then
        if (l == 0 .or. self%text(b:b + l - 1) == key) then
          c = self%echild(self%efirst(i) + j - 1)
          return
        end if
      end if
    end do
  end function jd_member

  ! ================================================================= parser
  subroutine json_parse(text, doc, stat, msg)
    character(*), intent(in) :: text
    type(json_doc), intent(out) :: doc
    integer, intent(out) :: stat
    character(len=:), allocatable, intent(out) :: msg
    type(json_parser) :: p
    integer :: idx

    p%src = text
    p%pos = 1
    p%stat = json_ok
    p%msg = ''
    call jp_skip_ws(p)
    if (p%pos > len(p%src)) then
      call jp_set_err(p, json_err_truncated, 'header is empty (no JSON value)')
    else
      call jp_value(p, doc, idx, 1)
      if (p%stat == json_ok) then
        doc%nroot = idx
        call jp_skip_trailing(p)
        if (p%pos <= len(p%src)) &
          call jp_set_err(p, json_err_trailing, &
                          'trailing data after the JSON value at byte '//trim(int2str(p%pos)))
      end if
    end if
    stat = p%stat
    msg = p%msg
    if (stat == json_ok) msg = ''
  end subroutine json_parse

  subroutine jp_set_err(p, code, text)
    type(json_parser), intent(inout) :: p
    integer, intent(in) :: code
    character(*), intent(in) :: text
    ! Keeps the FIRST error: it is the informative one (the rest are cascade).
    if (p%stat /= json_ok) return
    p%stat = code
    p%msg = text
  end subroutine jp_set_err

  subroutine jp_skip_ws(p)
    type(json_parser), intent(inout) :: p
    integer :: c
    do while (p%pos <= len(p%src))
      c = iachar(p%src(p%pos:p%pos))
      if (c == 32 .or. c == 9 .or. c == 10 .or. c == 13) then
        p%pos = p%pos + 1
      else
        exit
      end if
    end do
  end subroutine jp_skip_ws

  ! Space is the official padding (0x20); NUL is tolerated defensively.
  subroutine jp_skip_trailing(p)
    type(json_parser), intent(inout) :: p
    integer :: c
    do while (p%pos <= len(p%src))
      c = iachar(p%src(p%pos:p%pos))
      if (c == 32 .or. c == 9 .or. c == 10 .or. c == 13 .or. c == 0) then
        p%pos = p%pos + 1
      else
        exit
      end if
    end do
  end subroutine jp_skip_trailing

  recursive subroutine jp_value(p, d, idx, depth)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(inout) :: d
    integer, intent(out) :: idx
    integer, intent(in) :: depth
    character :: c

    call jd_add_node(d, idx)
    call jp_skip_ws(p)
    if (p%pos > len(p%src)) then
      call jp_set_err(p, json_err_truncated, 'unexpected end of header while expecting a value')
      return
    end if
    if (depth > max_depth) then
      call jp_set_err(p, json_err_depth, 'JSON nested deeper than 16 levels')
      return
    end if
    c = p%src(p%pos:p%pos)
    select case (c)
    case ('{')
      call jp_object(p, d, idx, depth)
    case ('[')
      call jp_array(p, d, idx, depth)
    case ('"')
      d%nkind(idx) = json_string
      call jp_string(p, d, d%sbeg(idx), d%slen(idx))
    case ('t')
      call jp_literal(p, 'true')
      if (p%stat == json_ok) then
        d%nkind(idx) = json_bool
        d%nflag(idx) = .true.
      end if
    case ('f')
      call jp_literal(p, 'false')
      if (p%stat == json_ok) then
        d%nkind(idx) = json_bool
        d%nflag(idx) = .false.
      end if
    case ('n')
      call jp_literal(p, 'null')
      if (p%stat == json_ok) d%nkind(idx) = json_null
    case ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
      d%nkind(idx) = json_number
      call jp_number(p, d%nnum(idx))
    case default
      call jp_set_err(p, json_err_syntax, &
                      'unexpected character '''//c//''' at byte '//trim(int2str(p%pos)))
    end select
  end subroutine jp_value

  subroutine jp_literal(p, word)
    type(json_parser), intent(inout) :: p
    character(*), intent(in) :: word
    integer :: l
    l = len(word)
    if (p%pos + l - 1 > len(p%src)) then
      call jp_set_err(p, json_err_truncated, 'truncated literal '''//word//'''')
      return
    end if
    if (p%src(p%pos:p%pos + l - 1) /= word) then
      call jp_set_err(p, json_err_syntax, 'invalid literal at byte '//trim(int2str(p%pos)))
      return
    end if
    p%pos = p%pos + l
  end subroutine jp_literal

  ! Integers only: see the module header.
  subroutine jp_number(p, val)
    type(json_parser), intent(inout) :: p
    integer(int64), intent(out) :: val
    logical :: neg
    integer :: c, ndigits
    integer(int64) :: acc
    integer(int64), parameter :: maxv = huge(0_int64)

    val = 0_int64
    if (p%pos > len(p%src)) then
      call jp_set_err(p, json_err_truncated, 'expected a number')
      return
    end if
    neg = .false.
    if (p%src(p%pos:p%pos) == '-') then
      neg = .true.
      p%pos = p%pos + 1
    end if
    acc = 0
    ndigits = 0
    do while (p%pos <= len(p%src))
      c = iachar(p%src(p%pos:p%pos))
      if (c < 48 .or. c > 57) exit
      if (acc > (maxv - int(c - 48, int64))/10_int64) then
        call jp_set_err(p, json_err_overflow, 'integer too large for int64 at byte '// &
                        trim(int2str(p%pos)))
        return
      end if
      acc = acc*10_int64 + int(c - 48, int64)
      ndigits = ndigits + 1
      p%pos = p%pos + 1
    end do
    if (ndigits == 0) then
      call jp_set_err(p, json_err_syntax, 'expected an integer at byte '//trim(int2str(p%pos)))
      return
    end if
    if (p%pos <= len(p%src)) then
      c = iachar(p%src(p%pos:p%pos))
      if (c == 46 .or. c == 101 .or. c == 69) then
        call jp_set_err(p, json_err_syntax, &
                        'non-integer number (decimals/exponents are not allowed in this '// &
                        'format) at byte '//trim(int2str(p%pos)))
        return
      end if
    end if
    if (neg) then
      val = -acc
    else
      val = acc
    end if
  end subroutine jp_number

  ! Decodes a JSON string straight into the arena's text buffer; it returns the
  ! span (beg,len). No intermediate string: less copying and fewer allocations.
  subroutine jp_string(p, d, beg, sln)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(inout) :: d
    integer(int64), intent(out) :: beg, sln
    character :: c
    integer :: ia

    beg = d%tlen + 1
    sln = 0_int64
    if (p%pos > len(p%src) .or. p%src(p%pos:p%pos) /= '"') then
      call jp_set_err(p, json_err_syntax, 'expected ''"'' at byte '//trim(int2str(p%pos)))
      return
    end if
    p%pos = p%pos + 1
    do
      if (p%pos > len(p%src)) then
        call jp_set_err(p, json_err_truncated, 'unterminated string (starts before byte '// &
                        trim(int2str(p%pos))//')')
        return
      end if
      c = p%src(p%pos:p%pos)
      ia = iachar(c)
      if (c == '"') then
        p%pos = p%pos + 1
        exit
      else if (ia == 92) then                    ! '\' by code, no literal confusion
        p%pos = p%pos + 1
        call jp_escape(p, d, sln)
        if (p%stat /= json_ok) return
      else
        if (ia < 32) then
          call jp_set_err(p, json_err_syntax, &
                          'raw control character (0x'//to_hex2(ia)//') inside a string at byte '// &
                          trim(int2str(p%pos))//'; it must be escaped')
          return
        end if
        call jd_put(d, c)
        sln = sln + 1
        p%pos = p%pos + 1
      end if
    end do
  end subroutine jp_string

  subroutine jp_escape(p, d, sln)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(inout) :: d
    integer(int64), intent(inout) :: sln
    character :: c
    integer :: ia, cp, lo

    if (p%pos > len(p%src)) then
      call jp_set_err(p, json_err_truncated, 'string ends right after a backslash')
      return
    end if
    c = p%src(p%pos:p%pos)
    select case (c)
    case ('"')
      call jd_put(d, '"')
      sln = sln + 1
      p%pos = p%pos + 1
    case ('\')
      call jd_put(d, achar(92))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('/')
      call jd_put(d, '/')
      sln = sln + 1
      p%pos = p%pos + 1
    case ('b')
      call jd_put(d, achar(8))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('f')
      call jd_put(d, achar(12))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('n')
      call jd_put(d, achar(10))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('r')
      call jd_put(d, achar(13))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('t')
      call jd_put(d, achar(9))
      sln = sln + 1
      p%pos = p%pos + 1
    case ('u')
      p%pos = p%pos + 1
      call jp_hex4(p, cp)
      if (p%stat /= json_ok) return
      if (cp >= 55296 .and. cp <= 56319) then               ! D800..DBFF: high surrogate
        if (p%pos + 1 <= len(p%src)) then
          if (p%src(p%pos:p%pos + 1) == '\u') then
            p%pos = p%pos + 2
            call jp_hex4(p, lo)
            if (p%stat /= json_ok) return
            if (lo >= 56320 .and. lo <= 57343) then         ! DC00..DFFF: low surrogate
              cp = 65536 + (cp - 55296)*1024 + (lo - 56320)
            else
              call jp_set_err(p, json_err_syntax, 'invalid low surrogate in \u escape')
              return
            end if
          else
            call jp_set_err(p, json_err_syntax, 'lone high surrogate in \u escape')
            return
          end if
        else
          call jp_set_err(p, json_err_truncated, 'truncated \u surrogate escape')
          return
        end if
      else if (cp >= 56320 .and. cp <= 57343) then
        call jp_set_err(p, json_err_syntax, 'lone low surrogate in \u escape')
        return
      end if
      call jd_put_utf8(d, cp, sln)
    case default
      ia = iachar(c)
      if (ia >= 32 .and. ia < 127) then
        call jp_set_err(p, json_err_syntax, &
                        'invalid escape sequence \'//c//' at byte '//trim(int2str(p%pos)))
      else
        call jp_set_err(p, json_err_syntax, &
                        'invalid escape sequence at byte '//trim(int2str(p%pos)))
      end if
    end select
  end subroutine jp_escape

  subroutine jp_hex4(p, cp)
    type(json_parser), intent(inout) :: p
    integer, intent(out) :: cp
    integer :: i, d
    cp = 0
    do i = 1, 4
      if (p%pos > len(p%src)) then
        call jp_set_err(p, json_err_truncated, 'truncated \u escape')
        return
      end if
      d = hexval(iachar(p%src(p%pos:p%pos)))
      if (d < 0) then
        call jp_set_err(p, json_err_syntax, 'invalid hex digit in \u escape at byte '// &
                        trim(int2str(p%pos)))
        return
      end if
      cp = cp*16 + d
      p%pos = p%pos + 1
    end do
  end subroutine jp_hex4

  pure function hexval(c) result(d)
    integer, intent(in) :: c
    integer :: d
    d = -1
    if (c >= 48 .and. c <= 57) d = c - 48
    if (c >= 97 .and. c <= 102) d = c - 87
    if (c >= 65 .and. c <= 70) d = c - 55
  end function hexval

  recursive subroutine jp_object(p, d, idx, depth)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(inout) :: d
    integer, intent(in) :: idx, depth
    integer, allocatable :: kids(:)
    integer(int64), allocatable :: kb(:), kl(:)
    integer :: nk, cap, ci
    integer(int64) :: b, l
    character(len=:), allocatable :: key

    d%nkind(idx) = json_object
    p%pos = p%pos + 1                                  ! consumes '{'
    nk = 0
    cap = 8
    allocate (kids(cap), kb(cap), kl(cap))
    call jp_skip_ws(p)
    if (p%pos <= len(p%src)) then
      if (p%src(p%pos:p%pos) == '}') then
        p%pos = p%pos + 1
        call jd_add_edges(d, idx, kids(1:0), kb(1:0), kl(1:0), p)
        return
      end if
    end if
    do
      call jp_skip_ws(p)
      call jp_string(p, d, b, l)
      if (p%stat /= json_ok) return
      call jp_skip_ws(p)
      if (p%pos > len(p%src) .or. p%src(p%pos:p%pos) /= ':') then
        key = ''
        if (l > 0) key = d%text(b:b + l - 1)
        call jp_set_err(p, json_err_syntax, 'expected '':'' after key '''//key//''' at byte '// &
                        trim(int2str(p%pos)))
        return
      end if
      p%pos = p%pos + 1
      call jp_value(p, d, ci, depth + 1)
      if (p%stat /= json_ok) return
      nk = nk + 1
      if (nk > cap) then
        call grow_obj(kids, kb, kl, cap)
      end if
      kids(nk) = ci
      kb(nk) = b
      kl(nk) = l
      call jp_skip_ws(p)
      if (p%pos > len(p%src)) then
        call jp_set_err(p, json_err_truncated, 'unterminated object (expected ''}'')')
        return
      end if
      if (p%src(p%pos:p%pos) == ',') then
        p%pos = p%pos + 1
        cycle
      else if (p%src(p%pos:p%pos) == '}') then
        p%pos = p%pos + 1
        exit
      else
        call jp_set_err(p, json_err_syntax, 'expected '','' or ''}'' at byte '// &
                        trim(int2str(p%pos)))
        return
      end if
    end do
    call jd_add_edges(d, idx, kids(1:nk), kb(1:nk), kl(1:nk), p)
    if (p%stat /= json_ok) return
    ! A duplicate key is invalid in this format (the official writer rejects it).
    ! Detected in O(n) with a hash set: a 100 MB header can hold ~1M keys, and a
    ! quadratic loop there would be exactly the DoS vector this format fights.
    call jp_check_duplicate_keys(p, d, idx)
  contains
    subroutine grow_obj(k, b, l, c)
      integer, allocatable, intent(inout) :: k(:)
      integer(int64), allocatable, intent(inout) :: b(:), l(:)
      integer, intent(inout) :: c
      integer, allocatable :: tk(:)
      integer(int64), allocatable :: tb(:), tl(:)
      allocate (tk(2*c), tb(2*c), tl(2*c))
      tk(1:c) = k(1:c)
      tb(1:c) = b(1:c)
      tl(1:c) = l(1:c)
      call move_alloc(tk, k)
      call move_alloc(tb, b)
      call move_alloc(tl, l)
      c = 2*c
    end subroutine grow_obj
  end subroutine jp_object

  recursive subroutine jp_array(p, d, idx, depth)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(inout) :: d
    integer, intent(in) :: idx, depth
    integer, allocatable :: kids(:)
    integer(int64), allocatable :: dum(:)
    integer :: n, cap, ci

    d%nkind(idx) = json_array
    p%pos = p%pos + 1
    n = 0
    cap = 8
    allocate (kids(cap), dum(cap))
    call jp_skip_ws(p)
    if (p%pos <= len(p%src)) then
      if (p%src(p%pos:p%pos) == ']') then
        p%pos = p%pos + 1
        call jd_add_edges(d, idx, kids(1:0), dum(1:0), dum(1:0), p)
        return
      end if
    end if
    do
      call jp_value(p, d, ci, depth + 1)
      if (p%stat /= json_ok) return
      n = n + 1
      if (n > cap) then
        block
          integer, allocatable :: tk(:)
          integer(int64), allocatable :: td(:)
          allocate (tk(2*cap), td(2*cap))
          tk(1:cap) = kids(1:cap)
          td(1:cap) = dum(1:cap)
          call move_alloc(tk, kids)
          call move_alloc(td, dum)
          cap = 2*cap
        end block
      end if
      kids(n) = ci
      call jp_skip_ws(p)
      if (p%pos > len(p%src)) then
        call jp_set_err(p, json_err_truncated, 'unterminated array (expected '']'')')
        return
      end if
      if (p%src(p%pos:p%pos) == ',') then
        p%pos = p%pos + 1
        cycle
      else if (p%src(p%pos:p%pos) == ']') then
        p%pos = p%pos + 1
        exit
      else
        call jp_set_err(p, json_err_syntax, 'expected '','' or '']'' at byte '// &
                        trim(int2str(p%pos)))
        return
      end if
    end do
    call jd_add_edges(d, idx, kids(1:n), dum(1:n), dum(1:n), p)
  end subroutine jp_array

  ! FNV-1a 64 + open-addressing table over the key spans.
  subroutine jp_check_duplicate_keys(p, d, idx)
    type(json_parser), intent(inout) :: p
    type(json_doc), intent(in) :: d
    integer, intent(in) :: idx
    integer, allocatable :: table(:)
    integer :: n, i, e, sz, slot, other
    integer(int64) :: h, mask

    n = d%ecount(idx)
    if (n < 2) return
    sz = 16
    do while (sz < 2*n)
      sz = 2*sz
    end do
    allocate (table(0:sz - 1), source=0)
    mask = int(sz - 1, int64)
    do i = 1, n
      e = d%efirst(idx) + i - 1
      h = span_hash(d, e)
      slot = int(iand(h, mask))
      do
        if (table(slot) == 0) then
          table(slot) = i
          exit
        end if
        other = d%efirst(idx) + table(slot) - 1
        if (same_span(d, e, other)) then
          call jp_set_err(p, json_err_duplicate, &
                          'duplicate key '''//d%text(d%kbeg(e):d%kbeg(e) + d%klen(e) - 1)// &
                          ''' in the header')
          return
        end if
        slot = int(iand(int(slot, int64) + 1_int64, mask))
      end do
    end do
  end subroutine jp_check_duplicate_keys

  pure function span_hash(d, e) result(h)
    type(json_doc), intent(in) :: d
    integer, intent(in) :: e
    integer(int64) :: h
    integer(int64) :: i
    h = -3750763034362895579_int64                 ! 14695981039346656037 mod 2^64
    do i = 1, d%klen(e)
      h = ieor(h, iand(int(iachar(d%text(d%kbeg(e) + i - 1:d%kbeg(e) + i - 1)), int64), 255_int64))
      h = h*1099511628211_int64                    ! benign overflow: mod 2^64
    end do
  end function span_hash

  pure function same_span(d, e1, e2) result(r)
    type(json_doc), intent(in) :: d
    integer, intent(in) :: e1, e2
    logical :: r
    r = .false.
    if (d%klen(e1) /= d%klen(e2)) return
    if (d%klen(e1) == 0) then
      r = .true.
      return
    end if
    r = d%text(d%kbeg(e1):d%kbeg(e1) + d%klen(e1) - 1) == &
        d%text(d%kbeg(e2):d%kbeg(e2) + d%klen(e2) - 1)
  end function same_span

  ! ============================================================ construction
  subroutine jd_add_node(d, idx)
    type(json_doc), intent(inout) :: d
    integer, intent(out) :: idx
    integer, allocatable :: ti(:)
    integer(int64), allocatable :: t64(:)
    logical, allocatable :: tl(:)
    integer :: cap

    if (.not. allocated(d%nkind)) then
      cap = 64
      allocate (d%nkind(cap), d%efirst(cap), d%ecount(cap))
      allocate (d%nnum(cap), d%sbeg(cap), d%slen(cap))
      allocate (d%nflag(cap))
      d%nkind = json_null
      d%nnum = 0_int64
      d%sbeg = 0_int64
      d%slen = 0_int64
      d%efirst = 0
      d%ecount = 0
      d%nflag = .false.
    end if
    if (d%nn + 1 > size(d%nkind)) then
      cap = 2*size(d%nkind)
      allocate (ti(cap)); ti(1:d%nn) = d%nkind(1:d%nn); call move_alloc(ti, d%nkind)
      allocate (ti(cap)); ti(1:d%nn) = d%efirst(1:d%nn); call move_alloc(ti, d%efirst)
      allocate (ti(cap)); ti(1:d%nn) = d%ecount(1:d%nn); call move_alloc(ti, d%ecount)
      allocate (t64(cap)); t64(1:d%nn) = d%nnum(1:d%nn); call move_alloc(t64, d%nnum)
      allocate (t64(cap)); t64(1:d%nn) = d%sbeg(1:d%nn); call move_alloc(t64, d%sbeg)
      allocate (t64(cap)); t64(1:d%nn) = d%slen(1:d%nn); call move_alloc(t64, d%slen)
      allocate (tl(cap)); tl(1:d%nn) = d%nflag(1:d%nn); call move_alloc(tl, d%nflag)
    end if
    d%nn = d%nn + 1
    idx = d%nn
    d%nkind(idx) = json_null
    d%nnum(idx) = 0_int64
    d%sbeg(idx) = 0_int64
    d%slen(idx) = 0_int64
    d%efirst(idx) = 0
    d%ecount(idx) = 0
    d%nflag(idx) = .false.
  end subroutine jd_add_node

  ! Edge (child) block of node `idx`: contiguous, carrying the keys (objects).
  subroutine jd_add_edges(d, idx, kids, kb, kl, p)
    type(json_doc), intent(inout) :: d
    integer, intent(in) :: idx
    integer, intent(in) :: kids(:)
    integer(int64), intent(in) :: kb(:), kl(:)
    type(json_parser), intent(inout) :: p
    integer, allocatable :: ti(:)
    integer(int64), allocatable :: t64(:)
    integer :: n, e, cap

    n = size(kids)
    if (n == 0) then
      d%efirst(idx) = d%ne + 1
      d%ecount(idx) = 0
      return
    end if
    if (.not. allocated(d%echild)) then
      cap = 128
      allocate (d%echild(cap), d%kbeg(cap), d%klen(cap))
      d%kbeg = 0_int64
      d%klen = 0_int64
    end if
    if (d%ne + n > size(d%echild)) then
      cap = max(2*size(d%echild), d%ne + n)
      allocate (ti(cap)); ti(1:d%ne) = d%echild(1:d%ne); call move_alloc(ti, d%echild)
      allocate (t64(cap)); t64(1:d%ne) = d%kbeg(1:d%ne); call move_alloc(t64, d%kbeg)
      allocate (t64(cap)); t64(1:d%ne) = d%klen(1:d%ne); call move_alloc(t64, d%klen)
    end if
    d%efirst(idx) = d%ne + 1
    d%ecount(idx) = n
    do e = 1, n
      d%ne = d%ne + 1
      d%echild(d%ne) = kids(e)
      d%kbeg(d%ne) = kb(e)
      d%klen(d%ne) = kl(e)
    end do
  end subroutine jd_add_edges

  ! Appends one byte to the arena's text buffer (geometric growth).
  subroutine jd_put(d, c)
    type(json_doc), intent(inout) :: d
    character, intent(in) :: c
    character(len=:), allocatable :: nb
    integer(int64) :: cap
    cap = 0
    if (allocated(d%text)) cap = int(len(d%text), int64)
    if (d%tlen + 1 > cap) then
      allocate (character(len=max(2*cap, 256_int64)) :: nb)
      if (d%tlen > 0) nb(1:d%tlen) = d%text(1:d%tlen)
      call move_alloc(nb, d%text)
    end if
    d%tlen = d%tlen + 1
    d%text(d%tlen:d%tlen) = c
  end subroutine jd_put

  ! UTF-8 for one code point: a writer that emits \u has to produce the same bytes
  ! our emitter would (serde_json emits non-ASCII raw, never as \u).
  subroutine jd_put_utf8(d, cp, sln)
    type(json_doc), intent(inout) :: d
    integer, intent(in) :: cp
    integer(int64), intent(inout) :: sln
    if (cp < 128) then
      call jd_put(d, achar(cp))
      sln = sln + 1
    else if (cp < 2048) then
      call jd_put(d, achar(192 + cp/64))
      call jd_put(d, achar(128 + mod(cp, 64)))
      sln = sln + 2
    else if (cp < 65536) then
      call jd_put(d, achar(224 + cp/4096))
      call jd_put(d, achar(128 + mod(cp/64, 64)))
      call jd_put(d, achar(128 + mod(cp, 64)))
      sln = sln + 3
    else
      call jd_put(d, achar(240 + cp/262144))
      call jd_put(d, achar(128 + mod(cp/4096, 64)))
      call jd_put(d, achar(128 + mod(cp/64, 64)))
      call jd_put(d, achar(128 + mod(cp, 64)))
      sln = sln + 4
    end if
  end subroutine jd_put_utf8

  ! ================================================================ emission
  subroutine jw_reset(self)
    class(json_writer), intent(inout) :: self
    self%n = 0
    if (allocated(self%buf)) deallocate (self%buf)
  end subroutine jw_reset

  subroutine jw_raw(self, text)
    class(json_writer), intent(inout) :: self
    character(*), intent(in) :: text
    call jw_reserve(self, int(len(text), int64))
    self%buf(self%n + 1:self%n + len(text)) = text
    self%n = self%n + len(text)
  end subroutine jw_raw

  subroutine jw_integer(self, val)
    class(json_writer), intent(inout) :: self
    integer(int64), intent(in) :: val
    call jw_raw(self, trim(int2str(val)))
  end subroutine jw_integer

  subroutine jw_string(self, s)
    class(json_writer), intent(inout) :: self
    character(*), intent(in) :: s
    call jw_raw(self, '"'//json_escape(s)//'"')
  end subroutine jw_string

  subroutine jw_reserve(self, extra)
    class(json_writer), intent(inout) :: self
    integer(int64), intent(in) :: extra
    character(len=:), allocatable :: nb
    integer(int64) :: want, cap
    cap = 0
    if (allocated(self%buf)) cap = int(len(self%buf), int64)
    want = self%n + extra
    if (want <= cap) return
    allocate (character(len=max(2*cap, max(want, 256_int64))) :: nb)
    if (self%n > 0) nb(1:self%n) = self%buf(1:self%n)
    call move_alloc(nb, self%buf)
  end subroutine jw_reserve

  function jw_finish(self) result(text)
    class(json_writer), intent(in) :: self
    character(len=:), allocatable :: text
    if (self%n == 0) then
      text = ''
    else
      text = self%buf(1:self%n)
    end if
  end function jw_finish

  ! Escapes the way serde_json does (see the module header). Public because the
  ! parity test uses the very same function to check the escaping.
  pure function json_escape(s) result(out)
    character(*), intent(in) :: s
    character(len=:), allocatable :: out
    character(len=len(s)*6) :: tmp
    character :: c
    integer :: i, pos, ia
    character(len=6) :: esc
    character(len=4), parameter :: hexd = '0123456789abcdef'   ! lowercase: same as serde_json

    pos = 0
    do i = 1, len(s)
      c = s(i:i)
      ia = iachar(c)
      select case (ia)
      case (34)
        tmp(pos + 1:pos + 2) = '\"'; pos = pos + 2
      case (92)
        tmp(pos + 1:pos + 2) = '\\'; pos = pos + 2
      case (8)
        tmp(pos + 1:pos + 2) = '\b'; pos = pos + 2
      case (9)
        tmp(pos + 1:pos + 2) = '\t'; pos = pos + 2
      case (10)
        tmp(pos + 1:pos + 2) = '\n'; pos = pos + 2
      case (12)
        tmp(pos + 1:pos + 2) = '\f'; pos = pos + 2
      case (13)
        tmp(pos + 1:pos + 2) = '\r'; pos = pos + 2
      case (0:7, 11, 14:31)
        esc = '\u00'//hexd(ia/16 + 1:ia/16 + 1)//hexd(mod(ia, 16) + 1:mod(ia, 16) + 1)
        tmp(pos + 1:pos + 6) = esc
        pos = pos + 6
      case default
        tmp(pos + 1:pos + 1) = c
        pos = pos + 1
      end select
    end do
    out = tmp(1:pos)
  end function json_escape

  ! ================================================================ utilities
  pure function int2str(v) result(s)
    integer(int64), intent(in) :: v
    character(len=:), allocatable :: s
    character(len=32) :: buf
    write (buf, '(I0)') v
    s = trim(buf)
  end function int2str

  pure function to_hex2(v) result(s)
    integer, intent(in) :: v
    character(len=2) :: s
    character(len=16), parameter :: h = '0123456789abcdef'
    integer :: x
    x = max(0, min(255, v))
    s(1:1) = h(x/16 + 1:x/16 + 1)
    s(2:2) = h(mod(x, 16) + 1:mod(x, 16) + 1)
  end function to_hex2

  ! ================================================================= sorting
  ! Heapsort over an index vector, ordered by `keys`. Used to sort the tensors by
  ! start offset (JSON does not guarantee key order).
  ! Heapsort (not insertion sort) because a hostile header can hold ~1M tensors,
  ! and O(n^2) there would be a DoS vector.
  subroutine sort_indices(keys, idx)
    integer(int64), intent(in) :: keys(:)
    integer, allocatable, intent(out) :: idx(:)
    integer :: i, n
    n = size(keys)
    allocate (idx(n))
    do i = 1, n
      idx(i) = i
    end do
    do i = n/2, 1, -1
      call sift(keys, idx, i, n)
    end do
    do i = n, 2, -1
      call swap_i(idx(1), idx(i))
      call sift(keys, idx, 1, i - 1)
    end do
  end subroutine sort_indices

  subroutine sift(keys, idx, root, n)
    integer(int64), intent(in) :: keys(:)
    integer, intent(inout) :: idx(:)
    integer, intent(in) :: root, n
    integer :: r, c
    r = root
    do
      c = 2*r
      if (c > n) exit
      if (c < n) then
        if (keys(idx(c + 1)) > keys(idx(c))) c = c + 1
      end if
      if (keys(idx(c)) <= keys(idx(r))) exit
      call swap_i(idx(r), idx(c))
      r = c
    end do
  end subroutine sift

  subroutine swap_i(a, b)
    integer, intent(inout) :: a, b
    integer :: t
    t = a
    a = b
    b = t
  end subroutine swap_i

end module safetensors_json
