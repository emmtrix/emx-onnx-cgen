"""Compile regex patterns to minimised DFA.

Pipeline: regex string → sre_parse AST → Thompson NFA → subset-construction
DFA → Hopcroft minimisation → renumbered DFA with explicit dead state.
"""

from __future__ import annotations

import re
import sre_parse
from collections import deque

try:
    from sre_constants import (
        ANY,
        AT,
        AT_BEGINNING,
        AT_BEGINNING_STRING,
        AT_END,
        AT_END_STRING,
        BRANCH,
        CATEGORY,
        CATEGORY_DIGIT,
        CATEGORY_NOT_DIGIT,
        CATEGORY_NOT_SPACE,
        CATEGORY_NOT_WORD,
        CATEGORY_SPACE,
        CATEGORY_WORD,
        IN,
        LITERAL,
        MAX_REPEAT,
        MIN_REPEAT,
        NEGATE,
        NOT_LITERAL,
        RANGE,
        SUBPATTERN,
    )
except ImportError:
    from re._constants import (  # Python ≥ 3.13
        ANY,
        AT,
        AT_BEGINNING,
        AT_BEGINNING_STRING,
        AT_END,
        AT_END_STRING,
        BRANCH,
        CATEGORY,
        CATEGORY_DIGIT,
        CATEGORY_NOT_DIGIT,
        CATEGORY_NOT_SPACE,
        CATEGORY_NOT_WORD,
        CATEGORY_SPACE,
        CATEGORY_WORD,
        IN,
        LITERAL,
        MAX_REPEAT,
        MIN_REPEAT,
        NEGATE,
        NOT_LITERAL,
        RANGE,
        SUBPATTERN,
    )

MAXREPEAT = sre_parse.MAXREPEAT

# Limit to prevent runaway DFA construction
_MAX_DFA_STATES = 10_000

# Limit for bit-parallel NFA (generated table is positions × 256 entries)
_MAX_BITNFA_POSITIONS = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIGIT = frozenset(range(48, 58))  # '0'-'9'
_WORD = (
    frozenset(range(48, 58))
    | frozenset(range(65, 91))
    | frozenset(range(97, 123))
    | frozenset({95})
)
_SPACE = frozenset({9, 10, 12, 13, 32})


def _build_casefold_table() -> dict[str, frozenset[int]]:
    """Build a lookup: fold-key → set of code-points in the same class.

    Uses ``str.casefold()`` for single-character folds and falls back to
    ``str.lower()`` for characters with multi-character casefolds (e.g. ß).
    This matches re2's *simple* case-folding behaviour.
    """
    table: dict[str, set[int]] = {}
    for cp in range(0x110000):
        ch = chr(cp)
        cf = ch.casefold()
        if len(cf) == 1:
            table.setdefault(cf, set()).add(cp)
        else:
            # Multi-char casefold (e.g. ß→ss): use lower() as simple fold
            lc = ch.lower()
            if len(lc) == 1:
                table.setdefault(lc, set()).add(cp)
    return {k: frozenset(v) for k, v in table.items()}


_CF_TABLE: dict[str, frozenset[int]] = _build_casefold_table()


def _category_bytes(cat: int) -> frozenset[int]:
    if cat == CATEGORY_DIGIT:
        return _DIGIT
    if cat == CATEGORY_NOT_DIGIT:
        return frozenset(range(256)) - _DIGIT
    if cat == CATEGORY_WORD:
        return _WORD
    if cat == CATEGORY_NOT_WORD:
        return frozenset(range(256)) - _WORD
    if cat == CATEGORY_SPACE:
        return _SPACE
    if cat == CATEGORY_NOT_SPACE:
        return frozenset(range(256)) - _SPACE
    return frozenset()


# ---------------------------------------------------------------------------
# NFA builder (Thompson construction)
# ---------------------------------------------------------------------------

class NFABuilder:
    """Build a Thompson NFA from an ``sre_parse`` AST."""

    def __init__(
        self,
        *,
        case_insensitive: bool = False,
        dot_all: bool = False,
        bytes_mode: bool = False,
    ):
        self._next: int = 0
        self.transitions: dict[tuple[int, int], set[int]] = {}
        self.epsilon: dict[int, set[int]] = {}
        self.case_insensitive = case_insensitive
        self.dot_all = dot_all
        self.bytes_mode = bytes_mode

    # -- state helpers -------------------------------------------------------

    def _new(self) -> int:
        s = self._next
        self._next += 1
        return s

    def _tr(self, src: int, byte: int, dst: int) -> None:
        self.transitions.setdefault((src, byte), set()).add(dst)

    def _eps(self, src: int, dst: int) -> None:
        self.epsilon.setdefault(src, set()).add(dst)

    # -- character helpers ---------------------------------------------------

    def _add_char(self, src: int, code: int, dst: int) -> None:
        """Add transition(s) for a single character code point."""
        if self.case_insensitive:
            for alt in self._case_variants(code):
                self._add_char_raw(src, alt, dst)
        else:
            self._add_char_raw(src, code, dst)

    def _add_char_raw(self, src: int, code: int, dst: int) -> None:
        """Add transition for one code point (no case expansion)."""
        if self.bytes_mode:
            if code <= 255:
                self._tr(src, code, dst)
            return
        if code <= 127:
            self._tr(src, code, dst)
        else:
            utf8 = chr(code).encode("utf-8")
            prev = src
            for i, b in enumerate(utf8):
                nxt = dst if i == len(utf8) - 1 else self._new()
                self._tr(prev, b, nxt)
                prev = nxt

    @staticmethod
    def _case_variants(code: int) -> frozenset[int]:
        """Return the Unicode simple-case-fold equivalence class of *code*."""
        ch = chr(code)
        cf = ch.casefold()
        if len(cf) == 1:
            return _CF_TABLE.get(cf, frozenset({code}))
        # Multi-character casefold (e.g. ß → ss): use lower() as key
        lc = ch.lower()
        if len(lc) == 1:
            return _CF_TABLE.get(lc, frozenset({code}))
        return frozenset({code})

    def _add_utf8_paths(
        self, src: int, dst: int, excluded_cps: set[int] | None = None
    ) -> None:
        """Add transitions for valid multi-byte UTF-8 sequences.

        If *excluded_cps* is given, code-points in that set are skipped.
        Uses shared intermediate states for compact NFA.
        """
        if excluded_cps is None:
            excluded_cps = set()

        # -- 2-byte sequences (U+0080 .. U+07FF) --
        # Group first-bytes by the set of accepted second-bytes so that
        # first-bytes with identical continuation ranges share one state.
        groups_2: dict[frozenset[int], list[int]] = {}
        for first in range(0xC2, 0xE0):
            acc: set[int] = set()
            for second in range(0x80, 0xC0):
                cp = ((first & 0x1F) << 6) | (second & 0x3F)
                if cp not in excluded_cps:
                    acc.add(second)
            if acc:
                groups_2.setdefault(frozenset(acc), []).append(first)
        for seconds, firsts in groups_2.items():
            inter = self._new()
            for fb in firsts:
                self._tr(src, fb, inter)
            for sb in seconds:
                self._tr(inter, sb, dst)

        # -- 3-byte sequences (U+0800 .. U+FFFF) --
        excl_3 = {cp for cp in excluded_cps if 0x0800 <= cp <= 0xFFFF}
        if not excl_3:
            # No exclusions – use shared continuation states
            cont3 = self._new()
            for b in range(0x80, 0xC0):
                self._tr(cont3, b, dst)
            s_e0 = self._new()
            self._tr(src, 0xE0, s_e0)
            for sb in range(0xA0, 0xC0):
                self._tr(s_e0, sb, cont3)
            s_norm3 = self._new()
            for fb in list(range(0xE1, 0xED)) + [0xEE, 0xEF]:
                self._tr(src, fb, s_norm3)
            for sb in range(0x80, 0xC0):
                self._tr(s_norm3, sb, cont3)
            s_ed = self._new()
            self._tr(src, 0xED, s_ed)
            for sb in range(0x80, 0xA0):
                self._tr(s_ed, sb, cont3)
        else:
            # Group by (first_byte, second_byte) to find restricted third-bytes
            excl_map: dict[int, dict[int, set[int]]] = {}
            for cp in excl_3:
                fb = ((cp >> 12) & 0x0F) | 0xE0
                sb = ((cp >> 6) & 0x3F) | 0x80
                tb = (cp & 0x3F) | 0x80
                excl_map.setdefault(fb, {}).setdefault(sb, set()).add(tb)

            def _second_range(fb: int) -> range:
                if fb == 0xE0:
                    return range(0xA0, 0xC0)
                if fb == 0xED:
                    return range(0x80, 0xA0)
                return range(0x80, 0xC0)

            for fb in range(0xE0, 0xF0):
                sr = _second_range(fb)
                if fb not in excl_map:
                    # No exclusions for this first byte – simple path
                    inter_fb = self._new()
                    self._tr(src, fb, inter_fb)
                    cont = self._new()
                    for b in range(0x80, 0xC0):
                        self._tr(cont, b, dst)
                    for sb in sr:
                        self._tr(inter_fb, sb, cont)
                else:
                    inter_fb = self._new()
                    self._tr(src, fb, inter_fb)
                    by_sb = excl_map[fb]
                    for sb in sr:
                        if sb in by_sb:
                            acc_thirds = set(range(0x80, 0xC0)) - by_sb[sb]
                            if acc_thirds:
                                inter_sb = self._new()
                                self._tr(inter_fb, sb, inter_sb)
                                for tb in acc_thirds:
                                    self._tr(inter_sb, tb, dst)
                        else:
                            cont = self._new()
                            self._tr(inter_fb, sb, cont)
                            for tb in range(0x80, 0xC0):
                                self._tr(cont, tb, dst)

        # -- 4-byte sequences (U+10000 .. U+10FFFF) --
        # Shared continuation state for last two bytes
        cont4_last = self._new()
        for b in range(0x80, 0xC0):
            self._tr(cont4_last, b, dst)
        cont4_mid = self._new()
        for b in range(0x80, 0xC0):
            self._tr(cont4_mid, b, cont4_last)
        # F0: F0 [90-BF] cont4_mid
        s_f0 = self._new()
        self._tr(src, 0xF0, s_f0)
        for sb in range(0x90, 0xC0):
            self._tr(s_f0, sb, cont4_mid)
        # F1-F3: [F1-F3] [80-BF] cont4_mid
        s_fnorm = self._new()
        for fb in range(0xF1, 0xF4):
            self._tr(src, fb, s_fnorm)
        for sb in range(0x80, 0xC0):
            self._tr(s_fnorm, sb, cont4_mid)
        # F4: F4 [80-8F] cont4_mid
        s_f4 = self._new()
        self._tr(src, 0xF4, s_f4)
        for sb in range(0x80, 0x90):
            self._tr(s_f4, sb, cont4_mid)

    def _char_set(self, items: list) -> set[int]:
        """Resolve an ``IN`` item list to a set of byte values.

        In UTF-8 mode only ASCII code-points (0-127) are included; non-ASCII
        code-points are handled separately via multi-byte UTF-8 paths.
        In bytes mode all byte values (0-255) are included directly.
        """
        byte_limit = 256 if self.bytes_mode else 128
        chars: set[int] = set()
        negate = False
        for op, value in items:
            if op == NEGATE:
                negate = True
            elif op == LITERAL:
                if self.case_insensitive:
                    for alt in self._case_variants(value):
                        if alt < byte_limit:
                            chars.add(alt)
                elif value < byte_limit:
                    chars.add(value)
            elif op == RANGE:
                lo, hi = value
                for c in range(lo, min(hi + 1, byte_limit)):
                    if self.case_insensitive:
                        for alt in self._case_variants(c):
                            if alt < byte_limit:
                                chars.add(alt)
                    else:
                        chars.add(c)
                # Case-insensitive (UTF-8 mode only): also find ASCII chars whose
                # non-ASCII case-fold equivalents fall within [lo, hi].
                if not self.bytes_mode and self.case_insensitive and hi >= 128:
                    non_ascii_lo = max(lo, 128)
                    for cf_key, equivalents in _CF_TABLE.items():
                        if len(cf_key) == 1 and ord(cf_key) <= 127:
                            if any(non_ascii_lo <= eq <= hi for eq in equivalents):
                                for alt in equivalents:
                                    if alt <= 127:
                                        chars.add(alt)
            elif op == CATEGORY:
                chars |= set(_category_bytes(value))
        chars &= set(range(byte_limit))
        if negate:
            chars = set(range(byte_limit)) - chars
        return chars

    def _excluded_cps(self, items: list) -> set[int]:
        """Collect non-ASCII code-points *excluded* by a negated IN list.

        When case-insensitive, also includes non-ASCII case-fold equivalents
        of excluded ASCII characters (e.g. 's' → U+017F LONG S).
        """
        cps: set[int] = set()
        for op, value in items:
            if op == LITERAL:
                if self.case_insensitive:
                    for alt in self._case_variants(value):
                        if alt >= 128:
                            cps.add(alt)
                elif value >= 128:
                    cps.add(value)
            elif op == RANGE:
                lo, hi = value
                for c in range(lo, min(hi + 1, 0x800)):
                    if self.case_insensitive:
                        for alt in self._case_variants(c):
                            if alt >= 128:
                                cps.add(alt)
                    elif c >= 128:
                        cps.add(c)
        return cps

    def _non_ascii_codes(self, items: list) -> set[int]:
        """Collect non-ASCII code-points from an ``IN`` item list.

        For case-insensitive mode, also includes non-ASCII case-fold
        equivalents of ASCII characters in the class.
        """
        codes: set[int] = set()
        negate = False
        for op, value in items:
            if op == NEGATE:
                negate = True
            elif op == LITERAL:
                if value > 127:
                    codes.add(value)
                if self.case_insensitive:
                    for alt in self._case_variants(value):
                        if alt > 127:
                            codes.add(alt)
            elif op == RANGE:
                lo, hi = value
                for c in range(max(lo, 128), min(hi + 1, 0x800)):
                    codes.add(c)
                if self.case_insensitive:
                    for c in range(lo, min(hi + 1, 128)):
                        for alt in self._case_variants(c):
                            if alt > 127:
                                codes.add(alt)
        # Negation handled separately via _add_utf8_paths
        if negate:
            return set()
        return codes

    def _add_3byte_range(self, src: int, dst: int, lo: int, hi: int) -> None:
        """Add NFA transitions accepting 3-byte UTF-8 code points in [lo, hi].

        Surrogates (U+D800–U+DFFF) are automatically excluded.
        """
        lo = max(lo, 0x800)
        hi = min(hi, 0xFFFF)
        if lo > hi:
            return
        # Split around surrogates [0xD800, 0xDFFF]
        if lo <= 0xD7FF:
            self._add_3byte_range_segment(src, dst, lo, min(hi, 0xD7FF))
        if hi >= 0xE000:
            self._add_3byte_range_segment(src, dst, max(lo, 0xE000), hi)

    def _add_3byte_range_segment(self, src: int, dst: int, lo: int, hi: int) -> None:
        """Add 3-byte UTF-8 NFA transitions for [lo, hi] (no surrogate check)."""
        if lo > hi:
            return
        lo_b1 = ((lo >> 12) & 0x0F) | 0xE0
        hi_b1 = ((hi >> 12) & 0x0F) | 0xE0
        for b1 in range(lo_b1, hi_b1 + 1):
            cp_b1_base = (b1 & 0x0F) << 12
            sub_lo = max(lo, cp_b1_base)
            sub_hi = min(hi, cp_b1_base | 0xFFF)
            if sub_lo > sub_hi:
                continue
            lo_b2 = ((sub_lo >> 6) & 0x3F) | 0x80
            hi_b2 = ((sub_hi >> 6) & 0x3F) | 0x80
            inter_b1 = self._new()
            self._tr(src, b1, inter_b1)
            for b2 in range(lo_b2, hi_b2 + 1):
                cp_b2_base = cp_b1_base | ((b2 & 0x3F) << 6)
                sub2_lo = max(sub_lo, cp_b2_base)
                sub2_hi = min(sub_hi, cp_b2_base | 0x3F)
                lo_b3 = (sub2_lo & 0x3F) | 0x80
                hi_b3 = (sub2_hi & 0x3F) | 0x80
                inter_b2 = self._new()
                self._tr(inter_b1, b2, inter_b2)
                for b3 in range(lo_b3, hi_b3 + 1):
                    self._tr(inter_b2, b3, dst)

    def _add_4byte_range(self, src: int, dst: int, lo: int, hi: int) -> None:
        """Add NFA transitions accepting 4-byte UTF-8 code points in [lo, hi]."""
        lo = max(lo, 0x10000)
        hi = min(hi, 0x10FFFF)
        if lo > hi:
            return
        lo_b1 = ((lo >> 18) & 0x07) | 0xF0
        hi_b1 = ((hi >> 18) & 0x07) | 0xF0
        for b1 in range(lo_b1, hi_b1 + 1):
            cp_b1_base = (b1 & 0x07) << 18
            sub_lo = max(lo, cp_b1_base)
            sub_hi = min(hi, cp_b1_base | 0x3FFFF)
            if sub_lo > sub_hi:
                continue
            lo_b2 = ((sub_lo >> 12) & 0x3F) | 0x80
            hi_b2 = ((sub_hi >> 12) & 0x3F) | 0x80
            inter_b1 = self._new()
            self._tr(src, b1, inter_b1)
            for b2 in range(lo_b2, hi_b2 + 1):
                cp_b2_base = cp_b1_base | ((b2 & 0x3F) << 12)
                sub2_lo = max(sub_lo, cp_b2_base)
                sub2_hi = min(sub_hi, cp_b2_base | 0xFFF)
                if sub2_lo > sub2_hi:
                    continue
                lo_b3 = ((sub2_lo >> 6) & 0x3F) | 0x80
                hi_b3 = ((sub2_hi >> 6) & 0x3F) | 0x80
                inter_b2 = self._new()
                self._tr(inter_b1, b2, inter_b2)
                for b3 in range(lo_b3, hi_b3 + 1):
                    cp_b3_base = cp_b2_base | ((b3 & 0x3F) << 6)
                    sub3_lo = max(sub2_lo, cp_b3_base)
                    sub3_hi = min(sub2_hi, cp_b3_base | 0x3F)
                    lo_b4 = (sub3_lo & 0x3F) | 0x80
                    hi_b4 = (sub3_hi & 0x3F) | 0x80
                    inter_b3 = self._new()
                    self._tr(inter_b2, b3, inter_b3)
                    for b4 in range(lo_b4, hi_b4 + 1):
                        self._tr(inter_b3, b4, dst)

    # -- fragment builders ---------------------------------------------------

    def build(self, parsed) -> tuple[int, int]:
        return self._seq(list(parsed))

    def _seq(self, items: list) -> tuple[int, int]:
        if not items:
            s0, s1 = self._new(), self._new()
            self._eps(s0, s1)
            return s0, s1
        frag = self._elem(items[0])
        for item in items[1:]:
            nxt = self._elem(item)
            self._eps(frag[1], nxt[0])
            frag = (frag[0], nxt[1])
        return frag

    def _elem(self, item) -> tuple[int, int]:
        op, val = item
        if op == LITERAL:
            return self._literal(val)
        if op == NOT_LITERAL:
            return self._not_literal(val)
        if op == ANY:
            return self._any()
        if op == IN:
            return self._in(val)
        if op == BRANCH:
            return self._branch(val)
        if op == SUBPATTERN:
            return self._subpat(val)
        if op in (MAX_REPEAT, MIN_REPEAT):
            return self._repeat(val)
        if op == AT:
            # ^, $, \A, \Z are no-ops for fullmatch
            if val in (AT_BEGINNING, AT_END, AT_BEGINNING_STRING, AT_END_STRING):
                s0, s1 = self._new(), self._new()
                self._eps(s0, s1)
                return s0, s1
            raise ValueError(f"Unsupported anchor: {val}")
        raise ValueError(f"Unsupported regex op: {op}")

    def _literal(self, code: int) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        self._add_char(s0, code, s1)
        return s0, s1

    def _not_literal(self, code: int) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        if self.bytes_mode:
            excluded: set[int] = set()
            if self.case_insensitive:
                for alt in self._case_variants(code):
                    if alt <= 255:
                        excluded.add(alt)
            elif code <= 255:
                excluded.add(code)
            for b in range(256):
                if b not in excluded:
                    self._tr(s0, b, s1)
            return s0, s1
        excluded_ascii: set[int] = set()
        excluded_nonascii: set[int] = set()
        if self.case_insensitive:
            for alt in self._case_variants(code):
                if alt <= 127:
                    excluded_ascii.add(alt)
                else:
                    excluded_nonascii.add(alt)
        else:
            if code <= 127:
                excluded_ascii.add(code)
            else:
                excluded_nonascii.add(code)
        for b in range(128):
            if b not in excluded_ascii:
                self._tr(s0, b, s1)
        # Accept multi-byte UTF-8 except excluded code-points
        self._add_utf8_paths(s0, s1, excluded_cps=excluded_nonascii)
        return s0, s1

    def _any(self) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        if self.bytes_mode:
            # Bytes mode: match any single byte (except \n in non-dotall mode)
            for b in range(256):
                if self.dot_all or b != 10:
                    self._tr(s0, b, s1)
            return s0, s1
        # ASCII bytes (except \n in non-dotall mode)
        for b in range(128):
            if self.dot_all or b != 10:  # '\n' == 10
                self._tr(s0, b, s1)
        # Multi-byte UTF-8 sequences (all valid non-ASCII code points)
        self._add_utf8_paths(s0, s1)
        return s0, s1

    def _in(self, items: list) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        negate = any(op == NEGATE for op, _ in items)
        # Byte transitions (all 256 in bytes mode, ASCII only in UTF-8 mode)
        for b in self._char_set(items):
            self._tr(s0, b, s1)
        if not self.bytes_mode:
            if negate:
                # Add multi-byte UTF-8 paths for non-excluded code points
                self._add_utf8_paths(s0, s1, excluded_cps=self._excluded_cps(items))
            else:
                # Add multi-byte UTF-8 paths for included non-ASCII code points
                for code in self._non_ascii_codes(items):
                    self._add_char(s0, code, s1)
                # Handle 3-byte and 4-byte code-point ranges not covered above
                for op, value in items:
                    if op == RANGE:
                        lo, hi = value
                        if hi >= 0x800:
                            self._add_3byte_range(s0, s1, lo, hi)
                        if hi >= 0x10000:
                            self._add_4byte_range(s0, s1, lo, hi)
                # NOT-categories (e.g. \S, \D, \W) implicitly include all non-ASCII
                if self._needs_utf8_accept(items):
                    self._add_utf8_paths(s0, s1)
        return s0, s1

    @staticmethod
    def _needs_utf8_accept(items: list) -> bool:
        """Return True if any item implicitly accepts non-ASCII characters."""
        for op, value in items:
            if op == CATEGORY and value in (
                CATEGORY_NOT_DIGIT,
                CATEGORY_NOT_WORD,
                CATEGORY_NOT_SPACE,
            ):
                return True
        return False

    def _branch(self, val) -> tuple[int, int]:
        _, branches = val
        s0, s1 = self._new(), self._new()
        for branch in branches:
            frag = self._seq(branch)
            self._eps(s0, frag[0])
            self._eps(frag[1], s1)
        return s0, s1

    def _subpat(self, val) -> tuple[int, int]:
        _gid, _add, _del, pattern = val
        return self._seq(pattern)

    def _repeat(self, val) -> tuple[int, int]:
        lo, hi, pattern = val
        if lo == 0 and hi == MAXREPEAT:
            return self._star(pattern)
        if lo == 1 and hi == MAXREPEAT:
            return self._plus(pattern)
        if lo == 0 and hi == 1:
            return self._opt(pattern)
        return self._bounded(pattern, lo, hi)

    def _star(self, pattern) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        f = self._seq(pattern)
        self._eps(s0, f[0])
        self._eps(s0, s1)
        self._eps(f[1], f[0])
        self._eps(f[1], s1)
        return s0, s1

    def _plus(self, pattern) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        f = self._seq(pattern)
        self._eps(s0, f[0])
        self._eps(f[1], f[0])
        self._eps(f[1], s1)
        return s0, s1

    def _opt(self, pattern) -> tuple[int, int]:
        s0, s1 = self._new(), self._new()
        f = self._seq(pattern)
        self._eps(s0, f[0])
        self._eps(s0, s1)
        self._eps(f[1], s1)
        return s0, s1

    def _bounded(self, pattern, lo: int, hi: int) -> tuple[int, int]:
        frags = []
        for _ in range(lo):
            frags.append(self._seq(pattern))
        if hi == MAXREPEAT:
            frags.append(self._star(pattern))
        else:
            for _ in range(hi - lo):
                frags.append(self._opt(pattern))
        if not frags:
            s0, s1 = self._new(), self._new()
            self._eps(s0, s1)
            return s0, s1
        result = frags[0]
        for f in frags[1:]:
            self._eps(result[1], f[0])
            result = (result[0], f[1])
        return result


# ---------------------------------------------------------------------------
# NFA → DFA (subset construction)
# ---------------------------------------------------------------------------

def _epsilon_closure(states: set[int], eps: dict[int, set[int]]) -> frozenset[int]:
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in eps.get(s, ()):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


def _nfa_to_dfa(
    builder: NFABuilder, start: int, accept: int
) -> dict:
    # Pre-compute per-state outgoing bytes for fast lookup
    state_bytes: dict[int, set[int]] = {}
    for (src, b) in builder.transitions:
        state_bytes.setdefault(src, set()).add(b)

    initial = _epsilon_closure({start}, builder.epsilon)

    dfa_map: dict[frozenset[int], int] = {initial: 0}
    dfa_trans: dict[tuple[int, int], int] = {}
    dfa_accept: set[int] = set()

    if accept in initial:
        dfa_accept.add(0)

    counter = 1
    queue: deque[frozenset[int]] = deque([initial])

    while queue:
        cur = queue.popleft()
        cur_id = dfa_map[cur]

        # Collect all bytes that have at least one transition from *cur*
        active_bytes: set[int] = set()
        for s in cur:
            if s in state_bytes:
                active_bytes |= state_bytes[s]

        for b in active_bytes:
            nxt_nfa: set[int] = set()
            for s in cur:
                nxt_nfa |= builder.transitions.get((s, b), set())
            if not nxt_nfa:
                continue
            nxt_closure = _epsilon_closure(nxt_nfa, builder.epsilon)
            if nxt_closure not in dfa_map:
                if counter >= _MAX_DFA_STATES:
                    raise ValueError(
                        f"DFA state limit exceeded ({_MAX_DFA_STATES})"
                    )
                dfa_map[nxt_closure] = counter
                counter += 1
                queue.append(nxt_closure)
                if accept in nxt_closure:
                    dfa_accept.add(dfa_map[nxt_closure])
            dfa_trans[(cur_id, b)] = dfa_map[nxt_closure]

    return {
        "num_states": counter,
        "initial": 0,
        "accept": dfa_accept,
        "transitions": dfa_trans,
    }


# ---------------------------------------------------------------------------
# DFA helpers
# ---------------------------------------------------------------------------

def _add_dead_state(dfa: dict) -> dict:
    """Ensure every (state, byte) pair has a transition.

    Missing transitions are redirected to an explicit *dead state* that
    loops to itself on every byte.
    """
    n = dfa["num_states"]
    trans = dict(dfa["transitions"])
    dead = n
    needs_dead = False

    for s in range(n):
        for b in range(256):
            if (s, b) not in trans:
                trans[(s, b)] = dead
                needs_dead = True

    if not needs_dead:
        return dfa

    for b in range(256):
        trans[(dead, b)] = dead

    return {
        "num_states": dead + 1,
        "initial": dfa["initial"],
        "accept": dfa["accept"],
        "transitions": trans,
    }


def _minimize_dfa(dfa: dict) -> dict:
    """Hopcroft's DFA minimisation."""
    n = dfa["num_states"]
    accept = dfa["accept"]
    trans = dfa["transitions"]
    initial = dfa["initial"]

    if n <= 1:
        return dfa

    # Pre-compute reverse transitions: (byte, target) -> set of sources
    reverse: dict[tuple[int, int], set[int]] = {}
    for (s, b), t in trans.items():
        reverse.setdefault((b, t), set()).add(s)

    accept_s = set(accept)
    non_accept_s = set(range(n)) - accept_s

    P: list[set[int]] = []
    if accept_s:
        P.append(accept_s)
    if non_accept_s:
        P.append(non_accept_s)

    W: list[set[int]] = [set(s) for s in P]

    while W:
        A = W.pop()
        for c in range(256):
            # X = states that transition to some state in A on byte c
            X: set[int] = set()
            for a in A:
                X |= reverse.get((c, a), set())
            if not X:
                continue

            new_P: list[set[int]] = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    new_P.append(inter)
                    new_P.append(diff)
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    new_P.append(Y)
            P = new_P

    state_map: dict[int, int] = {}
    for i, part in enumerate(P):
        for s in part:
            state_map[s] = i

    new_trans: dict[tuple[int, int], int] = {}
    new_accept: set[int] = set()
    for (s, b), t in trans.items():
        new_trans[(state_map[s], b)] = state_map[t]
    for s in accept:
        new_accept.add(state_map[s])

    return {
        "num_states": len(P),
        "initial": state_map[initial],
        "accept": new_accept,
        "transitions": new_trans,
    }


def _renumber(dfa: dict) -> dict:
    """Renumber states so that dead-state = 0, initial-state = 1, and all
    accepting states are grouped at the end (indices >= *first_accept*).

    This ordering allows the generated C matcher to use the efficient test
    ``return state >= first_accept;`` instead of a boolean lookup table.
    """
    n = dfa["num_states"]
    trans = dfa["transitions"]
    accept = dfa["accept"]
    initial = dfa["initial"]

    # Find dead state: non-accepting, all self-loops
    dead = None
    for s in range(n):
        if s in accept:
            continue
        if all(trans.get((s, b), s) == s for b in range(256)):
            dead = s
            break

    mapping: dict[int, int] = {}
    nxt = 0
    # 1. dead state → 0
    if dead is not None:
        mapping[dead] = 0
        nxt = 1
    # 2. initial state (if non-accepting) → next slot
    if initial not in mapping and initial not in accept:
        mapping[initial] = nxt
        nxt += 1
    # 3. remaining non-accepting states
    for s in range(n):
        if s not in mapping and s not in accept:
            mapping[s] = nxt
            nxt += 1
    # All states with index < first_accept are non-accepting.
    first_accept = nxt
    # 4. initial state (if accepting) → first accepting slot
    if initial not in mapping:
        mapping[initial] = nxt
        nxt += 1
    # 5. remaining accepting states
    for s in range(n):
        if s not in mapping:
            mapping[s] = nxt
            nxt += 1

    new_trans: dict[tuple[int, int], int] = {}
    new_accept: set[int] = set()
    for (s, b), t in trans.items():
        new_trans[(mapping[s], b)] = mapping[t]
    for s in accept:
        new_accept.add(mapping[s])

    return {
        "num_states": n,
        "initial": mapping[initial],
        "accept": new_accept,
        "first_accept": first_accept,
        "transitions": new_trans,
    }


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _preprocess_pattern(pattern: str) -> str:
    r"""Pre-process PCRE2 / re2 pattern extensions for ``sre_parse``.

    * ``\x{NNNN}`` hex escapes → literal characters.
    * POSIX character classes ``[:alpha:]`` etc. → equivalent ranges.
    """
    # POSIX class mapping (inside [...])
    _POSIX = {
        "[:alnum:]": "a-zA-Z0-9",
        "[:^alnum:]": "^a-zA-Z0-9",
        "[:alpha:]": "a-zA-Z",
        "[:^alpha:]": "^a-zA-Z",
        "[:ascii:]": "\\x00-\\x7f",
        "[:^ascii:]": "^\\x00-\\x7f",
        "[:blank:]": " \\t",
        "[:^blank:]": "^ \\t",
        "[:cntrl:]": "\\x00-\\x1f\\x7f",
        "[:^cntrl:]": "^\\x00-\\x1f\\x7f",
        "[:digit:]": "0-9",
        "[:^digit:]": "^0-9",
        "[:graph:]": "!-~",
        "[:^graph:]": "^!-~",
        "[:lower:]": "a-z",
        "[:^lower:]": "^a-z",
        "[:print:]": " -~",
        "[:^print:]": "^ -~",
        "[:punct:]": "!-/:-@[-`{-~",
        "[:^punct:]": "^!-/:-@[-`{-~",
        "[:space:]": " \\t\\n\\r\\f\\v",
        "[:^space:]": "^ \\t\\n\\r\\f\\v",
        "[:upper:]": "A-Z",
        "[:^upper:]": "^A-Z",
        "[:word:]": "a-zA-Z0-9_",
        "[:^word:]": "^a-zA-Z0-9_",
        "[:xdigit:]": "0-9a-fA-F",
        "[:^xdigit:]": "^0-9a-fA-F",
    }

    # Replace POSIX classes inside character classes
    for posix, replacement in _POSIX.items():
        pattern = pattern.replace(posix, replacement)

    # Replace \x{NNNN} hex escapes
    result: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == "\\x{":
            end = pattern.find("}", i + 3)
            if end != -1:
                hex_str = pattern[i + 3 : end]
                try:
                    code = int(hex_str, 16)
                    if code <= 0x10FFFF:
                        result.append(chr(code))
                        i = end + 1
                        continue
                except ValueError:
                    pass
        result.append(pattern[i])
        i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_nfa(
    pattern: str, flags: str = "", encoding: str = "utf8"
) -> tuple[NFABuilder, int, int]:
    """Shared front-end: parse *pattern* and build a Thompson NFA."""
    if encoding not in ("utf8", "bytes"):
        raise ValueError(f"encoding must be 'utf8' or 'bytes', got {encoding!r}")
    re_flags = 0
    ci = False
    da = False
    for f in flags:
        if f == "i":
            ci = True
            re_flags |= re.IGNORECASE
        elif f == "s":
            da = True
            re_flags |= re.DOTALL
        elif f == "m":
            re_flags |= re.MULTILINE
        # Note: 'x' (verbose) is intentionally NOT mapped to re.VERBOSE.
        # re2 does not support verbose mode; the flag is a PCRE2 test
        # directive that we ignore.

    pattern = _preprocess_pattern(pattern)
    parsed = sre_parse.parse(pattern, re_flags)

    builder = NFABuilder(case_insensitive=ci, dot_all=da, bytes_mode=(encoding == "bytes"))
    start, accept = builder.build(parsed)
    return builder, start, accept


def compile_regex(pattern: str, flags: str = "", encoding: str = "utf8") -> dict:
    """Compile *pattern* to a minimised, renumbered DFA.

    Returns a dict with ``num_states``, ``initial``, ``accept``, and
    ``transitions`` keys.

    Parameters
    ----------
    encoding:
        ``"utf8"`` (default) treats the input as UTF-8 text; ``.`` matches
        one Unicode code point and character classes are Unicode-aware.
        ``"bytes"`` uses pure byte semantics: ``.`` matches one arbitrary
        byte and all literals/classes operate on raw byte values (0-255).
    """
    builder, start, accept = _build_nfa(pattern, flags, encoding)

    dfa = _nfa_to_dfa(builder, start, accept)
    dfa = _add_dead_state(dfa)
    dfa = _minimize_dfa(dfa)
    dfa = _renumber(dfa)
    return dfa


def _reduce_to_position_nfa(
    builder: NFABuilder, start: int, accept: int
) -> dict:
    """Reduce a Thompson NFA to a position NFA (Glushkov-equivalent).

    Eliminates epsilon-only intermediate states so that bit-positions are
    assigned only to states with outgoing byte transitions and the accept
    state.  After construction the NFA is pruned: positions that are not
    forward-reachable from the initial set or that cannot reach any accept
    position are removed, and the remaining positions are renumbered
    contiguously starting at 0.

    Returns the same dict format as :func:`compile_nfa`.
    """

    # -- Step 1: identify byte-states (states with outgoing byte transitions)
    byte_states: set[int] = set()
    for src, _b in builder.transitions:
        byte_states.add(src)

    # All relevant positions: byte-states plus the accept state.
    # The accept state must keep a bit so the accept check works even when
    # it has no outgoing byte transitions itself.
    all_positions = sorted(byte_states | {accept})
    old_to_new = {s: i for i, s in enumerate(all_positions)}
    num_positions = len(all_positions)

    # Pre-compute per-state outgoing bytes for fast lookup (same as _nfa_to_dfa)
    state_bytes: dict[int, set[int]] = {}
    for src, b in builder.transitions:
        state_bytes.setdefault(src, set()).add(b)

    # -- Step 2: initial bitset (ε-closure of start ∩ relevant positions)
    start_closure = _epsilon_closure({start}, builder.epsilon)
    initial_bits = 0
    for s in start_closure:
        if s in old_to_new:
            initial_bits |= 1 << old_to_new[s]

    # -- Step 3: accept bitset
    # A position is accepting if the Thompson accept state is ε-reachable
    # from it *or* if it is the accept state itself.
    accept_bits = 0
    for s in all_positions:
        if accept in _epsilon_closure({s}, builder.epsilon):
            accept_bits |= 1 << old_to_new[s]

    # -- Step 4: transition masks using only the reduced position set
    trans_masks: list[dict[int, int]] = [{} for _ in range(num_positions)]
    for old_s in all_positions:
        new_pos = old_to_new[old_s]
        masks: dict[int, int] = {}
        for b in state_bytes.get(old_s, ()):
            dests = builder.transitions.get((old_s, b), set())
            if dests:
                closure = _epsilon_closure(dests, builder.epsilon)
                bits = 0
                for s in closure:
                    if s in old_to_new:
                        bits |= 1 << old_to_new[s]
                if bits:
                    masks[b] = bits
        trans_masks[new_pos] = masks

    # -- Step 5: reachability pruning using bitset arithmetic
    #  (a) Forward reachability from the initial set.
    reachable = initial_bits
    prev = 0
    while reachable != prev:
        prev = reachable
        p = reachable
        while p:
            pos = (p & -p).bit_length() - 1  # lowest set bit
            p &= p - 1
            for bits in trans_masks[pos].values():
                reachable |= bits

    #  (b) Backward reachability: positions from which some accept position
    #      is reachable (including accept positions themselves).
    #      Build reverse adjacency first.
    reverse_adj: list[int] = [0] * num_positions
    for p in range(num_positions):
        for bits in trans_masks[p].values():
            b = bits
            while b:
                q = (b & -b).bit_length() - 1
                b &= b - 1
                reverse_adj[q] |= 1 << p

    productive = accept_bits
    prev = 0
    while productive != prev:
        prev = productive
        p = productive
        while p:
            pos = (p & -p).bit_length() - 1
            p &= p - 1
            productive |= reverse_adj[pos]

    keep_mask = reachable & productive
    keep = []
    m = keep_mask
    while m:
        pos = (m & -m).bit_length() - 1
        m &= m - 1
        keep.append(pos)

    # -- Step 6: renumber if any positions were pruned
    if len(keep) < num_positions:
        remap = {old: new for new, old in enumerate(keep)}
        num_positions = len(keep)

        new_initial = 0
        for old in keep:
            if initial_bits & (1 << old):
                new_initial |= 1 << remap[old]

        new_accept = 0
        for old in keep:
            if accept_bits & (1 << old):
                new_accept |= 1 << remap[old]

        new_trans: list[dict[int, int]] = []
        for old in keep:
            masks = trans_masks[old]
            remapped: dict[int, int] = {}
            for b, bits in masks.items():
                new_bits = 0
                remaining = bits & keep_mask
                while remaining:
                    q = (remaining & -remaining).bit_length() - 1
                    remaining &= remaining - 1
                    new_bits |= 1 << remap[q]
                if new_bits:
                    remapped[b] = new_bits
            new_trans.append(remapped)

        initial_bits = new_initial
        accept_bits = new_accept
        trans_masks = new_trans

    return {
        "num_positions": num_positions,
        "initial": initial_bits,
        "accept": accept_bits,
        "trans_masks": trans_masks,
    }


def compile_nfa(pattern: str, flags: str = "", encoding: str = "utf8") -> dict:
    """Compile *pattern* to NFA data for bit-parallel simulation.

    The Thompson NFA is first reduced to a position NFA (Glushkov-
    equivalent) so that only states with actual byte transitions occupy
    bit-positions.  This typically halves the number of required bits
    compared to the raw Thompson NFA.

    Returns a dict with:

    * ``num_positions`` – total number of NFA states (bit-positions).
    * ``initial`` – Python int bitset of initially active positions.
    * ``accept`` – Python int bitset for the accept position.
    * ``trans_masks`` – list of length *num_positions*; each entry is a
      ``dict[int, int]`` mapping byte values to destination bitsets.
    """
    builder, start, accept = _build_nfa(pattern, flags, encoding)

    nfa = _reduce_to_position_nfa(builder, start, accept)

    if nfa["num_positions"] > _MAX_BITNFA_POSITIONS:
        raise ValueError(
            f"NFA has {nfa['num_positions']} positions "
            f"(limit: {_MAX_BITNFA_POSITIONS}); "
            "use the DFA engine for this pattern"
        )

    return nfa
