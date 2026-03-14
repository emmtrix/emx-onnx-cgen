"""Generate C source code for a bit-parallel NFA matcher."""

from __future__ import annotations

from .result import GeneratedCode


def _mask_lit(value: int, bits: int) -> str:
    """Format *value* as a C unsigned hex literal of the chosen width."""
    if bits <= 8:
        return f"0x{value & 0xff:02x}u"
    if bits <= 16:
        return f"0x{value & 0xffff:04x}u"
    return f"0x{value & 0xffffffff:08x}u"


def _min_type(max_val: int) -> tuple[str, int]:
    """Return the narrowest unsigned C type that can hold *max_val*."""
    if max_val <= 0xFF:
        return "uint8_t", 8
    if max_val <= 0xFFFF:
        return "uint16_t", 16
    return "uint32_t", 32


def _byte_designator(b: int) -> str:
    """Return a C array designator for byte *b*."""
    if 32 <= b <= 126:
        c = chr(b)
        if c == "'":
            return "['\\'']"
        if c == "\\":
            return "['\\\\']"
        return f"['{c}']"
    return f"[{b}]"


def _byte_cmp_lit(b: int) -> str:
    """Return a C character literal suitable for use in a comparison (b == …)."""
    if 32 <= b <= 126:
        c = chr(b)
        if c == "'":
            return "'\\''"
        if c == "\\":
            return "'\\\\'"
        return f"'{c}'"
    return str(b)


def _build_main(func_name: str) -> str:
    lines = [
        "int main(int argc, char *argv[]) {",
        "    if (argc != 2) {",
        '        fprintf(stderr, "Usage: %s <input>\\n", argv[0]);',
        "        return 2;",
        "    }",
        f"    return {func_name}(argv[1], strlen(argv[1])) ? 0 : 1;",
        "}",
    ]
    return "\n".join(lines)


def _emit_single_word(
    nfa: dict,
    state_t: str,
    bits: int,
    *,
    prefix: str,
    emit_main: bool,
    pattern: str | None,
    flags: str,
    encoding: str,
) -> GeneratedCode:
    num_pos = nfa["num_positions"]
    initial = nfa["initial"]
    accept = nfa["accept"]
    trans_masks = nfa["trans_masks"]

    active_positions = [p for p in range(num_pos) if trans_masks[p]]

    # Pre-classify each active position: can it be replaced by an inline
    # comparison expression?  A position qualifies when its transition map
    # has exactly one non-zero entry.  In that case we emit
    #   next |= ((b == c) ? mask : 0u);
    # instead of a 256-byte lookup table.
    inline: dict[int, tuple[int, int]] = {}  # p -> (byte, mask_value)
    for p in active_positions:
        masks = trans_masks[p]
        non_zero = [(b, v) for b, v in masks.items() if v != 0]
        if len(non_zero) == 1:
            b_val, v = non_zero[0]
            inline[p] = (b_val, v)

    includes = ["stddef.h", "stdbool.h", "stdint.h"]
    if emit_main:
        includes.extend(["string.h", "stdio.h"])

    global_lines: list[str] = []
    for p in active_positions:
        if p in inline:
            continue  # replaced by inline expression; no table needed
        masks = trans_masks[p]
        max_val = max(masks.values(), default=0)
        pos_type, pos_bits = _min_type(max_val)
        non_zero = sorted((b, v) for b, v in masks.items() if v != 0)
        if not non_zero:
            row_str = "{ 0 }"
        else:
            entries = [
                f"{_byte_designator(b)} = {_mask_lit(v, pos_bits)}"
                for b, v in non_zero
            ]
            row_str = "{ " + ", ".join(entries) + " }"
        global_lines.append(f"static const {pos_type} {prefix}_trans_{p}[256] = {row_str};")
    globals_str = "\n".join(global_lines)

    match_lines: list[str] = []
    if pattern is not None:
        match_lines.append(f'/* regex:    "{pattern}"')
        match_lines.append(f' * flags:    "{flags}"')
        match_lines.append(f" * encoding: {encoding}")
        match_lines.append(f" * engine:   bitnfa ({state_t})")
        match_lines.append(" */")
    else:
        match_lines.append(f"/* engine: bitnfa ({state_t}) */")

    func_name = f"{prefix}_match"
    match_lines.append(f"bool {func_name}(const char *input, size_t len) {{")
    match_lines.append(f"    {state_t} state = {_mask_lit(initial, bits)};")
    match_lines.append("    for (size_t i = 0; i < len; i++) {")
    match_lines.append("        unsigned char b = (unsigned char)input[i];")
    match_lines.append(f"        {state_t} next = 0;")
    for p in active_positions:
        state_check = f"state & {_mask_lit(1 << p, bits)}"
        if p in inline:
            b_val, v = inline[p]
            rhs = f"((b == {_byte_cmp_lit(b_val)}) ? {_mask_lit(v, bits)} : 0u)"
        else:
            rhs = f"{prefix}_trans_{p}[b]"
        match_lines.append(f"        if ({state_check}) next |= {rhs};")
    match_lines.append("        state = next;")
    match_lines.append("    }")
    match_lines.append(f"    return (state & {_mask_lit(accept, bits)}) != 0;")
    match_lines.append("}")
    match_str = "\n".join(match_lines)

    return GeneratedCode(
        includes=includes,
        globals=globals_str,
        match_function=match_str,
        main_function=_build_main(func_name) if emit_main else None,
    )


def _emit_array_variant(
    nfa: dict,
    num_words: int,
    *,
    prefix: str,
    emit_main: bool,
    pattern: str | None,
    flags: str,
    encoding: str,
) -> GeneratedCode:
    num_pos = nfa["num_positions"]
    initial = nfa["initial"]
    accept = nfa["accept"]
    trans_masks = nfa["trans_masks"]

    def to_words(v: int) -> list[int]:
        return [(v >> (w * 32)) & 0xFFFFFFFF for w in range(num_words)]

    init_words = to_words(initial)
    accept_words = to_words(accept)
    active_positions = [p for p in range(num_pos) if trans_masks[p]]

    # Pre-classify: positions with a single non-zero entry can be emitted as
    # an inline ternary comparison instead of a 2-D lookup table.
    inline: dict[int, tuple[int, int]] = {}  # p -> (byte, mask_value)
    for p in active_positions:
        masks = trans_masks[p]
        non_zero = [(b, v) for b, v in masks.items() if v != 0]
        if len(non_zero) == 1:
            b_val, v = non_zero[0]
            inline[p] = (b_val, v)

    includes = ["stddef.h", "stdbool.h", "stdint.h"]
    if emit_main:
        includes.extend(["string.h", "stdio.h"])

    global_lines: list[str] = []
    for p in active_positions:
        if p in inline:
            continue  # replaced by inline expression; no table needed
        masks = trans_masks[p]
        non_zero = sorted((b, v) for b, v in masks.items() if v != 0)
        if not non_zero:
            global_lines.append(
                f"static const uint32_t {prefix}_trans_{p}[256][{num_words}] = {{ 0 }};"
            )
            continue

        entries = []
        for b, v in non_zero:
            words = to_words(v)
            word_str = ", ".join(f"0x{w:08x}u" for w in words)
            entries.append(f"{_byte_designator(b)} = {{ {word_str} }}")
        row_str = "{ " + ", ".join(entries) + " }"
        global_lines.append(
            f"static const uint32_t {prefix}_trans_{p}[256][{num_words}] = {row_str};"
        )
    globals_str = "\n".join(global_lines)

    match_lines: list[str] = []
    if pattern is not None:
        match_lines.append(f'/* regex:    "{pattern}"')
        match_lines.append(f' * flags:    "{flags}"')
        match_lines.append(f" * encoding: {encoding}")
        match_lines.append(f" * engine:   bitnfa (uint32_t[{num_words}])")
        match_lines.append(" */")
    else:
        match_lines.append(f"/* engine: bitnfa (uint32_t[{num_words}]) */")

    func_name = f"{prefix}_match"
    match_lines.append(f"bool {func_name}(const char *input, size_t len) {{")
    for w in range(num_words):
        match_lines.append(f"    uint32_t s{w} = 0x{init_words[w]:08x}u;")
    match_lines.append("    for (size_t i = 0; i < len; i++) {")
    match_lines.append("        unsigned char b = (unsigned char)input[i];")
    for w in range(num_words):
        match_lines.append(f"        uint32_t n{w} = 0;")
    for p in active_positions:
        word_idx = p // 32
        bit_idx = p % 32
        bit_mask = f"0x{1 << bit_idx:08x}u"
        if p in inline:
            b_val, v = inline[p]
            words = to_words(v)
            assigns = " ".join(
                f"n{w} |= ((b == {_byte_cmp_lit(b_val)}) ? 0x{words[w]:08x}u : 0u);"
                for w in range(num_words)
                if words[w] != 0
            )
        else:
            assigns = " ".join(
                f"n{w} |= {prefix}_trans_{p}[b][{w}];" for w in range(num_words)
            )
        match_lines.append(f"        if (s{word_idx} & {bit_mask}) {{ {assigns} }}")
    for w in range(num_words):
        match_lines.append(f"        s{w} = n{w};")
    match_lines.append("    }")

    accept_parts = []
    for w in range(num_words):
        if accept_words[w]:
            accept_parts.append(f"(s{w} & 0x{accept_words[w]:08x}u)")
    match_lines.append(
        f"    return ({' || '.join(accept_parts)}) != 0;" if accept_parts else "    return false;"
    )
    match_lines.append("}")
    match_str = "\n".join(match_lines)

    return GeneratedCode(
        includes=includes,
        globals=globals_str,
        match_function=match_str,
        main_function=_build_main(func_name) if emit_main else None,
    )


def generate_bitnfa_c_code(
    nfa: dict,
    *,
    prefix: str = "regex",
    emit_main: bool = False,
    pattern: str | None = None,
    flags: str = "",
    encoding: str = "utf8",
) -> GeneratedCode:
    """Emit C code for a bit-parallel NFA matcher."""
    num_pos = nfa["num_positions"]

    if num_pos <= 8:
        state_t, bits = "uint8_t", 8
    elif num_pos <= 16:
        state_t, bits = "uint16_t", 16
    elif num_pos <= 32:
        state_t, bits = "uint32_t", 32
    else:
        return _emit_array_variant(
            nfa,
            (num_pos + 31) // 32,
            prefix=prefix,
            emit_main=emit_main,
            pattern=pattern,
            flags=flags,
            encoding=encoding,
        )

    return _emit_single_word(
        nfa,
        state_t,
        bits,
        prefix=prefix,
        emit_main=emit_main,
        pattern=pattern,
        flags=flags,
        encoding=encoding,
    )
