"""Generate C source code from a compiled DFA or bit-parallel NFA."""

from __future__ import annotations

from .compiler import compile_nfa, compile_regex
from .result import GeneratedCode


def _should_apply(mode: str, table_size: int, threshold: int) -> bool:
    """Resolve an ``"auto"``/``"yes"``/``"no"`` mode flag."""
    if mode == "yes":
        return True
    if mode == "no":
        return False
    # auto - apply only when the uncompressed table exceeds the threshold
    return table_size > threshold


def _compute_alphabet_classes(
    n: int, trans: dict
) -> tuple[list[int], int, list[int]]:
    """Compute byte equivalence classes for alphabet compression."""
    sig_to_class: dict[tuple[int, ...], int] = {}
    class_map: list[int] = [0] * 256
    class_reps: list[int] = []

    for b in range(256):
        sig = tuple(trans.get((s, b), 0) for s in range(n))
        if sig not in sig_to_class:
            sig_to_class[sig] = len(class_reps)
            class_reps.append(b)
        class_map[b] = sig_to_class[sig]

    return class_map, len(class_reps), class_reps


def generate_c_code(
    dfa: dict,
    *,
    prefix: str = "regex",
    emit_main: bool = False,
    row_dedup: str = "auto",
    alphabet_compression: str = "auto",
    size_threshold: int = 8192,
    pattern: str | None = None,
    flags: str = "",
    encoding: str = "utf8",
    early_exit: bool = False,
) -> GeneratedCode:
    """Emit C code for a table-driven DFA matcher."""
    n = dfa["num_states"]
    initial = dfa["initial"]
    first_accept = dfa["first_accept"]
    trans = dfa["transitions"]

    if n <= 256:
        state_t = "uint8_t"
    elif n <= 65536:
        state_t = "uint16_t"
    else:
        state_t = "uint32_t"

    table_size = n * 256
    do_dedup = _should_apply(row_dedup, table_size, size_threshold)
    do_alphabet = _should_apply(alphabet_compression, table_size, size_threshold)

    if do_alphabet:
        class_map, num_classes, class_reps = _compute_alphabet_classes(n, trans)
        if num_classes >= 256:
            do_alphabet = False

    if do_alphabet:
        num_cols = num_classes
    else:
        num_cols = 256
        class_reps = list(range(256))

    all_rows: list[tuple[int, ...]] = []
    for s in range(n):
        row = tuple(trans.get((s, class_reps[c]), 0) for c in range(num_cols))
        all_rows.append(row)

    if do_dedup:
        unique_rows: list[tuple[int, ...]] = []
        row_index: dict[tuple[int, ...], int] = {}
        state_to_row: list[int] = []
        for s in range(n):
            row = all_rows[s]
            if row not in row_index:
                row_index[row] = len(unique_rows)
                unique_rows.append(row)
            state_to_row.append(row_index[row])
        num_unique = len(unique_rows)
        has_row_map = num_unique < n
    else:
        unique_rows = all_rows
        state_to_row = list(range(n))
        num_unique = n
        has_row_map = False

    row_to_states: list[list[int]] = [[] for _ in range(num_unique)]
    for s, r in enumerate(state_to_row):
        row_to_states[r].append(s)

    includes = ["stddef.h", "stdbool.h", "stdint.h"]
    if emit_main:
        includes.extend(["string.h", "stdio.h"])

    global_lines: list[str] = []

    if do_alphabet:
        alpha_t = "uint8_t" if num_classes <= 256 else "uint16_t"
        global_lines.append(f"static const {alpha_t} {prefix}_alphabet[256] = {{")
        for row_start in range(0, 256, 16):
            row_end = min(row_start + 16, 256)
            vals = ", ".join(str(class_map[b]) for b in range(row_start, row_end))
            if row_end < 256:
                global_lines.append(f"    {vals},")
            else:
                global_lines.append(f"    {vals}")
        global_lines.append("};")
        global_lines.append("")

    global_lines.append(
        f"static const {state_t} {prefix}_transitions[{num_unique}][{num_cols}] = {{"
    )
    for row_idx, row in enumerate(unique_rows):
        states = row_to_states[row_idx]
        comment = (
            f"/* state {states[0]} */"
            if len(states) == 1
            else f"/* states {', '.join(str(s) for s in states)} */"
        )
        non_zero = [(i, v) for i, v in enumerate(row) if v != 0]
        if not non_zero:
            row_str = "{ 0 }"
        else:
            entries = []
            for i, v in non_zero:
                if do_alphabet:
                    idx = f"[{i}]"
                elif 32 <= i <= 126:
                    c = chr(i)
                    if c == "'":
                        idx = "['\\'']"
                    elif c == "\\":
                        idx = "['\\\\']"
                    else:
                        idx = f"['{c}']"
                else:
                    idx = f"[{i}]"
                entries.append(f"{idx} = {v}")
            row_str = "{ " + ", ".join(entries) + " }"
        global_lines.append(f"    {comment} {row_str},")
    global_lines.append("};")

    if has_row_map:
        if num_unique <= 256:
            row_t = "uint8_t"
        elif num_unique <= 65536:
            row_t = "uint16_t"
        else:
            row_t = "uint32_t"
        global_lines.append("")
        global_lines.append(f"static const {row_t} {prefix}_row_map[{n}] = {{")
        global_lines.append(f"    {', '.join(str(i) for i in state_to_row)}")
        global_lines.append("};")

    globals_str = "\n".join(global_lines)

    match_lines: list[str] = []
    if pattern is not None:
        match_lines.append(f'/* regex:                "{pattern}"')
        match_lines.append(f' * flags:                "{flags}"')
        match_lines.append(f' * encoding:             {encoding}')
        match_lines.append(f' * alphabet-compression: {"yes" if do_alphabet else "no"}')
        match_lines.append(f' * row-deduplication:    {"yes" if do_dedup else "no"}')
        match_lines.append(f' * early-exit:           {"yes" if early_exit else "no"}')
        match_lines.append(" */")
    else:
        match_lines.append(f'/* alphabet-compression: {"yes" if do_alphabet else "no"}')
        match_lines.append(f' * row-deduplication:    {"yes" if do_dedup else "no"}')
        match_lines.append(f' * early-exit:           {"yes" if early_exit else "no"}')
        match_lines.append(" */")

    col_expr = (
        f"{prefix}_alphabet[(unsigned char)input[i]]"
        if do_alphabet
        else "(unsigned char)input[i]"
    )
    row_expr = f"{prefix}_row_map[state]" if has_row_map else "state"
    func_name = f"{prefix}_match"

    match_lines.append(f"bool {func_name}(const char *input, size_t len) {{")
    match_lines.append(f"    {state_t} state = {initial};")
    match_lines.append("    for (size_t i = 0; i < len; i++) {")
    match_lines.append(f"        state = {prefix}_transitions[{row_expr}][{col_expr}];")
    if early_exit:
        match_lines.append("        if (state == 0) break;")
    match_lines.append("    }")
    match_lines.append(f"    return state >= {first_accept};")
    match_lines.append("}")
    match_str = "\n".join(match_lines)

    if emit_main:
        main_lines = [
            "int main(int argc, char *argv[]) {",
            "    if (argc != 2) {",
            '        fprintf(stderr, "Usage: %s <input>\\n", argv[0]);',
            "        return 2;",
            "    }",
            f"    return {func_name}(argv[1], strlen(argv[1])) ? 0 : 1;",
            "}",
        ]
        main_str: str | None = "\n".join(main_lines)
    else:
        main_str = None

    return GeneratedCode(
        includes=includes,
        globals=globals_str,
        match_function=match_str,
        main_function=main_str,
    )


def generate(
    pattern: str,
    flags: str = "",
    *,
    emit_main: bool = False,
    prefix: str = "regex",
    encoding: str = "utf8",
    engine: str = "dfa",
    row_dedup: str = "auto",
    alphabet_compression: str = "auto",
    size_threshold: int = 8192,
    early_exit: bool = False,
) -> GeneratedCode:
    """High-level API: compile *pattern* and return generated C code."""
    if engine == "bitnfa":
        from .codegen_bitnfa import generate_bitnfa_c_code

        nfa = compile_nfa(pattern, flags, encoding=encoding)
        return generate_bitnfa_c_code(
            nfa,
            prefix=prefix,
            emit_main=emit_main,
            pattern=pattern,
            flags=flags,
            encoding=encoding,
        )

    dfa = compile_regex(pattern, flags, encoding=encoding)
    return generate_c_code(
        dfa,
        prefix=prefix,
        emit_main=emit_main,
        row_dedup=row_dedup,
        alphabet_compression=alphabet_compression,
        size_threshold=size_threshold,
        pattern=pattern,
        flags=flags,
        encoding=encoding,
        early_exit=early_exit,
    )
