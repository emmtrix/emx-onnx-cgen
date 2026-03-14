"""Command-line interface for emx-regex-cgen."""

from __future__ import annotations

import argparse
import sys

from .codegen import generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="emx-regex-cgen",
        description="Generate C code that performs a fullmatch for a regular expression.",
    )
    parser.add_argument("pattern", help="Regular expression pattern")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--emit-main",
        action="store_true",
        help="Also emit a main() function (exit 0=match, 1=no match, 2=error)",
    )
    parser.add_argument(
        "--prefix",
        default="regex",
        help="Prefix for all generated C identifiers: arrays and match function (default: regex)",
    )
    parser.add_argument(
        "--flags",
        default="",
        help="Regex flags: i (case-insensitive), s (dot-all), m (multiline), x (verbose)",
    )
    parser.add_argument(
        "--encoding",
        choices=["utf8", "bytes"],
        default="utf8",
        help="Input encoding: utf8 (default, Unicode-aware) or bytes (raw byte semantics)",
    )
    parser.add_argument(
        "--row-dedup",
        choices=["yes", "no", "auto"],
        default="auto",
        help=(
            "Transition-row deduplication: yes (always), no (never), "
            "auto (when table exceeds --size-threshold; default)"
        ),
    )
    parser.add_argument(
        "--alphabet-compression",
        choices=["yes", "no", "auto"],
        default="auto",
        help=(
            "Alphabet compression into equivalence classes: yes (always), "
            "no (never), auto (when table exceeds --size-threshold; default)"
        ),
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=8192,
        help=(
            "Table-size threshold (cells = states × 256) for auto mode "
            "(default: 8192)"
        ),
    )
    parser.add_argument(
        "--early-exit",
        choices=["yes", "no"],
        default="no",
        help=(
            "Emit early-exit check inside the DFA loop: yes (break when dead "
            "state is reached), no (always process full input; default)"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["dfa", "bitnfa"],
        default="dfa",
        help=(
            "Backend engine: dfa (table-driven minimised DFA; default), "
            "bitnfa (bit-parallel NFA)"
        ),
    )

    args = parser.parse_args(argv)

    try:
        code = generate(
            args.pattern,
            flags=args.flags,
            emit_main=args.emit_main,
            prefix=args.prefix,
            encoding=args.encoding,
            engine=args.engine,
            row_dedup=args.row_dedup,
            alphabet_compression=args.alphabet_compression,
            size_threshold=args.size_threshold,
            early_exit=(args.early_exit == "yes"),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output == "-":
        sys.stdout.write(code.render())
    else:
        with open(args.output, "w") as fh:
            fh.write(code.render())


if __name__ == "__main__":
    main()
