"""emx-regex-cgen: A regex to C code generator.

Generates portable, static C code that performs fullmatch on a given
regular expression pattern.  The generated code uses a table-driven DFA
approach with a branch-free inner loop suitable for auto-vectorisation.
"""

from .codegen import generate
from .result import GeneratedCode

__all__ = ["generate", "GeneratedCode"]
