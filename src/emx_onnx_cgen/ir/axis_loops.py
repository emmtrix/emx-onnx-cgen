"""Loop nests that let generated kernels index their array parameters directly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AxisLoopNest:
    """The loops of a kernel that reduces over the axes `[reduce_begin, reduce_end)`.

    The kept axes become the outer loops, the reduced axes the inner reduction loops. Every
    axis has its own loop variable `i<axis>`, so that the kernel indexes the arrays directly
    (`x[i0][i1][i2]`) instead of through a flat pointer and a computed offset.
    """

    loop_vars: tuple[str, ...]
    loop_sizes: tuple[str | int, ...]
    reduce_vars: tuple[str, ...]
    reduce_sizes: tuple[str | int, ...]
    # Index of the current element, e.g. `[i0][i1][i2]`
    index: str
    # Index of the first element of the reduced axes, e.g. `[i0][0][i2]`
    first_index: str

    @staticmethod
    def over(
        shape_exprs: Sequence[str | int], reduce_begin: int, reduce_end: int
    ) -> "AxisLoopNest":
        rank = len(shape_exprs)
        if not 0 <= reduce_begin < reduce_end <= rank:
            raise ValueError(
                f"Reduced axes [{reduce_begin}, {reduce_end}) are invalid for rank {rank}"
            )

        def reduced(dim: int) -> bool:
            return reduce_begin <= dim < reduce_end

        kept = [dim for dim in range(rank) if not reduced(dim)]
        return AxisLoopNest(
            loop_vars=tuple(f"i{dim}" for dim in kept),
            loop_sizes=tuple(shape_exprs[dim] for dim in kept),
            reduce_vars=tuple(f"i{dim}" for dim in range(reduce_begin, reduce_end)),
            reduce_sizes=tuple(shape_exprs[reduce_begin:reduce_end]),
            index="".join(f"[i{dim}]" for dim in range(rank)),
            first_index="".join(
                f"[{0 if reduced(dim) else f'i{dim}'}]" for dim in range(rank)
            ),
        )

    @property
    def zero_starts(self) -> tuple[int, ...]:
        return (0,) * len(self.reduce_vars)

    @property
    def after_first_starts(self) -> tuple[int, ...]:
        """Starts of a pass that is seeded with the first element: a single reduced axis skips
        it, several reduced axes visit it again, which a max-like reduction tolerates.
        """
        if len(self.reduce_vars) == 1:
            return (1,)
        return self.zero_starts


@dataclass(frozen=True)
class ConcatPart:
    """One part of a tensor that is concatenated from (or split into) parts along `axis`.

    The loops run over the part; the part is indexed with `part_index` (`[i0][i1][i2]`), the
    whole tensor with `whole_index`, which shifts the axis by the sizes of the preceding parts
    (`[i0][i1 + 3][i2]`).
    """

    loop_vars: tuple[str, ...]
    loop_sizes: tuple[str | int, ...]
    part_index: str
    whole_index: str

    @staticmethod
    def along(
        axis: int, part_shapes: Sequence[Sequence[str | int]]
    ) -> tuple["ConcatPart | None", ...]:
        """The parts in order; `None` for a part that is statically empty."""
        parts: list[ConcatPart | None] = []
        offsets: list[str | int] = []
        for shape in part_shapes:
            offset = _sum_expr(offsets)
            offsets.append(shape[axis])
            if any(dim == 0 for dim in shape):
                parts.append(None)
                continue
            loop_vars = tuple(f"i{dim}" for dim in range(len(shape)))
            parts.append(
                ConcatPart(
                    loop_vars=loop_vars,
                    loop_sizes=tuple(shape),
                    part_index="".join(f"[{var}]" for var in loop_vars),
                    whole_index="".join(
                        (
                            f"[{var} + {offset}]"
                            if dim == axis and offset != "0"
                            else f"[{var}]"
                        )
                        for dim, var in enumerate(loop_vars)
                    ),
                )
            )
        return tuple(parts)


def _sum_expr(terms: Sequence[str | int]) -> str:
    """`terms` summed, with the static terms folded into one constant."""
    constant = sum(term for term in terms if isinstance(term, int))
    symbolic = [str(term) for term in terms if not isinstance(term, int)]
    if constant or not symbolic:
        symbolic.append(str(constant))
    return " + ".join(symbolic)
