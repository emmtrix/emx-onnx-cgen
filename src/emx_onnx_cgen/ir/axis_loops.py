"""Loop nests for kernels that reduce over a range of axes and then write every element."""

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
