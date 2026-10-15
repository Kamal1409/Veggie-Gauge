"""Stage E, pairwise mode: rank one crop group and compare neighbours with a margin delta."""

from __future__ import annotations

BIGGER, SMALLER, SIMILAR = "BIGGER", "SMALLER", "SIMILAR"


def relative_difference(a: float, b: float) -> float:
    """How much bigger the larger value is, relative to the smaller one.

    For example, 221 g against 139 g gives 0.59, i.e. 59% bigger.

    Args:
        a: First size (grams or mm²).
        b: Second size, in the same unit.

    Returns:
        `(larger - smaller) / smaller`, or infinity if the smaller value is zero or less.
    """
    lo, hi = min(a, b), max(a, b)
    return float("inf") if lo <= 0 else (hi - lo) / lo


def pair_verdict(a: float, b: float, delta: float) -> str:
    """Compare one item with another using the SIMILAR margin.

    Args:
        a: Size of the item being described.
        b: Size of the item it is compared with.
        delta: Relative difference at or below which the two count as SIMILAR.

    Returns:
        "BIGGER" or "SMALLER" for `a` relative to `b`, or "SIMILAR".
    """
    if relative_difference(a, b) <= delta:
        return SIMILAR
    return BIGGER if a > b else SMALLER


def _pct(d: float) -> float | None:
    """Turn a relative difference into a rounded percentage for the JSON output.

    Args:
        d: Relative difference, as returned by `relative_difference`.

    Returns:
        The percentage to one decimal place, or None if the difference is infinite.
    """
    return round(100 * d, 1) if d != float("inf") else None


def rank_group(values: list[float], delta: float = 0.08) -> dict:
    """Rank the items of one crop group and compare each neighbouring pair.

    Items are ranked from biggest (1) down. Each pair of neighbours in that ranking is
    then compared with the margin `delta`, so a group of three or more reads as a
    ladder, e.g. "1 is SIMILAR to 2, which is BIGGER than 3".

    Args:
        values: Size of each item (usually estimated weight), in input order.
        delta: Relative difference at or below which neighbours count as SIMILAR.

    Returns:
        A dict with:
            ranks: Rank of each item, in input order.
            adjacent: One entry per neighbouring pair, in rank order, with the indices of
                the higher and lower item, the verdict and the difference in percent.
            pairwise: For exactly two items, each item's BIGGER / SMALLER / SIMILAR verdict;
                None otherwise.
            difference_pct: Largest versus smallest, in percent (None for a single item).
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=True)
    ranks = [order.index(i) + 1 for i in range(n)]
    adjacent = []
    for hi, lo in zip(order, order[1:]):
        d = relative_difference(values[hi], values[lo])
        adjacent.append({"higher": hi, "lower": lo, "verdict": BIGGER if d > delta else SIMILAR, "difference_pct": _pct(d)})
    pairwise = None
    if n == 2:
        pairwise = [pair_verdict(values[0], values[1], delta), pair_verdict(values[1], values[0], delta)]
    return {
        "ranks": ranks,
        "adjacent": adjacent,
        "pairwise": pairwise,
        "difference_pct": _pct(relative_difference(max(values), min(values))) if n >= 2 else None,
    }
