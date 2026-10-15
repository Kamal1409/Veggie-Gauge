"""Stage C: group items by crop and set aside the ones that cannot be used.

Every item is classified on its own, so a photo may hold any mix of crops. Items are only
ever compared with items of the same crop: a photo with 2 tomatoes and 3 chamagadda gives
two separate comparisons, never a tomato-vs-chamagadda one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OK = "ok"
NO_ITEMS = "no_items"
MESSAGES = {OK: "ok", NO_ITEMS: "no confidently identified vegetables"}


@dataclass
class Grouping:
    """Which items go into which crop group, and which are left out.

    Attributes:
        groups: Detection indices for each crop, in left-to-right order.
        uncertain: Items whose crop probability was below τ; they are never compared.
        excluded: Items cut off by the frame, with the reason they were left out.
    """

    groups: dict[str, list[int]] = field(default_factory=dict)  # crop -> detection indices
    uncertain: list[int] = field(default_factory=list)  # crop probability below tau
    excluded: dict[int, str] = field(default_factory=dict)  # truncated items -> reason

    @property
    def status(self) -> str:
        """The overall status: "ok" if at least one group exists, otherwise "no_items"."""
        return OK if self.groups else NO_ITEMS

    @property
    def message(self) -> str:
        """A short human-readable version of `status`."""
        return MESSAGES[self.status]


def group_items(crops: list[str], probs: list[float], flags: list[list[str]], tau: float = 0.6, exclude_truncated: bool = True) -> Grouping:
    """Sort items into crop groups, setting aside the ones that cannot be used.

    An item cut off by the frame is excluded, since its size would be wrong. An item
    the classifier is unsure about is marked uncertain. Everything else joins the group
    for its predicted crop, so items of different crops are never compared.

    Args:
        crops: Predicted crop of each item.
        probs: Classifier probability for each prediction.
        flags: Segmentation flags of each item (e.g. "touches_border").
        tau: Lowest probability for an item to join a group.
        exclude_truncated: Whether items touching the frame edge are left out.

    Returns:
        The grouping.
    """
    g = Grouping()
    for i, (crop, p, fl) in enumerate(zip(crops, probs, flags)):
        if exclude_truncated and "touches_border" in fl:
            g.excluded[i] = "truncated: touches image border"
        elif p < tau:
            g.uncertain.append(i)
        else:
            g.groups.setdefault(crop, []).append(i)
    return g
