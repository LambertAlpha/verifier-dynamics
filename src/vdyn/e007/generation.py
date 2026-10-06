"""Batched sampling that keeps the stop token (E007c). mlx_lm.batch_generate drops the stop token
from returned completions, so policy-gradient training on its output never trains the decision to
end a response; that bug affected E007 and E007b."""

from collections.abc import Callable, Sequence
from typing import Any

from mlx_lm.generate import BatchGenerator


def generate(
    model: Any,
    stop_ids: Sequence[int],
    prompts: list[list[int]],
    max_tokens: int,
    sampler: Callable[..., Any],
    completion_batch_size: int = 64,
) -> tuple[list[list[int]], list[str | None]]:
    """Returns (completion token ids incl. the stop token when stopped, finish reasons)."""
    g = BatchGenerator(model, stop_tokens=[[t] for t in stop_ids], sampler=sampler,
                       completion_batch_size=completion_batch_size)  # fmt: skip
    uids = g.insert(prompts, [max_tokens] * len(prompts))
    toks: dict[int, list[int]] = {u: [] for u in uids}
    reason: dict[int, str | None] = {}
    while responses := g.next_generated():
        for r in responses:
            toks[r.uid].append(int(r.token))
            if r.finish_reason is not None:
                reason[r.uid] = r.finish_reason
    g.close()
    return [toks[u] for u in uids], [reason.get(u) for u in uids]
