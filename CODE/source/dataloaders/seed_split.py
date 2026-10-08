"""
seed_split.py
=============
Fixes the determinism bug documented in docs/DESIGN.md §6.

In the SynFoC repo (dataloaders/dataloader.py) the labeled set is selected as
    lb_idxs = list(range(lb_num))
over a `sorted()` file list, with a hardcoded module constant `SEED = 1212`.
Consequence: passing --seed 1337/42/2024 does NOT change which images are
labeled, nor the split — so a "3-seed" run is effectively ONE data seed, which
is why we observed an implausible 0.04 macro-Dice std.

This module provides a drop-in replacement that makes the labeled subset (and,
optionally, the labeled/unlabeled partition) depend on the RUN seed, and logs
the selected indices so the split is auditable.

INTEGRATION (in train.py, replacing the lines around the original
`lb_idxs = list(range(lb_num))`):

    from dataloaders.seed_split import select_labeled_idxs
    lb_idxs, unlabeled_idxs = select_labeled_idxs(
        data_num=data_num, lb_num=lb_num, seed=args.seed, log=logging)

That single change is what makes our reported std meaningful. Run the
ablation/seed sweep ONLY after this patch is in place.
"""
from __future__ import annotations
import random
from typing import List, Tuple, Optional


def select_labeled_idxs(data_num: int,
                        lb_num: int,
                        seed: int,
                        log: Optional[object] = None) -> Tuple[List[int], List[int]]:
    """Choose `lb_num` labeled indices out of `data_num`, dependent on `seed`.

    Uses an independent random.Random(seed) so it does not perturb the global
    RNG state that controls weight init / augmentation (keeps everything else
    comparable across the rest of the pipeline).

    Returns (lb_idxs_sorted, unlabeled_idxs_sorted).
    """
    assert 0 < lb_num <= data_num, f"bad lb_num={lb_num} for data_num={data_num}"
    rng = random.Random(seed)
    lb_idxs = sorted(rng.sample(range(data_num), lb_num))
    lb_set = set(lb_idxs)
    unlabeled_idxs = [i for i in range(data_num) if i not in lb_set]

    msg = (f"[seed_split] seed={seed} data_num={data_num} lb_num={lb_num} "
           f"labeled_idxs={lb_idxs}")
    if log is not None:
        try:
            log.info(msg)
        except Exception:
            print(msg)
    else:
        print(msg)
    return lb_idxs, unlabeled_idxs


def _selftest():
    # Different seeds -> different labeled sets (the whole point).
    a, _ = select_labeled_idxs(100, 20, seed=1337)
    b, _ = select_labeled_idxs(100, 20, seed=42)
    c, _ = select_labeled_idxs(100, 20, seed=1337)
    assert a == c, "same seed must reproduce the same split"
    assert a != b, "different seeds must give different splits"
    # partition is valid
    lb, ulb = select_labeled_idxs(50, 10, seed=7)
    assert len(lb) == 10 and len(ulb) == 40
    assert set(lb).isdisjoint(set(ulb))
    assert set(lb) | set(ulb) == set(range(50))
    print("[seed_split] self-test PASSED")


if __name__ == "__main__":
    _selftest()
