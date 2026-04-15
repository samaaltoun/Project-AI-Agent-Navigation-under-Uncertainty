"""
Belief map for POMDP navigation under uncertainty.

Each cell in the grid is modelled as a categorical distribution over
K = 11 object types (MiniGrid OBJECT_TO_IDX, indices 0–10).  The
distribution is updated step-by-step using Bayesian inference with
a configurable sensor model (true-positive rate) and temporal decay.

MiniGrid object type indices (OBJECT_TO_IDX):
    0: unseen   1: empty   2: wall    3: floor   4: door
    5: key      6: ball    7: box     8: goal    9: lava   10: agent

Sensor model
─────────────
    tpr  — True Positive Rate: probability that the sensor reports the
           correct object type.  Defaults to  1 − 0.9 × noise_prob.
    fpr  — False Positive Rate per alternative type: (1 − tpr) / 9.

    The likelihood for observation z given true type t:
        P(z | t) = tpr      if t == z
        P(z | t) = fpr      if t != z   (for each of the 9 other types)

Temporal decay
──────────────
    decay — After each step, every cell's belief drifts toward the
            uninformative uniform prior:
                belief ← (1 − decay) · belief + decay · uniform
            A decay of 0.0 means perfect memory; 0.05 means 5% drift
            per step, so unobserved cells gradually forget.

Usage (eval loop):
    bmap = BeliefMap(grid_size=15, noise_prob=0.2, decay=0.02)
    bmap.reset()

    # After every env.reset() / env.step():
    raw_img = obs["image"]                          # (H, W, 3) from MiniGrid
    bmap.update(raw_img, base.agent_pos, base.agent_dir)

    h  = bmap.mean_entropy()                        # scalar, bits
    kl = bmap.kl_from_truth(base.grid)              # scalar, bits
"""

from __future__ import annotations

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_TYPES = 11       # type indices 0–10
VALID_TYPES = 10     # types 1–10 are observable object types (0 = "unseen")
MAX_ENTROPY = np.log2(VALID_TYPES)   # entropy of uniform prior ≈ 3.32 bits

# Uniform prior over valid types (used for reset and decay target)
_UNIFORM_PRIOR = np.zeros(NUM_TYPES, dtype=np.float64)
_UNIFORM_PRIOR[1:] = 1.0 / VALID_TYPES


# ── BeliefMap ─────────────────────────────────────────────────────────────────
class BeliefMap:
    """
    Categorical belief map over a (grid_size × grid_size) world.

    belief[x, y] ∈ Δ^NUM_TYPES  — probability vector over object types for
    cell (x, y) in world coordinates.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (e.g. 15).
    noise_prob : float
        Sensor noise probability matching the SensorNoiseWrapper value.
        Used to derive the default true-positive rate when *tpr* is None.
    tpr : float or None
        True Positive Rate of the sensor.  If None, derived as
        ``1.0 − 0.9 × noise_prob``.
    decay : float
        Per-step decay toward the uniform prior.  0.0 = no forgetting.
    view_size : int
        Agent egocentric view size (default 7 for a 7×7 window).
    """

    NUM_TYPES = NUM_TYPES

    def __init__(
        self,
        grid_size: int,
        noise_prob: float = 0.0,
        view_size: int = 7,
        tpr: float | None = None,
        decay: float = 0.0,
    ):
        self.grid_size = grid_size
        self.noise_prob = float(noise_prob)
        self.view_size = view_size
        self.decay = float(decay)

        # True positive rate
        if tpr is not None:
            self.tpr = float(tpr)
        else:
            # Derive from noise_prob:  TPR = 1 − 0.9·p
            self.tpr = 1.0 - 0.9 * self.noise_prob

        # False positive rate per alternative type
        self.fpr = (1.0 - self.tpr) / max(VALID_TYPES - 1, 1)

        # belief[x, y, t] = P(cell (x,y) has type t)
        self.belief = np.empty((grid_size, grid_size, NUM_TYPES), dtype=np.float64)
        self.reset()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset belief to the uninformative prior at episode start.

        Prior: uniform over valid object types 1–10  (1/10 each).
        Type 0 (unseen) is excluded — it is not a real object type.
        """
        self.belief[:] = 0.0
        self.belief[:, :, 1:] = 1.0 / VALID_TYPES   # 0.1 for types 1–10

    def update(
        self,
        obs_image: np.ndarray,
        agent_pos,
        agent_dir: int,
    ) -> None:
        """
        Apply temporal decay, then Bayesian-update every visible cell.

        Parameters
        ----------
        obs_image : np.ndarray, shape (view_size, view_size, 3)
            Raw egocentric image.  obs_image[lx, ly, 0] is the object-type
            index at local position (lx, ly).
        agent_pos : array-like of length 2
            Agent's current world (x, y) position.
        agent_dir : int
            Agent's facing direction: 0=right, 1=down, 2=left, 3=up.
        """
        # ── Temporal decay ────────────────────────────────────────────────────
        if self.decay > 0:
            self.belief[:] = (
                (1.0 - self.decay) * self.belief
                + self.decay * _UNIFORM_PRIOR[np.newaxis, np.newaxis, :]
            )

        # ── Bayesian update for visible cells ─────────────────────────────────
        vs = self.view_size
        ax, ay = int(agent_pos[0]), int(agent_pos[1])

        for lx in range(vs):
            for ly in range(vs):
                z = int(obs_image[lx, ly, 0])

                # type 0 → cell is outside the visibility cone; skip
                if z == 0:
                    continue

                wx, wy = self._local_to_world(lx, ly, ax, ay, agent_dir)

                # guard against edge effects / rounding near map boundaries
                if not (0 <= wx < self.grid_size and 0 <= wy < self.grid_size):
                    continue

                self._bayesian_update(wx, wy, z)

    def entropy_map(self) -> np.ndarray:
        """
        Shannon entropy of the belief at every cell (base-2, in bits).

        Returns shape (grid_size, grid_size).
        0 bits → completely certain;  log₂(10) ≈ 3.32 bits → maximally uncertain.
        """
        b = self.belief
        with np.errstate(divide="ignore", invalid="ignore"):
            log_b = np.where(b > 1e-12, np.log2(b), 0.0)
        return -(b * log_b).sum(axis=2)

    def mean_entropy(self) -> float:
        """Mean entropy (bits) averaged over the whole grid — scalar metric."""
        return float(self.entropy_map().mean())

    def kl_from_truth(self, ground_truth_grid) -> float:
        """
        Mean KL divergence from belief to the true world state.

        For a one-hot ground-truth distribution at each cell:
            KL( truth ‖ belief ) = −log₂( belief[true_type] )

        Returns float — mean KL divergence in bits.
        """
        from minigrid.core.constants import OBJECT_TO_IDX

        total = 0.0
        gs = self.grid_size

        for wx in range(gs):
            for wy in range(gs):
                cell = ground_truth_grid.get(wx, wy)
                if cell is None:
                    true_type = OBJECT_TO_IDX["empty"]   # 1
                else:
                    true_type = OBJECT_TO_IDX.get(cell.type, 1)

                p = max(float(self.belief[wx, wy, true_type]), 1e-10)
                total += -np.log2(p)

        return total / (gs * gs)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _local_to_world(
        self, lx: int, ly: int, ax: int, ay: int, agent_dir: int
    ) -> tuple[int, int]:
        """Convert local egocentric coordinates to world (x, y) coordinates."""
        s = self.view_size - 1     # 6
        h = self.view_size // 2    # 3

        if agent_dir == 0:    # right
            return ax + s - ly, ay - h + lx
        elif agent_dir == 1:  # down
            return ax + h - lx, ay + s - ly
        elif agent_dir == 2:  # left
            return ax - s + ly, ay + h - lx
        else:                 # up  (agent_dir == 3)
            return ax - h + lx, ay - s + ly

    def _bayesian_update(self, wx: int, wy: int, z: int) -> None:
        """
        Update the categorical belief at cell (wx, wy) given observation z.

        Sensor model (parametrised by TPR)
        ───────────────────────────────────
        P(z | true_type = t) =
            tpr                    if t == z     (correct detection)
            (1 − tpr) / 9         otherwise     (false positive)

        Update rule:
            posterior[t] ∝ P(z | t) · prior[t]
        """
        tpr = self.tpr
        fpr = self.fpr

        prior = self.belief[wx, wy]

        # Apply likelihood: default = false-positive rate for all types
        posterior = prior * fpr

        # Boost likelihood for the observed type z
        if 1 <= z <= 10:
            posterior[z] = prior[z] * tpr

        # Type 0 (unseen) stays at 0 because prior[0] == 0 after reset

        total = posterior.sum()
        if total > 1e-12:
            self.belief[wx, wy] = posterior / total
        # If total ≈ 0 (degenerate), keep current belief to avoid NaN.
