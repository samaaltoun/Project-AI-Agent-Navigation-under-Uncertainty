"""
Belief map for POMDP navigation under uncertainty.
Each cell in the grid is modelled as a categorical distribution over
K = 11 object types (MiniGrid OBJECT_TO_IDX, indices 0–10).  The
distribution is updated step-by-step using Bayesian inference, with the
noise model from SensorNoiseWrapper as the likelihood.
MiniGrid object type indices (OBJECT_TO_IDX):
    0: unseen   1: empty   2: wall    3: floor   4: door
    5: key      6: ball    7: box     8: goal    9: lava   10: agent

Theoretical framing: the whole task is a POMDP.  The agent acts under
partial observability (7×7 egocentric view in a 13×13 world) and sensor
noise.  The belief map is the POMDP belief state b(s) — a distribution
over possible world states.
Usage (eval loop):
    bmap = BeliefMap(grid_size=13, noise_prob=0.2)
    bmap.reset()                                   # start of episode

    #After every env.reset() / env.step():
    raw_img = obs[0].transpose(1, 2, 0)            #(C,H,W) → (H,W,C)
    bmap.update(raw_img, base.agent_pos, base.agent_dir)

    h = bmap.mean_entropy()                        #scalar, decreases as
                                                   #agent explores
    kl = bmap.kl_from_truth(base.grid)             # scalar, higher noise
                                                   #   → higher KL
"""

from __future__ import annotations
import numpy as np

#Constants
NUM_TYPES = 11       #type indices 0–10
VALID_TYPES = 10     #types 1–10 are observable object types (check above)
MAX_ENTROPY = np.log2(VALID_TYPES)   #entropy of uniform prior roughly 3.32 bits


#beliefmap
class BeliefMap:
    """
    Categorical belief map over a (grid_size × grid_size) world.
    belief[x, y] ∈ Δ^NUM_TYPES  — probability vector over object types for
    cell (x, y) in world coordinates.
    Parameters
    grid_size : int
        Side length of the square grid (e.g. 13).
    noise_prob : float
        Sensor noise probability matching the SensorNoiseWrapper value.
        0.0 → perfectly reliable observations;
        1.0 → every visible cell is randomly re-labelled.
    view_size : int
        Agent egocentric view size (default 7 for a 7×7 window).
    """

    NUM_TYPES = NUM_TYPES

    def __init__(self, grid_size: int, noise_prob: float = 0.0, view_size: int = 7):
        self.grid_size = grid_size
        self.noise_prob = float(noise_prob)
        self.view_size = view_size

        #belief[x, y, t] = P(cell (x,y) has type t)
        self.belief = np.empty((grid_size, grid_size, NUM_TYPES), dtype=np.float64)
        self.reset()

    #pub api

    def reset(self) -> None:
        """
        Reset belief to the uninformative prior at episode start.
        Prior: uniform over valid object types 1–10  (1/10 each).
        Type 0 (unseen) is excluded — it is not a real object type.
        """
        self.belief[:] = 0.0
        self.belief[:, :, 1:] = 1.0 / VALID_TYPES   #0.1 for types 1–10

    def update(
        self,
        obs_image: np.ndarray,
        agent_pos,
        agent_dir: int,
    ) -> None:
        """
        Bayesian update for every visible cell in the current observation.
        Parameters
        ----------
        obs_image : np.ndarray, shape (view_size, view_size, 3)
            Raw egocentric image as returned by ImgObsWrapper (before
            VecTransposeImage).  obs_image[lx, ly, 0] is the object-type
            index at local position (lx, ly).
        agent_pos : array-like of length 2
            Agent's current world (x, y) position.
        agent_dir : int
            Agent's facing direction: 0=right, 1=down, 2=left, 3=up.
        """
        vs = self.view_size
        ax, ay = int(agent_pos[0]), int(agent_pos[1])

        for lx in range(vs):
            for ly in range(vs):
                z = int(obs_image[lx, ly, 0])

                #type 0 so cell is outside the visibility cone; skip
                if z == 0:
                    continue

                wx, wy = self._local_to_world(lx, ly, ax, ay, agent_dir)

                #guard against edge effects / rounding near map boundaries
                if not (0 <= wx < self.grid_size and 0 <= wy < self.grid_size):
                    continue

                self._bayesian_update(wx, wy, z)

    def entropy_map(self) -> np.ndarray:
        """
        Shannon entropy of the belief at every cell (base-2, in bits).
        H[x, y] = −∑_t  B[x,y,t] · log₂ B[x,y,t]

        Returns
        np.ndarray, shape (grid_size, grid_size)
            Entropy per cell.  0 bits so completely certain;
            log₂(10) roughly 3.32 bits so maximally uncertain (uniform prior).
        """
        b = self.belief
        with np.errstate(divide="ignore", invalid="ignore"):
            log_b = np.where(b > 1e-12, np.log2(b), 0.0)
        return -(b * log_b).sum(axis=2)

    def mean_entropy(self) -> float:
        """Mean entropy (bits) averaged over the whole grid - scalar metric."""
        return float(self.entropy_map().mean())

    def kl_from_truth(self, ground_truth_grid) -> float:
        """
        Mean KL divergence from belief to the true world state.
        For a one-hot ground-truth distribution at each cell:
            KL( truth ‖ belief ) = −log₂( belief[true_type] )

        This measures how surprised the belief map is by the true
        object layout.  Higher noise → weaker Bayesian updates →
        higher KL divergence.

        Parameters
        ground_truth_grid : minigrid.core.grid.Grid
            The environment's internal Grid object (env.unwrapped.grid).

        Returns
        float
            Mean KL divergence in bits, averaged over all grid cells.
        """
        from minigrid.core.constants import OBJECT_TO_IDX

        total = 0.0
        gs = self.grid_size

        for wx in range(gs):
            for wy in range(gs):
                cell = ground_truth_grid.get(wx, wy)
                if cell is None:
                    true_type = OBJECT_TO_IDX["empty"]   #1
                else:
                    true_type = OBJECT_TO_IDX.get(cell.type, 1)

                # P(true_type) in our belief — clamp to avoid log(0)
                p = max(float(self.belief[wx, wy, true_type]), 1e-10)
                total += -np.log2(p)

        return total / (gs * gs)

    #helpers internal

    def _local_to_world(
        self, lx: int, ly: int, ax: int, ay: int, agent_dir: int
    ) -> tuple[int, int]:
        """
        Convert local egocentric coordinates to world (x, y) coordinates.
        Derivation
        MiniGrid's gen_obs_grid():
          1. Slices the world grid starting at (topX, topY) — a corner that
             depends on the agent's position and facing direction.
          2. Applies (agent_dir + 1) rotate_left operations to the slice.

        To invert:
          - Apply (agent_dir + 1) rotate_right operations:
              (x, y)  →rotate_right→  (size−1−y, x)   [for a size×size grid]
          - Add (topX, topY).

        Formulae (verified for all four directions, view_size=7):
            s = view_size − 1 = 6
            h = view_size // 2 = 3

            dir=0 (right) :  wx = ax + s − ly,  wy = ay − h + lx
            dir=1 (down)  :  wx = ax + h − lx,  wy = ay + s − ly
            dir=2 (left)  :  wx = ax − s + ly,  wy = ay + h − lx
            dir=3 (up)    :  wx = ax − h + lx,  wy = ay − s + ly
        """
        s = self.view_size - 1     #6
        h = self.view_size // 2    #3

        if agent_dir == 0:    #right
            return ax + s - ly, ay - h + lx
        elif agent_dir == 1:  #down
            return ax + h - lx, ay + s - ly
        elif agent_dir == 2:  #left
            return ax - s + ly, ay + h - lx
        else:                 # up, agent dir 3 
            return ax - h + lx, ay - s + ly

    def _bayesian_update(self, wx: int, wy: int, z: int) -> None:
        """
        Update the categorical belief at cell (wx, wy) given observation z.

        Sensor model (from SensorNoiseWrapper)
        With probability  (1 − noise_prob): observation equals true type.
        With probability  noise_prob      : observation is uniform over
                                            types 1–10 (regardless of truth).

        Likelihood:
            P(z | true_type = t) =
                (1 − p) + p/10    if t == z
                p/10              otherwise

        Where p = self.noise_prob.
        Special cases:
            p = 0.0  →  L(t) = [t == z]  → posterior = Dirac at z
            p = 1.0  →  L(t) = 1/10 ∀t  → posterior = prior (no info)

        Update rule (unnormalised):
            posterior[t] ∝ L(t) · prior[t]
        """
        p = self.noise_prob
        uniform_lh = p / VALID_TYPES   # p/10

        prior = self.belief[wx, wy]          #numpy view into belief array

        #Start by applying the uniform component of the likelihood to all types
        posterior = prior * uniform_lh       # new array — does not modify prior

        #boost likelihood for the observed type z (if it's a valid type 1–10)
        if 1 <= z <= 10:
            posterior[z] = prior[z] * ((1.0 - p) + uniform_lh)

        #Type 0 (unseen) has zero likelihood so posterior stays 0 because
        #prior[0] == 0 after reset(), and we only update valid types.

        total = posterior.sum()
        if total > 1e-12:
            self.belief[wx, wy] = posterior / total
        # If total roughly 0 (degenerate belief), keep the prior to avoid NaN.
