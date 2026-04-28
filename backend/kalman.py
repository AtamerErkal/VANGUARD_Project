"""
VANGUARD AI — Kalman Filter & Dempster-Shafer Fusion
Track-level state estimation and evidence theory fusion.
"""

import numpy as np
from dataclasses import dataclass, field


# ── Kalman Filter ─────────────────────────────────────────────────────────────

@dataclass
class KalmanState:
    position: np.ndarray        # [lat, lon, alt_kft]  — smoothed estimate
    velocity: np.ndarray        # [dlat, dlon, dalt] per time step
    covariance: np.ndarray      # 6×6 posterior covariance
    innovations: list = field(default_factory=list)  # innovation sequence for NIS


class ConstantVelocityKalman:
    """
    6-state constant-velocity Kalman filter for 3-D aircraft track smoothing.

    State vector: x = [lat, lon, alt, d_lat, d_lon, d_alt]
    Measurement:  z = [lat, lon, alt]   (position only)

    Altitude is normalised to thousands of feet so all three axes are
    numerically comparable to latitude/longitude degrees.
    """

    def __init__(self, sigma_process: float = 0.03, sigma_meas: float = 0.06):
        dt = 1.0  # one time step (5 min in wall-clock; dimensionless here)

        # State transition — constant velocity
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Observation matrix — we measure position only
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise — velocity states are noisier (manoeuvre uncertainty)
        q = sigma_process ** 2
        self.Q = np.diag([q, q, q * 2, q * 8, q * 8, q * 12])

        # Measurement noise
        r = sigma_meas ** 2
        self.R = np.diag([r, r, r * 1.5])

    def filter(self, measurements: np.ndarray) -> KalmanState:
        """
        Run the filter over a sequence of measurements.

        Parameters
        ----------
        measurements : (N, 3) ndarray — [lat, lon, alt_kft]

        Returns
        -------
        KalmanState with final posterior estimate and covariance.
        """
        N = len(measurements)

        # Initialise state from first two measurements
        x = np.zeros(6)
        x[:3] = measurements[0]
        if N > 1:
            x[3:] = measurements[1] - measurements[0]

        P = np.eye(6) * 1.0
        innovations = []

        for z in measurements:
            # ── Predict ───────────────────────────────────────────────────────
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q

            # ── Update ────────────────────────────────────────────────────────
            innov = z - self.H @ x          # innovation (measurement residual)
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            x = x + K @ innov
            P = (np.eye(6) - K @ self.H) @ P
            innovations.append(innov.tolist())

        return KalmanState(
            position=x[:3].copy(),
            velocity=x[3:].copy(),
            covariance=P.copy(),
            innovations=innovations,
        )


def compute_track_quality(state: KalmanState) -> dict:
    """
    Derive a track quality assessment from the posterior covariance.

    The position-only trace (sum of position variances) gives a scalar
    measure of positional uncertainty. Lower trace → higher quality track.
    """
    pos_trace = float(np.trace(state.covariance[:3, :3]))

    # Normalise to [0, 1] uncertainty score (saturates at pos_trace ≥ 0.6)
    uncertainty_score = min(1.0, pos_trace / 0.6)

    if uncertainty_score < 0.20:
        quality_label = "HIGH"
    elif uncertainty_score < 0.55:
        quality_label = "MEDIUM"
    else:
        quality_label = "LOW"

    return {
        "label": quality_label,
        "uncertainty": round(uncertainty_score, 3),
        "covariance_trace": round(pos_trace, 5),
        "smoothed_position": {
            "lat": round(float(state.position[0]), 5),
            "lon": round(float(state.position[1]), 5),
            "alt_ft": round(float(state.position[2]) * 1000, 0),
        },
        "estimated_velocity": {
            "dlat_per_step": round(float(state.velocity[0]), 5),
            "dlon_per_step": round(float(state.velocity[1]), 5),
        },
    }


# ── Dempster-Shafer Evidence Theory ───────────────────────────────────────────

_FRAME = ["HOSTILE", "SUSPECT", "FRIEND", "ASSUMED FRIEND", "NEUTRAL", "UNKNOWN"]
_THETA = "Θ"   # open-world ignorance element


def _ds_combine(m1: dict, m2: dict) -> dict:
    """
    Dempster's orthogonal sum (rule of combination) for two mass functions.

    Focal elements are class singletons or Θ (total ignorance).
    Conflicting mass is renormalised away (closed-world assumption).
    """
    result: dict[str, float] = {}
    conflict = 0.0

    for A, mA in m1.items():
        for B, mB in m2.items():
            # Determine intersection of focal elements A ∩ B
            if A == _THETA:
                intersection = B
            elif B == _THETA:
                intersection = A
            elif A == B:
                intersection = A
            else:
                # Non-overlapping singletons → empty set → conflict
                conflict += mA * mB
                continue

            result[intersection] = result.get(intersection, 0.0) + mA * mB

    # Normalise by 1 − K  (K = total conflict)
    denom = 1.0 - conflict
    if denom < 1e-9:
        # Highly contradictory sensors — fall back to uniform
        return {c: 1.0 / len(_FRAME) for c in _FRAME}

    return {k: v / denom for k, v in result.items()}


def dempster_shafer_fusion(sensor_votes: dict) -> dict:
    """
    Fuse sensor evidence using Dempster-Shafer evidence theory.

    Each sensor contributes a basic probability assignment (BPA):
        m(voted_class) = confidence
        m(Θ)          = 1 − confidence   (ignorance / ambiguity)

    BPAs are combined sequentially with Dempster's rule.

    Returns best class, per-class beliefs, and conflict mass.
    """
    # Start from total ignorance
    combined: dict[str, float] = {_THETA: 1.0}

    conflict_accumulator = 0.0
    for sensor, vd in sensor_votes.items():
        vote = vd["vote"]
        conf = float(vd["conf"])
        sensor_bpa = {vote: conf, _THETA: 1.0 - conf}
        prev = combined.copy()
        combined = _ds_combine(combined, sensor_bpa)
        # Accumulate pairwise conflict for reporting
        conflict_accumulator = max(conflict_accumulator,
                                   sum(v for k, v in prev.items() if k != _THETA and k != vote) * conf)

    # Project onto the frame (discard leftover Θ mass into UNKNOWN)
    probs = {c: combined.get(c, 0.0) for c in _FRAME}
    probs["UNKNOWN"] = probs.get("UNKNOWN", 0.0) + combined.get(_THETA, 0.0)
    total = sum(probs.values()) or 1.0
    probs = {c: round(v / total, 4) for c, v in probs.items()}

    return {
        "best":          max(probs, key=probs.get),
        "probs":         probs,
        "conflict_mass": round(min(conflict_accumulator, 1.0), 4),
    }
