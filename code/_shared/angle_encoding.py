"""Training-fitted angle encoders without changing the frozen experiments."""
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import RobustScaler

MODES = ("hard_clip", "linear_no_clip", "smooth_atan")


def matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite feature matrix")
    return values


def inject_risk(values, risk, weights, representation, strength):
    values = matrix(values)
    risk = np.asarray(risk, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if representation not in ("baseline", "convex", "feature_specific"):
        raise ValueError("Unknown risk representation")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Use the prescribed zero-to-one risk strength")
    if risk.shape != (len(values),) or not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("Risk must contain one probability per row")
    if weights.shape != (values.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Feature weights must be nonnegative and sum to one")
    strength = 0.0 if representation == "baseline" else strength
    weight = weights if representation == "feature_specific" else np.ones(values.shape[1])
    return values * (1 + strength * risk[:, None] * weight)


@dataclass(frozen=True)
class AngleEncoding:
    mode: str
    center: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    span: np.ndarray
    train_rows: int

    @classmethod
    def fit(cls, train, mode="smooth_atan"):
        train = matrix(train)
        if mode not in MODES:
            raise ValueError("Unknown angle encoding")
        minimum = train.min(0)
        span = np.maximum(train.max(0) - minimum, 1e-12)
        if mode == "smooth_atan":
            scaler = RobustScaler(unit_variance=True).fit(train)
            center, scale = scaler.center_.copy(), scaler.scale_.copy()
        else:
            center, scale = minimum.copy(), span.copy()
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid training-fitted scale")
        for item in (center, scale, minimum, span):
            item.setflags(write=False)
        return cls(mode, center, scale, minimum, span, len(train))

    def transform(self, values):
        values = matrix(values)
        if values.shape[1] != len(self.center):
            raise ValueError("Feature dimension changed")
        scaled = (values - self.center) / self.scale
        if self.mode == "smooth_atan":
            return np.pi / 2 + np.arctan(scaled)
        if self.mode == "linear_no_clip":
            return np.pi * scaled
        return np.pi * np.clip(scaled, 0, 1)

    def diagnostics(self, values, encoded=None):
        values = matrix(values)
        encoded = self.transform(values) if encoded is None else encoded
        unit = (values - self.minimum) / self.span
        outside = (unit < 0) | (unit > 1)
        if self.mode == "smooth_atan":
            standardized = (values - self.center) / self.scale
            # The ratio to the central derivative detects severe soft compression.
            gain_ratio = (1 / np.hypot(1, standardized)) ** 2
            compressed = gain_ratio < .01
        else:
            compressed = np.zeros_like(values, dtype=bool)
        hard = outside if self.mode == "hard_clip" else np.zeros_like(outside)
        boundary = ((encoded <= 0) | (encoded >= np.pi)) if self.mode == "smooth_atan" else np.zeros_like(outside)
        return {"hard_clipped_coordinate_fraction": float(hard.mean()), "hard_clipped_row_fraction": float(hard.any(1).mean()),
                "outside_legacy_domain_fraction": float(outside.mean()), "soft_compressed_coordinate_fraction": float(compressed.mean()),
                "soft_compressed_row_fraction": float(compressed.any(1).mean()), "numerical_boundary_fraction": float(boundary.mean()),
                "minimum_angle": float(encoded.min()), "maximum_angle": float(encoded.max())}

    def parameters(self):
        return {"mode": self.mode, "center": self.center.tolist(), "scale": self.scale.tolist(),
                "minimum": self.minimum.tolist(), "span": self.span.tolist(), "train_rows": self.train_rows,
                "fit_scope": "Unmodified training PCA features only; shared across all lambdas and representations"}
