"""Fit partition-local elastic-net Cox components and calibrate standardized late fusion."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from survival_metrics import harrell_concordance
from model_config import CoxConfig

NUMERIC = ("age", "lvef", "egfr")
CATEGORICAL = ("sex", "cardiac_surgery", "cied", "va", "hx_va", "pci", "respiratory_history", "diabetes")
SCAR = ("scar_burden_hard", "scar_burden_soft", "scar_extent_fraction_active")
PENALTIES = CoxConfig().penalties
L1_RATIOS = CoxConfig().l1_ratios
FUSION_WEIGHTS = CoxConfig().fusion_weights
HORIZONS = (1.0, 3.0, 5.0)


def validate_outcomes(time, event):
    time = np.asarray(time, dtype=float)
    raw_event = np.asarray(event)
    if time.ndim != 1 or raw_event.shape != time.shape or not len(time):
        raise ValueError("Outcomes must be nonempty matching one-dimensional arrays")
    if not np.isfinite(time).all() or np.any(time <= 0):
        raise ValueError("Follow-up time must be finite, positive, and expressed in years")
    if not np.isin(raw_event, [0, 1, False, True]).all():
        raise ValueError("Event must be binary")
    if not np.any(raw_event):
        raise ValueError("At least one observed death is required")
    return time, raw_event.astype(bool)


def clinical_frame(frame):
    result = frame.loc[:, NUMERIC + CATEGORICAL].copy()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if np.isinf(result[name].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite clinical value: {name}")
    for name in CATEGORICAL:
        result[name] = result[name].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    return result


def preprocessor(kind, config=CoxConfig()):
    if not isinstance(config, CoxConfig):
        raise TypeError("Cox preprocessing requires a CoxConfig")
    if kind not in {"clinical", "scar", "early"}:
        raise ValueError("Cox component kind must be clinical, scar or early")
    numeric = list(NUMERIC if kind == "clinical" else SCAR if kind == "scar" else NUMERIC + SCAR)
    transforms = [("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]), numeric)]
    if kind != "scar":
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(drop="first", min_frequency=config.categorical_min_frequency,
                                     handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]), list(CATEGORICAL)))
    return ColumnTransformer(transforms, remainder="drop", sparse_threshold=0)


def stratified_splits(event, folds, seed):
    raw_event = np.asarray(event)
    if raw_event.ndim != 1 or not np.isin(raw_event, (0, 1, False, True)).all():
        raise ValueError("Stratification requires a one-dimensional binary event vector")
    if not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError("Cross-validation requires at least two folds")
    event = raw_event.astype(bool)
    counts = np.bincount(event.astype(int), minlength=2)
    if min(counts) < folds:
        raise ValueError(f"Each event stratum needs at least {folds} patients")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(event)), event))


@dataclass
class CoxComponent:
    kind: str
    penalty: float
    l1_ratio: float
    transformer: object
    estimator: object
    mean: float
    scale: float

    @classmethod
    def fit(cls, frame, time, event, kind, penalty, l1_ratio, config=CoxConfig()):
        time, event = validate_outcomes(time, event)
        if len(frame) != len(time):
            raise ValueError("Feature and outcome rows must agree")
        if not np.isfinite(penalty) or penalty <= 0 or not np.isfinite(l1_ratio) or not 0 < l1_ratio <= 1:
            raise ValueError("Cox penalty must be positive and l1_ratio must be in (0, 1]")
        transform = preprocessor(kind, config)
        x = transform.fit_transform(frame)
        model = CoxnetSurvivalAnalysis(alphas=[penalty], l1_ratio=l1_ratio,
                                      normalize=False, max_iter=100000, tol=1e-7)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="all coefficients are zero")
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(x, Surv.from_arrays(event, time))
            except ConvergenceWarning as exc:
                raise ValueError("Cox candidate did not converge within 100000 iterations") from exc
        raw = model.predict(x)
        if not np.isfinite(raw).all():
            raise ValueError("Cox fitting produced nonfinite scores")
        scale = float(np.std(raw, ddof=0))
        return cls(kind, penalty, l1_ratio, transform, model, float(np.mean(raw)), scale if scale > 1e-12 else 0.0)

    def raw(self, frame):
        values = self.estimator.predict(self.transformer.transform(frame))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite Cox prediction")
        return values

    def standardized(self, frame):
        values = self.raw(frame)
        return (values - self.mean) / self.scale if self.scale else np.zeros(len(values))

    def coefficients(self):
        return pd.DataFrame({"feature": self.transformer.get_feature_names_out(),
                             "coefficient": self.estimator.coef_[:, 0]})


def tune_component(frame, time, event, kind, splits, penalties=None, l1_ratios=None, config=CoxConfig()):
    if not isinstance(config, CoxConfig):
        raise TypeError("Cox tuning requires a CoxConfig")
    configured = CoxConfig(penalties=config.penalties if penalties is None else penalties,
                           l1_ratios=config.l1_ratios if l1_ratios is None else l1_ratios,
                           categorical_min_frequency=config.categorical_min_frequency,
                           fusion_weights=config.fusion_weights)
    penalties, l1_ratios = configured.penalties, configured.l1_ratios
    time, event = validate_outcomes(time, event)
    if len(frame) != len(time):
        raise ValueError("Feature and outcome rows must agree")
    coverage = np.zeros(len(time), dtype=int)
    normalized_splits = []
    for train, valid in splits:
        train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
        if (train.ndim != 1 or valid.ndim != 1 or not len(train) or not len(valid)
                or np.any(train < 0) or np.any(train >= len(time))
                or np.any(valid < 0) or np.any(valid >= len(time))
                or len(np.unique(train)) != len(train)
                or len(np.unique(valid)) != len(valid)
                or np.intersect1d(train, valid).size
                or len(train) + len(valid) != len(time)):
            raise ValueError("Each tuning split must be a disjoint complete partition")
        coverage[valid] += 1
        normalized_splits.append((train, valid))
    if not np.all(coverage == 1):
        raise ValueError("Each tuning patient must receive exactly one held-out prediction")
    best = None
    history = []
    for penalty in penalties:
        for ratio in l1_ratios:
            score_sum, pair_sum = 0.0, 0
            z = np.full(len(frame), np.nan)
            failure = None
            for train, valid in normalized_splits:
                try:
                    component = CoxComponent.fit(frame.iloc[train], time[train], event[train], kind, penalty, ratio,
                                                 config=configured)
                    z[valid] = component.standardized(frame.iloc[valid])
                    concordance, pairs = harrell_concordance(time[valid], event[valid], z[valid])
                    if pairs and np.isfinite(concordance):
                        score_sum += concordance * pairs
                        pair_sum += pairs
                except (ValueError, ArithmeticError) as exc:
                    failure = str(exc)
                    break
            score = score_sum / pair_sum if pair_sum and failure is None and np.isfinite(z).all() else -np.inf
            history.append({"penalty": penalty, "l1_ratio": ratio, "concordance": score, "failure": failure})
            key = (score, penalty, -ratio)
            if np.isfinite(score) and (best is None or key > best[0]):
                best = (key, penalty, ratio, z)
    if best is None:
        raise RuntimeError(f"All {kind} Cox candidates failed: {history}")
    _, penalty, ratio, z = best
    fitted = CoxComponent.fit(frame, time, event, kind, penalty, ratio, config=configured)
    return fitted, z, history


def fit_scalar_cox(q, time, event):
    time, event = validate_outcomes(time, event)
    q = np.asarray(q, dtype=float)
    if q.shape != time.shape or not np.isfinite(q).all():
        raise ValueError("Invalid calibration score")
    if np.std(q) < 1e-12:
        return 0.0
    death_times = np.unique(time[event])
    sets = [(q[time >= value], q[(time == value) & event]) for value in death_times]

    def objective(parameter):
        gamma = parameter[0]
        loss, gradient = 0.0, 0.0
        for at_risk, deaths in sets:
            log_risk = gamma * at_risk
            normalizer = logsumexp(log_risk)
            weights = np.exp(log_risk - normalizer)
            loss += len(deaths) * normalizer - gamma * deaths.sum()
            gradient += len(deaths) * np.dot(weights, at_risk) - deaths.sum()
        return loss, np.array([gradient])

    result = minimize(objective, [1.0], method="BFGS", jac=True, options={"gtol": 1e-7, "maxiter": 1000})
    if not np.isfinite(result.x).all() or (not result.success and abs(result.jac[0]) > 1e-5):
        raise RuntimeError(f"Scalar Cox calibration failed: {result.message}")
    return float(result.x[0])


@dataclass
class CalibratedRisk:
    gamma: float
    death_times: np.ndarray
    log_cumulative_hazard: np.ndarray

    @classmethod
    def fit(cls, oof_q, final_q, time, event):
        time, event = validate_outcomes(time, event)
        final_q = np.asarray(final_q, dtype=float)
        if final_q.shape != time.shape or not np.isfinite(final_q).all():
            raise ValueError("Final calibration scores must be finite and match outcomes")
        gamma = fit_scalar_cox(oof_q, time, event)
        eta = gamma * final_q
        if not np.isfinite(eta).all():
            raise ValueError("Calibrated log hazards must be finite")
        times = np.unique(time[event])
        increments = np.array([np.log(np.sum(event & (time == value))) - logsumexp(eta[time >= value]) for value in times])
        cumulative = np.logaddexp.accumulate(increments)
        return cls(gamma, times, cumulative)

    def survival(self, q, horizons=HORIZONS):
        horizons = np.asarray(horizons, dtype=float)
        if (horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all()
                or np.any(horizons < 0) or np.any(np.diff(horizons) <= 0)):
            raise ValueError("Prediction horizons must be finite, nonnegative and strictly increasing")
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or not np.isfinite(q).all():
            raise ValueError("Prediction scores must be a finite one-dimensional vector")
        positions = np.searchsorted(self.death_times, horizons, side="right") - 1
        log_h = np.full(len(horizons), -np.inf)
        valid = positions >= 0
        log_h[valid] = self.log_cumulative_hazard[positions[valid]]
        with np.errstate(over="ignore", under="ignore"):
            eta = self.gamma * q
            output = np.ones((len(q), len(horizons)))
            log_total = eta[:, None] + log_h[None, valid]
            output[:, valid] = np.exp(-np.exp(log_total))
        return output
