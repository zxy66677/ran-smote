# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

__all__ = ["RANSMOTE", "noise_detection", "smote_with_noise", "ran_smote_oversampler"]

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

def _models(seed):
    return [
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        LinearSVC(dual="auto", random_state=seed, max_iter=10000),
        RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=1),
        GradientBoostingClassifier(random_state=seed),
    ]

def _fit_calibrated_model(X, y, tr, va, model_id, seed):
    """Fit one calibrated base learner for one outer fold."""
    y_tr = y[tr]
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    sample_weight = np.where(y_tr == 0, cw[0], cw[1])
    calibrated = CalibratedClassifierCV(
        clone(_models(seed)[model_id]), cv=3, method="isotonic", n_jobs=1
    )
    try:
        calibrated.fit(X[tr], y_tr, sample_weight=sample_weight)
    except TypeError:
        calibrated.fit(X[tr], y_tr)
    return calibrated.predict_proba(X[va])[:, 1]


def _fuse_probabilities(probabilities, eps):
    P = np.clip(np.vstack(probabilities), eps, 1 - eps)
    return _inv_logit(_logit(P, eps).mean(axis=0))


def _fit_outer_fold(X, y, fold_id, tr, va, seed, eps):
    probabilities = [
        _fit_calibrated_model(X, y, tr, va, model_id, seed)
        for model_id in range(len(_models(seed)))
    ]
    return fold_id, va, _fuse_probabilities(probabilities, eps)


def _fit_fold_model(X, y, fold_id, model_id, tr, va, seed):
    probability = _fit_calibrated_model(X, y, tr, va, model_id, seed)
    return fold_id, model_id, probability


def _parallel(tasks, n_jobs):
    """Run one non-nested process pool and cap every worker at one inner thread."""
    if n_jobs == 1:
        return Parallel(n_jobs=1, batch_size=1)(tasks)
    with parallel_config(
        backend="loky", n_jobs=n_jobs, inner_max_num_threads=1
    ):
        return Parallel(batch_size=1, pre_dispatch="n_jobs")(tasks)


def noise_detection(
    train_df, label_col, random_seed=42, min_repeats=5,
    ci_tolerance=0.05, stable_quantile=0.90, confidence_z=1.96,
    parallel_level="none", n_jobs=1,
):
    """Return error rates; parallel_level is none, fold, or fold_model."""
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame.")
    if label_col not in train_df:
        raise ValueError(f"label_col={label_col!r} is not in train_df.")
    if int(min_repeats) < 2:
        raise ValueError("min_repeats must be at least 2.")
    if not 0 < float(stable_quantile) <= 1:
        raise ValueError("stable_quantile must be in (0, 1].")
    if float(ci_tolerance) <= 0:
        raise ValueError("ci_tolerance must be positive.")
    parallel_level = str(parallel_level).lower()
    if parallel_level not in {"none", "fold", "fold_model"}:
        raise ValueError(
            "parallel_level must be 'none', 'fold', or 'fold_model'."
        )
    if not isinstance(n_jobs, (int, np.integer)) or int(n_jobs) < 1:
        raise ValueError("n_jobs must be a positive integer.")
    n_jobs = int(n_jobs)

    eps = 1e-6
    df = train_df.reset_index(drop=False).rename(columns={"index": "__orig_index__"})
    y_raw = df[label_col].to_numpy()
    X = df.drop(columns=[label_col, "__orig_index__"]).to_numpy()
    labels, counts = np.unique(y_raw, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"RAN-SMOTE supports binary classification only; got {len(labels)} classes.")

    minority_label, majority_label = labels[np.argmin(counts)], labels[np.argmax(counts)]
    y = (y_raw == minority_label).astype(int)
    min_mask, n_min = y == 1, int((y == 1).sum())
    K, max_repeats = min(max(2, math.ceil(n_min / 5)), 5), max(30, int(min_repeats))
    mean_l = np.zeros(len(df), dtype=float)
    m2_l = np.zeros(len(df), dtype=float)
    probabilities, repeats_done = [], 0
    seed, model_count = int(random_seed), len(_models(int(random_seed)))

    for repeat in range(1, max_repeats + 1):
        skf = StratifiedKFold(
            n_splits=min(K, max(2, n_min)), shuffle=True,
            random_state=int(random_seed + 101 * (repeat + 1)),
        )
        folds = list(skf.split(X, y))
        p_repeat = np.zeros(len(df), dtype=float)

        if parallel_level == "fold":
            tasks = [
                delayed(_fit_outer_fold)(
                    X, y, fold_id, tr, va, seed, eps
                )
                for fold_id, (tr, va) in enumerate(folds)
            ]
            for _, va, probability in sorted(
                _parallel(tasks, n_jobs), key=lambda item: item[0]
            ):
                p_repeat[va] = probability
        elif parallel_level == "fold_model":
            tasks = [
                delayed(_fit_fold_model)(
                    X, y, fold_id, model_id, tr, va, seed
                )
                for fold_id, (tr, va) in enumerate(folds)
                for model_id in range(model_count)
            ]
            result_map = {
                (fold_id, model_id): probability
                for fold_id, model_id, probability in _parallel(tasks, n_jobs)
            }
            for fold_id, (_, va) in enumerate(folds):
                p_repeat[va] = _fuse_probabilities(
                    [
                        result_map[(fold_id, model_id)]
                        for model_id in range(model_count)
                    ],
                    eps,
                )
        else:
            for fold_id, (tr, va) in enumerate(folds):
                _, _, probability = _fit_outer_fold(
                    X, y, fold_id, tr, va, seed, eps
                )
                p_repeat[va] = probability

        probabilities.append(p_repeat.copy())
        l_repeat = _logit(np.clip(p_repeat, eps, 1 - eps), eps)
        repeats_done += 1
        delta = l_repeat - mean_l
        mean_l += delta / repeats_done
        m2_l += delta * (l_repeat - mean_l)

        if repeats_done >= int(min_repeats):
            var_l = m2_l / max(1, repeats_done - 1)
            se_l = np.sqrt(var_l[min_mask] / repeats_done)
            p_min = _inv_logit(mean_l)[min_mask]
            se_p = se_l * p_min * (1.0 - p_min)
            ci_half = float(confidence_z) * se_p
            if np.quantile(ci_half, float(stable_quantile)) <= float(ci_tolerance):
                break

    probability_runs = np.vstack(probabilities)
    error_runs = np.zeros_like(probability_runs)
    for repeat in range(repeats_done):
        p = probability_runs[repeat]
        error_runs[repeat] = 1 - np.where(y == 1, p, 1 - p)
    df["error_rate"] = np.clip(error_runs.mean(axis=0), 1e-6, 1 - 1e-6)
    minority = df[df[label_col] == minority_label].drop(columns="__orig_index__").copy()
    majority = df[df[label_col] == majority_label].drop(columns="__orig_index__").copy()
    return minority, majority


def _beta_point(a, b, err_a, err_b, scale=5.0):
    safe_a, safe_b = max(1 - float(err_a), 1e-6), max(1 - float(err_b), 1e-6)
    weight_b = safe_b / (safe_a + safe_b)
    alpha = max(float(scale) * weight_b, 1e-3)
    beta = max(float(scale) * (1 - weight_b), 1e-3)
    t = np.random.beta(alpha, beta)
    return (1 - t) * a + t * b


def smote_with_noise(minority, majority, feature_cols, label_col="label", random_seed=42):
    np.random.seed(int(random_seed))
    feature_cols = list(feature_cols)
    if "error_rate" not in minority:
        raise ValueError("minority DataFrame must contain an 'error_rate' column.")

    n_min, n_maj = len(minority), len(majority)
    if n_min < 2 or n_maj <= n_min:
        result = pd.concat(
            [minority, majority, pd.DataFrame(columns=minority.columns)],
            ignore_index=True,
        )
        return result.dropna(subset=[label_col])

    n_new = n_maj - n_min
    X_min = minority[feature_cols].to_numpy(dtype=float)
    errors = np.clip(
        minority["error_rate"].astype(float).fillna(0.5).to_numpy(), 0, 1
    )
    noise_limit, k_min = 0.7, 3 if n_min > 3 else max(1, n_min - 1)
    k_max = max(k_min, int(np.sqrt(n_min)))
    query_k, mix = min(k_max + 1, n_min), 0.5

    weights = np.zeros(n_min, dtype=float)
    safe = errors <= noise_limit
    if not np.any(safe):
        safe, noise_limit = np.ones(n_min, dtype=bool), 1.0
    weights[safe] = errors[safe]
    if weights.sum() == 0:
        weights[safe] = 1.0
    expected = weights / weights.sum() * n_new
    counts = np.floor(expected).astype(int)
    remainder = int(n_new - counts.sum())
    if remainder > 0:
        counts[np.argsort(expected - counts)[-remainder:]] += 1

    nn = NearestNeighbors(n_neighbors=query_k, metric="euclidean").fit(X_min)
    distances, indices = nn.kneighbors(X_min)
    distances, indices = distances[:, 1:], indices[:, 1:]
    candidates = distances.shape[1]
    k_min, k_max = min(k_min, candidates), min(k_max, candidates)
    if k_max == k_min:
        adaptive_k = np.full(n_min, k_max, dtype=int)
    else:
        adaptive_k = np.floor(
            k_min + (1 - errors / noise_limit) * (k_max - k_min)
        ).astype(int)
        adaptive_k = np.clip(adaptive_k, 1, k_max)
    adaptive_k = np.clip(adaptive_k, 1, candidates)

    eps = 1e-12
    normalized = np.zeros_like(distances)
    for i in range(n_min):
        d = distances[i]
        denominator = d.max() - d.min()
        if denominator > eps:
            normalized[i] = (d - d.min()) / (denominator + eps)
    scores = mix * normalized + (1 - mix) * errors[indices]
    sorted_positions = np.argsort(scores, axis=1)

    new_samples, new_errors = [], []
    minority_label = minority[label_col].iloc[0]
    for i in range(n_min):
        if counts[i] <= 0 or errors[i] > noise_limit:
            continue
        positions = sorted_positions[i][:adaptive_k[i]]
        neighbors, selected_scores = indices[i, positions], scores[i, positions]
        inverse = 1 / (selected_scores + eps)
        probabilities = (
            np.full_like(inverse, 1 / len(inverse), dtype=float)
            if inverse.sum() == 0 else inverse / inverse.sum()
        )
        for _ in range(counts[i]):
            if len(neighbors) == 1:
                idx1 = idx2 = neighbors[0]
            else:
                choice = np.random.choice(
                    len(neighbors), size=2, replace=True, p=probabilities
                )
                idx1, idx2 = neighbors[choice[0]], neighbors[choice[1]]
            z1 = _beta_point(X_min[i], X_min[idx1], errors[i], errors[idx1], 5.0)
            z2 = _beta_point(X_min[i], X_min[idx2], errors[i], errors[idx2], 5.0)
            gamma = np.random.rand()
            new_samples.append((1 - gamma) * z1 + gamma * z2)
            new_errors.append(errors[i])

    if new_samples:
        new_df = pd.DataFrame(new_samples, columns=feature_cols)
        new_df[label_col], new_df["error_rate"] = minority_label, new_errors
    else:
        new_df = pd.DataFrame(columns=minority.columns)
    return pd.concat([minority, majority, new_df], ignore_index=True).dropna(
        subset=[label_col]
    )


def ran_smote_oversampler(
    train_df, feature_cols, label_col, random_seed=42,
    noise_random_seed=None, sampling_random_seed=None, return_error_rate=False,
    parallel_level="none", n_jobs=1,
):
    """General DataFrame oversampling interface."""
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a pandas DataFrame.")
    feature_cols = list(feature_cols)
    required = [*feature_cols, label_col]
    missing = [column for column in required if column not in train_df]
    if missing:
        raise ValueError(f"train_df is missing required columns: {missing}")
    noise_seed = int(random_seed if noise_random_seed is None else noise_random_seed)
    sample_seed = int(random_seed if sampling_random_seed is None else sampling_random_seed)
    minority, majority = noise_detection(
        train_df, label_col, noise_seed,
        parallel_level=parallel_level, n_jobs=n_jobs,
    )
    result = smote_with_noise(
        minority, majority, feature_cols, label_col, sample_seed
    ).reset_index(drop=True)
    return result if return_error_rate else result[required]


class RANSMOTE:

    def __init__(
        self, random_state=42, parallel_level="none", n_jobs=1,
    ):
        self.random_state = int(random_state)
        self.parallel_level, self.n_jobs = parallel_level, int(n_jobs)

    def fit_resample(self, X, y):
        X, y = np.asarray(X), np.asarray(y).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        columns, label = [f"feature_{i}" for i in range(X.shape[1])], "__label__"
        df = pd.DataFrame(X, columns=columns)
        df[label] = y
        result = ran_smote_oversampler(
            df, columns, label, random_seed=self.random_state,
            parallel_level=self.parallel_level, n_jobs=self.n_jobs,
        )
        return result[columns].to_numpy(), result[label].to_numpy()

    def sample(self, X, y):
        return self.fit_resample(X, y)
        
if __name__ == "__main__":
    from collections import Counter
    from time import perf_counter
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        class_sep=1.5,
        flip_y=0,
        random_state=107,
    )

    sampler = RANSMOTE(random_state=107)

    start = perf_counter()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
