#!/usr/bin/env python3
"""
stats_toolkit.py
=================
All the "Part 1/2/6" statistics from Comparison_Metrics_Notes.docx, kept in
one dependency-light module (numpy/scipy only; pyCompare/pingouin are used
if available but are NOT required, since installing them can be flaky).

Implemented:
  - coefficient_of_variation(values)            -> CV% = (sigma/mu)*100
  - bland_altman(x, y)                            -> bias, SD(diff), 95% LoA
  - icc_2_1(ratings)                              -> two-way random effects,
                                                      absolute agreement ICC
  - passing_bablok(x, y)                          -> slope, intercept, 95% CIs
  - pearson_and_spearman(x, y)                    -> r, R^2, rho, p-values
  - expected_calibration_error(confidences, correct, n_bins=10)
  - bootstrap_ci(values, statistic_fn, n_boot, confidence)
  - cohens_kappa(rater1, rater2, labels)          -> inter-annotator agreement
"""

import numpy as np


# ----------------------------------------------------------------------------
# CV% (Part 1)
# ----------------------------------------------------------------------------
def coefficient_of_variation(values):
    """CV% = (sigma / mu) * 100, per the notes' definition. Uses the sample
    standard deviation (ddof=1), the conventional choice for repeated
    measurements of the same specimen."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    mu = values.mean()
    if mu == 0:
        return np.nan
    sigma = values.std(ddof=1)
    return (sigma / mu) * 100.0


# ----------------------------------------------------------------------------
# Bland-Altman (Part 2)
# ----------------------------------------------------------------------------
def bland_altman(system_counts, manual_counts):
    """bias = mean(system - manual); LoA = bias +/- 1.96 * SD(diff)."""
    system_counts = np.asarray(system_counts, dtype=float)
    manual_counts = np.asarray(manual_counts, dtype=float)
    diffs = system_counts - manual_counts
    means = (system_counts + manual_counts) / 2.0

    bias = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff

    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "diffs": diffs,
        "means": means,
        "n": len(diffs),
    }


# ----------------------------------------------------------------------------
# ICC(2,1) - two-way random effects, absolute agreement, single rater
# (Part 2). Implemented from the standard Shrout & Fleiss ANOVA formulas so
# we don't require the `pingouin` package, but we use pingouin if present
# for a cross-check.
# ----------------------------------------------------------------------------
def icc_2_1(ratings_matrix):
    """
    ratings_matrix: array [n_subjects, n_raters] (here n_raters=2: system,
    manual). Returns dict with icc value and a rough 95% CI (F-distribution
    based, Shrout & Fleiss 1979 formulas).
    """
    try:
        import pingouin as pg
        import pandas as pd
        n, k = ratings_matrix.shape
        long_rows = []
        for subj in range(n):
            for rater in range(k):
                long_rows.append({"subject": subj, "rater": f"r{rater}", "score": ratings_matrix[subj, rater]})
        df = pd.DataFrame(long_rows)
        icc_table = pg.intraclass_corr(data=df, targets="subject", raters="rater", ratings="score")
        row = icc_table[icc_table["Type"] == "ICC2"].iloc[0]
        return {
            "icc": float(row["ICC"]),
            "ci95_lower": float(row["CI95%"][0]),
            "ci95_upper": float(row["CI95%"][1]),
            "p_value": float(row["pval"]),
            "method": "pingouin ICC2",
        }
    except Exception:
        return _icc_2_1_manual(ratings_matrix)


def _icc_2_1_manual(ratings_matrix):
    """Manual Shrout & Fleiss ICC(2,1) via one-way/two-way random-effects
    ANOVA mean squares. ratings_matrix: [n, k]."""
    X = np.asarray(ratings_matrix, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return {"icc": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan, "p_value": np.nan, "method": "manual (insufficient data)"}

    grand_mean = X.mean()
    row_means = X.mean(axis=1)  # per-subject
    col_means = X.mean(axis=0)  # per-rater

    ss_total = ((X - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()      # between subjects
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()      # between raters
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error))

    # F-distribution based 95% CI (Shrout & Fleiss / McGraw & Wong ICC(2,1))
    try:
        from scipy.stats import f as f_dist
        F_j = ms_rows / ms_error if ms_error > 0 else np.inf
        dfn = n - 1
        dfd = (n - 1) * (k - 1)
        Fl = F_j / f_dist.ppf(0.975, dfn, dfd)
        Fu = F_j * f_dist.ppf(0.975, dfd, dfn)
        ci_lower = (Fl - 1) / (Fl + (k - 1))
        ci_upper = (Fu - 1) / (Fu + (k - 1))
    except Exception:
        ci_lower, ci_upper = np.nan, np.nan

    return {
        "icc": float(icc),
        "ci95_lower": float(ci_lower) if ci_lower == ci_lower else np.nan,
        "ci95_upper": float(ci_upper) if ci_upper == ci_upper else np.nan,
        "p_value": np.nan,
        "method": "manual Shrout-Fleiss ICC(2,1)",
    }


# ----------------------------------------------------------------------------
# Passing-Bablok regression (Part 6, Step 3) - robust, non-parametric,
# error-in-both-variables method-comparison regression. Implemented from
# scratch (median of pairwise slopes) since pyCompare's API changes between
# versions; used automatically if pyCompare is unavailable.
# ----------------------------------------------------------------------------
def passing_bablok(x, y):
    """Returns slope, intercept and approximate 95% CIs via the classic
    Passing & Bablok (1983) pairwise-slope method.
    x, y: paired measurements from method A (x) and method B (y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return {"slope": np.nan, "intercept": np.nan,
                "slope_ci_lower": np.nan, "slope_ci_upper": np.nan,
                "intercept_ci_lower": np.nan, "intercept_ci_upper": np.nan, "n_pairs": 0}

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                s = np.inf if dy > 0 else -np.inf
            else:
                s = dy / dx
            if s != -1:  # PB convention: exclude slopes of exactly -1
                slopes.append(s)

    slopes = np.array(sorted(slopes))
    m = len(slopes)
    if m == 0:
        return {"slope": np.nan, "intercept": np.nan,
                "slope_ci_lower": np.nan, "slope_ci_upper": np.nan,
                "intercept_ci_lower": np.nan, "intercept_ci_upper": np.nan, "n_pairs": 0}

    K = int(np.sum(slopes < -1))
    median_idx = (m + 1) // 2 - 1 if m % 2 == 1 else m // 2  # 0-indexed
    shifted = np.roll(slopes, -K) if K else slopes  # shift so ordering matches PB convention
    # Standard implementation: index by rank with offset K, using 1-indexed rank formulas
    def _rank_slope(rank_1_indexed):
        idx = rank_1_indexed + K - 1
        idx = min(max(idx, 0), m - 1)
        return slopes[idx]

    b = _rank_slope((m + 1) // 2) if m % 2 == 1 else 0.5 * (_rank_slope(m // 2) + _rank_slope(m // 2 + 1))

    # Confidence interval on the slope (normal approximation, PB 1983)
    from scipy.stats import norm
    w = norm.ppf(0.975) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    c1 = int(round((m - w) / 2))
    c2 = m - c1 + 1
    c1 = max(1, c1)
    c2 = min(m, c2)
    lower = _rank_slope(c1)
    upper = _rank_slope(c2)

    intercept = np.median(y - b * x)
    intercept_lower = np.median(y - upper * x)
    intercept_upper = np.median(y - lower * x)

    return {
        "slope": float(b),
        "intercept": float(intercept),
        "slope_ci_lower": float(lower),
        "slope_ci_upper": float(upper),
        "intercept_ci_lower": float(min(intercept_lower, intercept_upper)),
        "intercept_ci_upper": float(max(intercept_lower, intercept_upper)),
        "n_pairs": int(n),
    }


# ----------------------------------------------------------------------------
# Correlation (Part 6, Step 2)
# ----------------------------------------------------------------------------
def pearson_and_spearman(x, y):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "r_squared": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    r, p = pearsonr(x, y)
    rho, sp = spearmanr(x, y)
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "r_squared": float(r ** 2),
        "spearman_rho": float(rho),
        "spearman_p": float(sp),
    }


# ----------------------------------------------------------------------------
# Expected Calibration Error (next-step item #3)
# ----------------------------------------------------------------------------
def expected_calibration_error(confidences, correct, n_bins=10):
    """
    confidences: array of predicted confidence scores in [0, 1]
    correct: array of 0/1, whether that prediction was actually correct
    Returns (ece, bin_table) where bin_table has per-bin accuracy/confidence/count
    for plotting a reliability diagram.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_rows = []
    n_total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        if count == 0:
            bin_rows.append({"bin_lower": lo, "bin_upper": hi, "count": 0,
                              "avg_confidence": np.nan, "accuracy": np.nan})
            continue
        avg_conf = confidences[mask].mean()
        acc = correct[mask].mean()
        ece += (count / n_total) * abs(acc - avg_conf)
        bin_rows.append({"bin_lower": lo, "bin_upper": hi, "count": int(count),
                          "avg_confidence": float(avg_conf), "accuracy": float(acc)})

    return float(ece), bin_rows


# ----------------------------------------------------------------------------
# Bootstrap confidence intervals (used for CV%, Bland-Altman bias/LoA CIs,
# per the notes: "Report accuracy/CV%/agreement numbers with confidence
# intervals ... bootstrap CI for CV% and Bland-Altman bias/LoA")
# ----------------------------------------------------------------------------
def bootstrap_ci(values, statistic_fn, n_boot=2000, confidence=0.95, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_boot": 0}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        stats[b] = statistic_fn(sample)
    alpha = (1 - confidence) / 2
    lower, upper = np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": float(statistic_fn(values)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot": n_boot,
    }


def bootstrap_ci_paired(x, y, statistic_fn, n_boot=2000, confidence=0.95, seed=42):
    """Same as bootstrap_ci but resamples paired (x, y) indices together
    (needed for Bland-Altman bias/LoA and correlation CIs, where x and y
    must stay paired per-subject)."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(seed)
    if n == 0:
        return {"point": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_boot": 0}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic_fn(x[idx], y[idx])
    alpha = (1 - confidence) / 2
    lower, upper = np.nanpercentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": float(statistic_fn(x, y)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot": n_boot,
    }


# ----------------------------------------------------------------------------
# Cohen's kappa (inter-annotator agreement, notes Part 3)
# ----------------------------------------------------------------------------
def cohens_kappa(rater1_labels, rater2_labels, labels=None):
    """Cohen's kappa for per-cell-type classification agreement between two
    annotators labelling the same subset of images."""
    r1 = np.asarray(rater1_labels)
    r2 = np.asarray(rater2_labels)
    if labels is None:
        labels = sorted(set(r1.tolist()) | set(r2.tolist()))
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    cm = np.zeros((n_labels, n_labels), dtype=float)
    for a, b in zip(r1, r2):
        cm[label_to_idx[a], label_to_idx[b]] += 1

    n = cm.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float((row_marg * col_marg).sum())
    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return float(kappa)
