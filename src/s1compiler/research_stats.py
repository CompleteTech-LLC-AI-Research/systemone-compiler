"""Task metrics and paired cluster inference; NumPy is an optional research dependency.

Confidence intervals condition on the predeclared optimization seeds. They do
not treat repeated predictions for the same example as independent observations.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .errors import DataError


def np_module():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Research reporting requires pip install -e '.[research]'.") from exc
    return np


def wilson(errors, count):
    if not count:
        return None
    z = 1.959963984540054
    p = errors / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-half), min(1., center+half)]


def binary_ranking(y, p):
    """Tie-aware ROC AUC and noninterpolated average precision."""
    np = np_module()
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None}
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p)-1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, np.cumsum(~y)[ends]]
    recall, fpr = tp / positives, fp / negatives
    auc = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
    ap = float(np.sum(np.diff(recall) * tp[1:] / (tp[1:] + fp[1:])))
    return {"auroc": auc, "average_precision": ap}


def calibration_bins(confidence, correct, bins=10):
    output = []
    for i in range(bins):
        selected = [(float(p), bool(y)) for p, y in zip(confidence, correct) if min(int(p*bins), bins-1) == i]
        count = len(selected)
        output.append({"lower": i/bins, "upper": (i+1)/bins, "n": count,
                       "mean_probability": sum(p for p, _ in selected)/count if count else None,
                       "empirical_frequency": sum(y for _, y in selected)/count if count else None})
    n = len(confidence)
    return {"ece": sum(b["n"]/n * abs(b["mean_probability"]-b["empirical_frequency"])
                       for b in output if b["n"]), "bins": output}


def observations(task, evidence, source):
    """Fixed-threshold task arrays, independent of learned review gates."""
    np = np_module()
    names = list(source.decisions)
    if task == "goemotions":
        y = np.array([[r["gold"][k] for k in names] for r in evidence], dtype=bool)
        p = np.array([[r["decisions"][k]["p_true"] for k in names] for r in evidence], dtype=float)
        pred = p >= .5
        # Sufficient statistics: TP/FP/FN for every label, one independent row per comment.
        stats = np.concatenate([y & pred, ~y & pred, y & ~pred], axis=1).astype(float)
        return y, pred, p, stats, names
    name = names[0]
    if task == "boolq":
        y = np.array([r["gold"][name] for r in evidence], dtype=bool)
        p = np.array([r["decisions"][name]["p_true"] for r in evidence], dtype=float)
        pred = p >= .5
        stats = np.column_stack([pred == y, np.ones(len(y))])
        return y, pred, p, stats, ["false", "true"]
    if task == "sst5":
        labels = list(range(5))
        y = np.array([r["gold"][name] for r in evidence], dtype=int)
        p = np.array([[r["decisions"][name]["probabilities"][str(i)] for i in labels] for r in evidence])
        pred = p.argmax(axis=1)
        expected = np.array([r["decisions"][name]["value"] for r in evidence], dtype=float)
        stats = np.column_stack([abs(expected-y), np.ones(len(y))])
        return y, pred, p, stats, labels
    labels = list(source.decisions[name].criteria)
    y = np.array([labels.index(r["gold"][name]) for r in evidence], dtype=int)
    p = np.array([[r["decisions"][name]["probabilities"][k] for k in labels] for r in evidence])
    # Preserve the actual returned Choice in ties rather than inventing another argmax.
    pred = np.array([labels.index(r["decisions"][name]["value"]) for r in evidence], dtype=int)
    if task == "clinc150":
        gold = y[:, None] == np.arange(len(labels))
        selected = pred[:, None] == np.arange(len(labels))
        stats = np.concatenate([gold & selected, ~gold & selected, gold & ~selected], axis=1).astype(float)
    else:
        stats = np.column_stack([y == pred, np.ones(len(y))])
    return y, pred, p, stats, labels


def primary_from_stats(task, stats):
    np = np_module()
    if task in {"goemotions", "clinc150"}:
        tp, fp, fn = np.split(stats, 3, axis=-1)
        denominator = 2*tp + fp + fn
        return np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator != 0).mean(axis=-1)
    return np.divide(stats[..., 0], stats[..., 1], out=np.zeros_like(stats[..., 0]), where=stats[..., 1] != 0)


def task_metrics(task, evidence, source):
    np = np_module()
    if not evidence:
        raise DataError("Cannot report an empty evaluation.")
    y, pred, p, stats, labels = observations(task, evidence, source)
    result = {"n": len(evidence), "n_clusters": len({r["group"] or r["id"] for r in evidence}),
              "primary": float(primary_from_stats(task, stats.sum(axis=0))),
              "zero_division_rule": "Precision/recall/F1 = 0 for zero denominator; absent labels remain in macro average."}
    if task == "goemotions":
        tp, fp, fn = np.split(stats.sum(axis=0), 3)
        correct = (pred == y).all(axis=1)
        result.update(exact_match_accuracy=float(correct.mean()),
            micro_f1=float(2*tp.sum()/max(1, (2*tp+fp+fn).sum())), macro_f1=result["primary"],
            hamming_loss=float((pred != y).mean()), brier=float(((p-y)**2).mean()),
            binary_log_loss=float(-(y*np.log(np.clip(p, 1e-12, 1)) + (~y)*np.log(np.clip(1-p, 1e-12, 1))).mean()),
            per_label={str(label): {"support": int(y[:, i].sum()), "tp": int(tp[i]), "fp": int(fp[i]),
                "fn": int(fn[i]), "precision": float(tp[i]/max(1, tp[i]+fp[i])),
                "recall": float(tp[i]/max(1, tp[i]+fn[i])), "f1": float(2*tp[i]/max(1, 2*tp[i]+fp[i]+fn[i])),
                "brier": float(((p[:, i]-y[:, i])**2).mean()),
                "calibration": calibration_bins(p[:, i].tolist(), y[:, i].tolist())}
                for i, label in enumerate(labels)},
            threshold=.5, joint_probability=None)
    else:
        yi, pi = y.astype(int), pred.astype(int)
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        np.add.at(cm, (yi, pi), 1)
        tp = cm.diagonal()
        f1_den = cm.sum(axis=0)+cm.sum(axis=1)
        f1 = np.divide(2*tp, f1_den, out=np.zeros(len(labels)), where=f1_den != 0)
        correct = y == pred
        if task == "boolq":
            prob_gold = np.where(y, p, 1-p)
            confidence = np.where(pred, p, 1-p)
            brier = float(((p-y)**2).mean())
        else:
            prob_gold = p[np.arange(len(y)), y]
            confidence = p[np.arange(len(y)), pred]
            brier = float(((p - (y[:, None] == np.arange(len(labels))))**2).sum(axis=1).mean())
        result.update(accuracy=float(correct.mean()), macro_f1=float(f1.mean()), brier=brier,
                      log_loss=float(-np.log(np.clip(prob_gold, 1e-12, 1)).mean()), labels=labels,
                      confusion=cm.tolist(), support=cm.sum(axis=1).tolist(),
                      calibration=calibration_bins(confidence.tolist(), correct.tolist()))
        if task == "sst5":
            errors = stats[:, 0]
            weights = (np.arange(5)[:, None]-np.arange(5))**2 / 16
            expected_cm = np.outer(cm.sum(axis=1), cm.sum(axis=0))/len(y)
            denominator = (weights*expected_cm).sum()
            result.update(mae=float(errors.mean()), rmse=float(np.sqrt((errors**2).mean())),
                          quadratic_weighted_kappa=float(1-(weights*cm).sum()/denominator) if denominator else None,
                          score_units="Original 0..4 ordinal levels; equal-distance assumption for MAE.",
                          classification_rule="argmax level probability, lowest index on ties; not rounded expected score")
        elif task == "clinc150":
            index = labels.index("oos")
            oos_gold, oos_pred = y == index, pred == index
            positives = int((oos_gold & oos_pred).sum())
            result["out_of_scope"] = {"n": int(oos_gold.sum()),
                "precision": positives/int(oos_pred.sum()) if oos_pred.any() else 0.,
                "recall": positives/int(oos_gold.sum()) if oos_gold.any() else 0.,
                "false_accept_rate": float((~oos_pred)[oos_gold].mean()) if oos_gold.any() else None,
                **binary_ranking(oos_gold, p[:, index])}
            result["in_scope_accuracy"] = float(correct[~oos_gold].mean()) if (~oos_gold).any() else None
        elif task == "boolq":
            result.update(threshold=.5, **binary_ranking(y, p))
    # Report review decisions separately from fixed-threshold benchmark metrics.
    review = {}
    for key, declaration in source.decisions.items():
        accepted = [r for r in evidence if not r["decisions"][key]["review_required"]]
        errors = sum((abs(r["decisions"][key]["value"]-r["gold"][key]) > declaration.score_tolerance)
                     if declaration.type == "score" else r["decisions"][key]["value"] != r["gold"][key]
                     for r in accepted)
        review[key] = {"accepted_n": len(accepted), "coverage": len(accepted)/len(evidence),
                       "errors": errors, "error_rate": errors/len(accepted) if accepted else None,
                       "wilson95_descriptive": wilson(errors, len(accepted)),
                       "basis": "Frozen calibration-fitted policy; not a risk guarantee."}
    latencies = [r["latency_ms"] for r in evidence if not r["cache_hit"]]
    result.update(review=review, latency_ms={"mean": float(np.mean(latencies)) if latencies else None,
                  "p50": float(np.quantile(latencies, .5)) if latencies else None,
                  "p95": float(np.quantile(latencies, .95)) if latencies else None})
    return result


def clustered_pair(task, left_runs, right_runs, source, *, bootstrap=5000, permutations=10000,
                   seed=90210, family_size=10):
    """Paired cluster bootstrap and paired group-swap randomization over mean seed effects."""
    np = np_module()
    if len(left_runs) != len(right_runs) or not left_runs or bootstrap < 100 or permutations < 100 or family_size < 1:
        raise ValueError("Invalid paired experiment or resampling parameters.")
    reference = left_runs[0]
    keys = [(r["id"], r["group"], r["gold"]) for r in reference]
    if len({r["id"] for r in reference}) != len(reference):
        raise DataError("Repeated evaluation IDs in paired inference.")
    for run in [*left_runs, *right_runs]:
        if [(r["id"], r["group"], r["gold"]) for r in run] != keys:
            raise DataError("Paired predictions must have identical ordered IDs, groups and gold labels.")
    group_names = sorted({r["group"] or r["id"] for r in reference})
    if len(group_names) < 2:
        raise DataError("Cluster inference requires at least two independent groups.")
    indexes = {name: i for i, name in enumerate(group_names)}
    membership = np.array([indexes[r["group"] or r["id"]] for r in reference])
    grouped = []
    for run in [*left_runs, *right_runs]:
        stats = observations(task, run, source)[3]
        sums = np.zeros((len(group_names), stats.shape[1]))
        np.add.at(sums, membership, stats)
        grouped.append(sums)
    n_seeds = len(left_runs)
    left, right = grouped[:n_seeds], grouped[n_seeds:]
    raw_deltas = np.array([primary_from_stats(task, a.sum(axis=0))-primary_from_stats(task, b.sum(axis=0))
                           for a, b in zip(left, right)])
    direction = -1 if task == "sst5" else 1
    observed = float(raw_deltas.mean())
    rng = np.random.default_rng(seed)
    boot, null = [], []
    size = len(group_names)
    for start in range(0, bootstrap, 128):
        count = min(128, bootstrap-start)
        weights = rng.multinomial(size, np.full(size, 1/size), size=count)
        effects = [primary_from_stats(task, weights@a)-primary_from_stats(task, weights@b) for a, b in zip(left, right)]
        boot.extend(np.mean(effects, axis=0).tolist())
    for start in range(0, permutations, 128):
        count = min(128, permutations-start)
        swap = rng.integers(0, 2, size=(count, size))
        effects = []
        for a, b in zip(left, right):
            delta = swap@(b-a)
            effects.append(primary_from_stats(task, a.sum(axis=0)+delta)-primary_from_stats(task, b.sum(axis=0)-delta))
        null.extend(np.mean(effects, axis=0).tolist())
    p_value = (1+sum(abs(x) >= abs(observed)-1e-12 for x in null))/(permutations+1)
    tail = .025/family_size
    signed_boot = direction*np.asarray(boot)
    return {"raw_delta_left_minus_right": observed, "improvement": direction*observed,
        "improvement_ci95": np.quantile(signed_boot, [.025, .975]).tolist(),
        "improvement_bonferroni_ci": np.quantile(signed_boot, [tail, 1-tail]).tolist(),
        "family_size": family_size, "two_sided_cluster_randomization_p": p_value,
        "per_seed_raw_deltas": raw_deltas.tolist(),
        "seed_delta_sd": float(raw_deltas.std(ddof=1)) if n_seeds > 1 else None,
        "n_examples": len(reference), "n_clusters": size, "optimization_seeds": n_seeds,
        "bootstrap_replicates": bootstrap, "randomization_replicates": permutations, "analysis_seed": seed,
        "minimum_monte_carlo_p": 1/(permutations+1),
        "inference_scope": "Example-weighted task metric; resample whole groups paired across arms and fixed seeds. "
                           "Conditional on these seeds, not inference over future optimization runs.",
        "bootstrap_degenerate": bool(max(boot) == min(boot))}


def holm(p_values):
    """Family-wise adjusted p values; missing planned tests must be supplied as p=1."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted, previous = {}, 0.
    for i, key in enumerate(ordered):
        value = p_values[key]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid p value.")
        previous = max(previous, min(1., (len(ordered)-i)*value))
        adjusted[key] = previous
    return adjusted


def strata_metrics(task, evidence, source):
    buckets = defaultdict(list)
    for row in evidence:
        buckets[row["stratum"]].append(row)
    return {name: task_metrics(task, rows, source) for name, rows in sorted(buckets.items())} if len(buckets) > 1 else {}
