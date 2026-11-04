import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare import Drone, Model, GOAL, MODES, cbf, tensor


def load_models(folder):
    models = {}
    for name in MODES:
        path = folder / f"{name}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        model = Model()
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    return models


def run_episode(model, who, seed, horizon):
    env = Drone(seed, horizon)
    minimum = env.clearance()
    path_length = 0.0
    risk_sum = 0.0
    intervention = 0.0
    nsteps = 0
    status = "timeout"

    for _ in range(horizon):
        obs = env.teacher_obs() if who == "teacher" else env.student_obs()
        with torch.no_grad():
            nominal = model.action(who, tensor(obs)[None])[0].numpy()

        action = nominal
        if who == "teacher":
            action, feasible = cbf(env, nominal)
            if not feasible:
                status = "qp_failure"
                break
            intervention += float(np.sum((action - nominal) ** 2))

        before = env.p.copy()
        _, done, info = env.step(action)
        path_length += float(np.linalg.norm(env.p - before))
        minimum = min(minimum, info["clearance"])
        risk_sum += info["risk"]
        nsteps += 1

        if done:
            if info["success"]:
                status = "success"
            elif info["collision"]:
                status = "collision"
            elif info["boundary"]:
                status = "boundary"
            else:
                status = "timeout"
            break

    return {
        "seed": seed,
        "policy": who,
        "status": status,
        "success": int(status == "success"),
        "collision": int(status == "collision"),
        "qp_failure": int(status == "qp_failure"),
        "boundary": int(status == "boundary"),
        "timeout": int(status == "timeout"),
        "steps": nsteps,
        "goal_distance": float(np.linalg.norm(env.p - GOAL)),
        "min_clearance": minimum,
        "path_length": path_length,
        "mean_risk": risk_sum / max(nsteps, 1),
        "cbf_intervention": intervention / max(nsteps, 1),
    }


def wilson_interval(yes, n, z=1.959963984540054):
    p = yes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - half), float(center + half)


def paired_bootstrap(differences, rng, repeats=5000):
    differences = np.asarray(differences, dtype=np.float64)
    n = len(differences)
    estimates = np.empty(repeats)
    for k in range(repeats):
        idx = rng.integers(0, n, size=n)
        estimates[k] = differences[idx].mean()
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, n, rng):
    summary = []
    for who in ("teacher", "student"):
        for method in MODES:
            selected = [r for r in rows if r["method"] == method and r["policy"] == who]
            item = {"method": method, "policy": who, "episodes": n}
            for key in ("success", "collision", "qp_failure", "boundary", "timeout"):
                rate = np.mean([r[key] for r in selected])
                lo, hi = wilson_interval(sum(r[key] for r in selected), n)
                item[key + "_rate"] = float(rate)
                item[key + "_ci_low"] = lo
                item[key + "_ci_high"] = hi
            for key in ("steps", "goal_distance", "min_clearance", "path_length", "mean_risk", "cbf_intervention"):
                item[key + "_mean"] = float(np.mean([r[key] for r in selected]))
            summary.append(item)
    return summary


def paired_comparisons(rows, n, rng, repeats):
    by_key = {(r["method"], r["policy"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    comparisons = []
    for who in ("teacher", "student"):
        for baseline in ("cbf_only", "cbf_risk"):
            for measure in ("success", "collision"):
                a = np.array([by_key[("sitt_cbf_risk", who, s)][measure] for s in seeds])
                b = np.array([by_key[(baseline, who, s)][measure] for s in seeds])
                delta = a - b
                lo, hi = paired_bootstrap(delta, rng, repeats)
                comparisons.append({
                    "policy": who,
                    "metric": measure,
                    "method": "sitt_cbf_risk",
                    "baseline": baseline,
                    "difference": float(delta.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "sitt_only_success_count": int(np.sum((a == 1) & (b == 0))) if measure == "success" else "",
                    "baseline_only_success_count": int(np.sum((a == 0) & (b == 1))) if measure == "success" else "",
                })
    return comparisons


def plot_summary(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    keys = ("success", "collision", "qp_failure")
    xs = np.arange(len(MODES))
    width = 0.36
    for ax, key in zip(axes, keys):
        for j, who in enumerate(("teacher", "student")):
            ss = [next(s for s in summary if s["method"] == m and s["policy"] == who) for m in MODES]
            vals = np.array([s[key + "_rate"] for s in ss])
            lo = np.array([s[key + "_ci_low"] for s in ss])
            hi = np.array([s[key + "_ci_high"] for s in ss])
            x = xs + (j - 0.5) * width
            ax.bar(x, vals, width, label=who)
            ax.errorbar(x, vals, yerr=np.maximum(0.0, np.vstack((vals - lo, hi - vals))), fmt="none", capsize=3, color="black", lw=1)
        ax.set_xticks(xs, ["CBF", "CBF+Risk", "SITT+CBF+Risk"], rotation=15)
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of trained SITT-CBF checkpoints")
    parser.add_argument("--models", type=Path, default=Path("sitt_safe_results"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("monte_carlo_results"))
    args = parser.parse_args()
    if args.episodes < 2 or args.horizon < 1 or args.bootstrap < 1:
        parser.error("episodes >= 2, horizon >= 1, bootstrap >= 1 required")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    models = load_models(args.models)
    rows = []

    for i in range(args.episodes):
        seed = args.seed + i
        # Same environment seed for every model and both policies.
        for method, model in models.items():
            for who in ("teacher", "student"):
                result = run_episode(model, who, seed, args.horizon)
                rows.append({"method": method, **result})
        if (i + 1) % max(1, args.episodes // 10) == 0:
            print(f"Evaluated {i + 1}/{args.episodes} environments", flush=True)

    rng = np.random.default_rng(args.seed + 987654)
    summary = summarize(rows, args.episodes, rng)
    comparisons = paired_comparisons(rows, args.episodes, rng, args.bootstrap)
    save_csv(args.out / "episodes.csv", rows)
    save_csv(args.out / "summary.csv", summary)
    save_csv(args.out / "paired_differences.csv", comparisons)
    plot_summary(summary, args.out / "summary.png")

    print("\nMonte Carlo estimates (95% Wilson CI)")
    for s in summary:
        print(f"{s['method']:15s} {s['policy']:7s} "
              f"success={s['success_rate']:.1%} [{s['success_ci_low']:.1%}, {s['success_ci_high']:.1%}] "
              f"collision={s['collision_rate']:.1%} "
              f"qp_fail={s['qp_failure_rate']:.1%} "
              f"clearance={s['min_clearance_mean']:.3f}")
    print("\nPaired differences: SITT minus baseline (95% bootstrap CI)")
    for c in comparisons:
        print(f"{c['policy']:7s} {c['metric']:9s} vs {c['baseline']:9s} "
              f"diff={c['difference']:+.1%} [{c['ci_low']:+.1%}, {c['ci_high']:+.1%}]")
    print(f"\nSaved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
