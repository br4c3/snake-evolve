
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import torch

from compare import (
    Drone,
    Model,
    GOAL,
    MARGIN,
    MODES,
    cbf,
    tensor
)


# ============================================================
# Configuration
# ============================================================

METHOD_COLORS = {
    "cbf_only": "tab:blue",
    "cbf_risk": "tab:orange",
    "sitt_cbf_risk": "tab:green",
}

GROUP_LABELS = {
    "in": "Train-range start",
    "ood": "OOD start"
}

METRICS = (
    "success",
    "collision",
    "qp_failure",
    "goal_distance",
    "min_clearance",
    "path_length",
    "steps"
)


# ============================================================
# Load Trained Models
# ============================================================

def load_models(folder):
    models = {}

    for name in MODES:
        filename = folder / f"{name}.pt"

        if not filename.is_file():
            raise FileNotFoundError(filename)

        network = Model()

        state = torch.load(
            filename,
            map_location="cpu",
            weights_only=True
        )

        network.load_state_dict(state)
        network.eval()

        models[name] = network

    return models


# ============================================================
# Starting Point Distribution
# ============================================================

def create_starts(
    in_starts,
    ood_side,
    ood_near,
    max_y
):
    # Training start range:
    # x = 0, y ~ Uniform(-0.3, 0.3)

    starts = [
        ("in", float(y))
        for y in np.linspace(
            -0.3,
            0.3,
            in_starts
        )
    ]

    # Out-of-distribution starts.
    # Avoid the training interval.

    if ood_side:
        negative = np.linspace(
            -max_y,
            -ood_near,
            ood_side
        )

        positive = np.linspace(
            ood_near,
            max_y,
            ood_side
        )

        starts.extend(
            ("ood", float(y))
            for y in np.r_[negative, positive]
        )

    return [
        (i, group, y)
        for i, (group, y) in enumerate(starts)
    ]


# ============================================================
# Single Trajectory Simulation
# ============================================================

def simulate(
    model,
    who,
    seed,
    start_y,
    horizon
):
    env = Drone(seed, horizon)

    # Keep obstacle positions identical.
    # Change initial position only.

    env.p = np.array(
        [0.0, start_y],
        dtype=float
    )

    env.v = np.zeros(2)
    env.t = 0

    if env.clearance() <= MARGIN:
        return None

    points = [env.p.copy()]

    clearance = env.clearance()
    correction = 0.0
    status = "timeout"

    for _ in range(horizon):
        obs = (
            env.teacher_obs()
            if who == "teacher"
            else env.student_obs()
        )

        with torch.no_grad():
            nominal = model.action(
                who,
                tensor(obs)[None]
            )[0].cpu().numpy()

        action = nominal

        # Teacher uses CBF.
        # Student executes without CBF.

        if who == "teacher":
            action, feasible = cbf(
                env,
                nominal
            )

            if not feasible:
                status = "qp_failure"
                break

            correction += float(
                np.sum(
                    (action - nominal)**2
                )
            )

        _, done, info = env.step(action)

        points.append(env.p.copy())

        clearance = min(
            clearance,
            float(info["clearance"])
        )

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

    path = np.asarray(points)

    record = {
        "status": status,
        "success": int(
            status == "success"
        ),
        "collision": int(
            status == "collision"
        ),
        "qp_failure": int(
            status == "qp_failure"
        ),
        "boundary": int(
            status == "boundary"
        ),
        "timeout": int(
            status == "timeout"
        ),
        "steps": len(path) - 1,
        "goal_distance": float(
            np.linalg.norm(env.p - GOAL)
        ),
        "min_clearance": float(clearance),
        "path_length": float(
            np.linalg.norm(
                np.diff(path, axis=0),
                axis=1
            ).sum()
        ),
        "cbf_correction": correction
    }

    return record, path


# ============================================================
# Overlay Trajectories
# ============================================================

def plot_overlay(
    seed,
    curves,
    centers,
    radii,
    output
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.5),
        sharex=True,
        sharey=True
    )

    for ax, who in zip(
        axes,
        ("teacher", "student")
    ):
        # Draw obstacles.

        for c, radius in zip(
            centers,
            radii
        ):
            ax.add_patch(
                plt.Circle(
                    c,
                    radius + MARGIN,
                    color="tomato",
                    alpha=0.14
                )
            )

            ax.add_patch(
                plt.Circle(
                    c,
                    radius,
                    fill=False,
                    edgecolor="tomato",
                    lw=1.2
                )
            )

        # Draw trajectories of all models
        # on the SAME axes.

        for method in MODES:
            for record, path in curves[
                (method, who)
            ]:
                is_ood = (
                    record["group"] == "ood"
                )

                ax.plot(
                    path[:, 0],
                    path[:, 1],

                    color=METHOD_COLORS[method],

                    linestyle=(
                        "--" if is_ood else "-"
                    ),

                    alpha=(
                        0.34
                        if record["success"]
                        else 0.19
                    ),

                    lw=(
                        1.6
                        if record["success"]
                        else 1.1
                    )
                )

                # Starting point

                ax.scatter(
                    path[0, 0],
                    path[0, 1],
                    s=9,
                    color=METHOD_COLORS[method],
                    alpha=0.7
                )

        # Goal

        ax.scatter(
            *GOAL,
            color="black",
            marker="*",
            s=120,
            zorder=9
        )

        ax.set(
            title=f"{who.capitalize()} — all models and starts",
            xlim=(-0.7, 10.8),
            ylim=(-4, 4),
            xlabel="x (m)",
            ylabel="y (m)"
        )

        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # Shared legend

    legend = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[m],
            lw=2,
            label=m
        )
        for m in MODES
    ]

    legend.extend([
        Line2D(
            [0],
            [0],
            color="gray",
            lw=2,
            linestyle="-",
            label="Train-range start"
        ),

        Line2D(
            [0],
            [0],
            color="gray",
            lw=2,
            linestyle="--",
            label="OOD start"
        )
    ])

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        f"Shared obstacle layout {seed} "
        "| overlaid trajectories"
    )

    fig.tight_layout(
        rect=(0, 0.045, 1, 0.95)
    )

    fig.savefig(
        output,
        dpi=165,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CSV Utilities
# ============================================================

def save_csv(file, rows):
    if not rows:
        return

    with file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Per-Layout Metric Aggregation
# ============================================================

def layout_metric_arrays(
    rows,
    group,
    who,
    metric,
    methods,
    layout_seeds
):
    # First compute a within-layout mean.
    #
    # Multiple starting points in the same
    # obstacle layout are correlated.
    #
    # Each layout contributes one cluster
    # observation per model.

    mapping = {
        (
            r["layout_seed"],
            r["method"],
            r["policy"],
            r["group"],
            r["start_id"]
        ): r
        for r in rows
    }

    by_layout = []

    for seed in layout_seeds:
        start_ids = sorted({
            r["start_id"]
            for r in rows
            if (
                r["layout_seed"] == seed
                and r["group"] == group
            )
        })

        if not start_ids:
            continue

        means = []

        for method in methods:
            values = [
                mapping[
                    (
                        seed,
                        method,
                        who,
                        group,
                        k
                    )
                ][metric]
                for k in start_ids
            ]

            means.append(
                float(np.mean(values))
            )

        by_layout.append(means)

    return np.asarray(
        by_layout,
        dtype=float
    ).reshape(-1, len(methods))


# ============================================================
# Cluster Bootstrap
# ============================================================

def cluster_bootstrap(
    values,
    generator,
    n_boot
):
    """
    values:
        [n_layouts, n_methods]

    Resample obstacle layouts, not individual
    trajectories.

    All starting points within a selected
    layout remain together.
    """

    n = len(values)

    if n == 0:
        return None

    samples = np.empty(
        (n_boot, values.shape[1]),
        dtype=float
    )

    for b in range(n_boot):
        idx = generator.integers(
            0,
            n,
            n
        )

        samples[b] = (
            values[idx].mean(axis=0)
        )

    return samples


# ============================================================
# Cluster Bootstrap Analysis
# ============================================================

def analyze(
    rows,
    seeds,
    repeats,
    random_seed
):
    rng = np.random.default_rng(
        random_seed
    )

    summaries = []
    comparisons = []

    groups = (
        "in",
        "ood",
        "all"
    )

    for group in groups:
        for who in (
            "teacher",
            "student"
        ):
            for metric in METRICS:

                if group == "all":
                    # Merge ID and OOD into a single
                    # combined group.
                    sub = [
                        dict(r, group="all")
                        for r in rows
                    ]
                else:
                    sub = rows

                values = layout_metric_arrays(
                    sub,
                    group,
                    who,
                    metric,
                    MODES,
                    seeds
                )

                if not len(values):
                    continue

                bootstrap = cluster_bootstrap(
                    values,
                    rng,
                    repeats
                )

                mean = values.mean(axis=0)

                lower, upper = np.percentile(
                    bootstrap,
                    [2.5, 97.5],
                    axis=0
                )

                # -------------------------------------------
                # Model-level confidence intervals
                # -------------------------------------------

                for i, method in enumerate(MODES):
                    summaries.append({
                        "group": group,
                        "policy": who,
                        "metric": metric,
                        "method": method,
                        "layouts": len(values),
                        "mean": mean[i],
                        "ci_low": lower[i],
                        "ci_high": upper[i]
                    })

                # -------------------------------------------
                # Paired model differences
                # -------------------------------------------

                sitt = MODES.index(
                    "sitt_cbf_risk"
                )

                for baseline in (
                    "cbf_only",
                    "cbf_risk"
                ):
                    base = MODES.index(
                        baseline
                    )

                    # Difference within each layout.
                    diff = (
                        values[:, sitt]
                        - values[:, base]
                    )

                    bs = cluster_bootstrap(
                        diff[:, None],
                        rng,
                        repeats
                    )[:, 0]

                    lo, hi = np.percentile(
                        bs,
                        [2.5, 97.5]
                    )

                    comparisons.append({
                        "group": group,
                        "policy": who,
                        "metric": metric,
                        "model": "sitt_cbf_risk",
                        "baseline": baseline,
                        "layouts": len(values),
                        "difference": float(
                            diff.mean()
                        ),
                        "ci_low": float(lo),
                        "ci_high": float(hi)
                    })

    return summaries, comparisons


# ============================================================
# Success Rate Plot
# ============================================================

def plot_statistics(
    summary,
    output
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.7),
        sharey=True
    )

    for ax, group in zip(
        axes,
        ("in", "ood")
    ):
        for j, who in enumerate(
            ("teacher", "student")
        ):
            for i, method in enumerate(MODES):
                row = next(
                    (
                        r for r in summary
                        if (
                            r["group"] == group
                            and r["policy"] == who
                            and r["metric"] == "success"
                            and r["method"] == method
                        )
                    ),
                    None
                )

                if row is None:
                    continue

                x = i + (
                    -0.17
                    if who == "teacher"
                    else 0.17
                )

                ax.bar(
                    x,
                    row["mean"],
                    width=0.31,
                    color=METHOD_COLORS[method],
                    alpha=(
                        0.50
                        if who == "teacher"
                        else 1.0
                    )
                )

                ax.errorbar(
                    x,
                    row["mean"],
                    yerr=[
                        [
                            max(
                                0,
                                row["mean"] - row["ci_low"]
                            )
                        ],
                        [
                            max(
                                0,
                                row["ci_high"] - row["mean"]
                            )
                        ]
                    ],
                    fmt="none",
                    color="black",
                    capsize=3
                )

        ax.set_xticks(
            range(len(MODES)),
            [
                "CBF",
                "CBF+Risk",
                "SITT"
            ]
        )

        ax.set_title(
            GROUP_LABELS[group]
        )

        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel(
        "Success probability"
    )

    fig.suptitle(
        "Monte Carlo success by start distribution "
        "(95% layout-cluster bootstrap CI)"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=165
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("sitt_safe_results")
    )

    parser.add_argument(
        "--layouts",
        type=int,
        default=100
    )

    parser.add_argument(
        "--in-starts",
        type=int,
        default=7
    )

    parser.add_argument(
        "--ood-starts-per-side",
        type=int,
        default=7
    )

    parser.add_argument(
        "--ood-near",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--max-y",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=180
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--plot-layouts",
        type=int,
        default=5
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mc_cluster_overlay")
    )

    args = parser.parse_args()

    if min(
        args.layouts,
        args.in_starts,
        args.horizon,
        args.bootstrap
    ) < 1:
        parser.error(
            "layouts/in-starts/horizon/bootstrap "
            "must be >= 1"
        )

    if (
        args.ood_starts_per_side < 0
        or args.plot_layouts < 0
    ):
        parser.error(
            "ood-starts-per-side and plot-layouts "
            "must be >= 0"
        )

    if not (
        0.3 < args.ood_near
        <= args.max_y < 3.9
    ):
        parser.error(
            "Require 0.3 < ood-near <= max-y < 3.9"
        )

    torch.set_num_threads(1)

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    models = load_models(
        args.models
    )

    starts = create_starts(
        args.in_starts,
        args.ood_starts_per_side,
        args.ood_near,
        args.max_y
    )

    episodes = []
    trajectory_points = []
    layout_seeds = []

    # --------------------------------------------------------
    # Monte Carlo across obstacle layouts
    # --------------------------------------------------------

    for layout_id in range(args.layouts):
        seed = args.seed + layout_id

        # One shared obstacle layout.

        template = Drone(
            seed,
            args.horizon
        )

        eligible = []

        # Determine valid starts once.
        # All methods use the same valid starts.

        for start_id, group, y in starts:
            clearance = float(
                np.min(
                    np.linalg.norm(
                        template.cs
                        - np.array([0.0, y]),
                        axis=1
                    ) - template.rs
                )
            )

            if clearance > MARGIN:
                eligible.append(
                    (start_id, group, y)
                )

        if not eligible:
            print(
                f"Layout {seed}: "
                "no eligible starts; skipped"
            )
            continue

        layout_seeds.append(seed)

        curves = {
            (m, w): []
            for m in MODES
            for w in ("teacher", "student")
        }

        # ----------------------------------------------------
        # Same starts for all models
        # ----------------------------------------------------

        for start_id, group, y in eligible:
            for method in MODES:
                model = models[method]

                for who in (
                    "teacher",
                    "student"
                ):
                    result = simulate(
                        model,
                        who,
                        seed,
                        y,
                        args.horizon
                    )

                    if result is None:
                        raise RuntimeError(
                            "Eligible start was "
                            "unexpectedly rejected"
                        )

                    record, path = result

                    record = dict(
                        layout_seed=seed,
                        start_id=start_id,
                        group=group,
                        start_x=0.0,
                        start_y=y,
                        method=method,
                        policy=who,
                        **record
                    )

                    episodes.append(record)

                    # Save trajectory coordinates
                    # only for plotted layouts.

                    if layout_id < args.plot_layouts:
                        curves[
                            (method, who)
                        ].append(
                            (record, path)
                        )

                        for t, pt in enumerate(path):
                            trajectory_points.append({
                                "layout_seed": seed,
                                "start_id": start_id,
                                "group": group,
                                "method": method,
                                "policy": who,
                                "step": t,
                                "x": float(pt[0]),
                                "y": float(pt[1])
                            })

        # ----------------------------------------------------
        # Overlaid trajectory plot
        # ----------------------------------------------------

        if layout_id < args.plot_layouts:
            plot_overlay(
                seed,
                curves,
                template.cs,
                template.rs,

                args.out
                / f"overlay_seed_{seed}.png"
            )

        if (
            (layout_id + 1)
            % max(1, args.layouts // 10)
            == 0
        ):
            print(
                f"Layouts "
                f"{layout_id + 1}/{args.layouts}",
                flush=True
            )

    if not layout_seeds:
        raise RuntimeError(
            "No valid layouts"
        )

    # --------------------------------------------------------
    # Save individual episodes
    # --------------------------------------------------------

    save_csv(
        args.out / "episodes.csv",
        episodes
    )

    save_csv(
        args.out / "trajectory_points.csv",
        trajectory_points
    )

    # --------------------------------------------------------
    # Cluster Bootstrap Statistics
    # --------------------------------------------------------

    summary, paired = analyze(
        episodes,
        layout_seeds,
        args.bootstrap,
        args.seed + 999
    )

    save_csv(
        args.out / "summary_cluster_ci.csv",
        summary
    )

    save_csv(
        args.out / "paired_cluster_ci.csv",
        paired
    )

    plot_statistics(
        summary,
        args.out / "success_cluster_ci.png"
    )

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print(
        "\nSuccess rates: "
        "95% layout-cluster bootstrap CI"
    )

    for group in (
        "in",
        "ood"
    ):
        print(
            f"\n{GROUP_LABELS[group]}:"
        )

        for who in (
            "teacher",
            "student"
        ):
            for method in MODES:
                item = next(
                    (
                        r for r in summary
                        if (
                            r["group"] == group
                            and r["policy"] == who
                            and r["metric"] == "success"
                            and r["method"] == method
                        )
                    ),
                    None
                )

                if item:
                    print(
                        f"  {who:7s} "
                        f"{method:15s} "
                        f"{item['mean']:.1%} "
                        f"[{item['ci_low']:.1%}, "
                        f"{item['ci_high']:.1%}]"
                    )

    print(
        "\nStudent paired success difference "
        "(SITT minus baseline):"
    )

    for row in paired:
        if (
            row["policy"] == "student"
            and row["metric"] == "success"
            and row["group"] != "all"
        ):
            print(
                f"  {row['group']:3s} "
                f"vs {row['baseline']:10s}: "
                f"{row['difference']:+.1%} "
                f"[{row['ci_low']:+.1%}, "
                f"{row['ci_high']:+.1%}]"
            )

    print(
        "\nSaved:",
        args.out.resolve()
    )


if __name__ == "__main__":
    main()
