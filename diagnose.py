
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare import (
    Policies,
    DroneEnv,
    GOAL,
    ACC,
    cbf_filter,
    tensor
)


# ============================================================
# Diagnostic measurements
# ============================================================

def measure(model, obs_t, obs_s, safe_action):
    with torch.no_grad():
        ot = tensor(obs_t)[None]
        os = tensor(obs_s)[None]

        ft = model.features("teacher", ot)
        fs = model.features("student", os)

        student = model.mean(
            "student", os
        ).squeeze(0).numpy()

        teacher = model.mean(
            "teacher", ot
        ).squeeze(0).numpy()

        proxy = model.mean(
            "proxy", ot
        ).squeeze(0).numpy()

        return {
            "imitation_mse": float(
                np.mean((student - safe_action)**2)
            ),
            "nominal_mse": float(
                np.mean((student - teacher)**2)
            ),
            "feature_l1": float(
                (ft - fs).abs().mean()
            ),
            "proxy_mse": float(
                np.mean((proxy - student)**2)
            ),
            "proxy_safe_mse": float(
                np.mean((proxy - safe_action)**2)
            )
        }, student, teacher


# ============================================================
# Student evaluation
# ============================================================

def evaluate_episode(model, mode, seed, horizon):
    env = DroneEnv(
        np.random.default_rng(seed),
        horizon
    )

    # DroneEnv.__init__ already calls reset().
    positions = [env.p.copy()]
    trace = []

    previous_action = None

    success = False
    collision = False
    boundary = False
    qp_failure = False

    for step in range(horizon):
        ot = env.teacher_obs()
        os = env.student_obs()

        with torch.no_grad():
            nominal = model.mean(
                "teacher",
                tensor(ot)[None]
            ).squeeze(0).numpy()

        # Compute teacher's target action at the
        # state currently visited by the student.
        if mode == "sitt":
            safe = np.clip(
                nominal, -ACC, ACC
            )
        else:
            safe, feasible = cbf_filter(
                env.p,
                env.v,
                env.centers,
                nominal
            )

            if not feasible:
                qp_failure = True
                break

        metrics, student_action, _ = measure(
            model, ot, os, safe
        )

        if previous_action is None:
            oscillation = 0.0
        else:
            oscillation = float(
                np.sum(
                    (student_action - previous_action)**2
                )
            )

        previous_action = student_action.copy()

        old_distance = np.linalg.norm(
            GOAL - env.p
        )

        _, _, reward, done, info = env.step(
            student_action
        )

        distance = np.linalg.norm(
            GOAL - env.p
        )

        intervention = float(
            np.sum(
                (
                    safe - np.clip(
                        nominal, -ACC, ACC
                    )
                )**2
            )
        )

        trace.append({
            "step": step,
            "goal_distance": float(distance),
            "progress": float(
                old_distance - distance
            ),
            "speed": float(
                np.linalg.norm(env.v)
            ),
            "clearance": float(
                info["clearance"]
            ),
            "oscillation": oscillation,
            "intervention": intervention,
            **metrics
        })

        positions.append(
            env.p.copy()
        )

        if done:
            success = bool(info["success"])
            collision = bool(info["collision"])
            boundary = bool(info["boundary"])
            break

    return {
        "success": int(success),
        "collision": int(collision),
        "boundary": int(boundary),
        "qp_failure": int(qp_failure),
        "goal_distance": float(
            np.linalg.norm(GOAL - env.p)
        ),
        "steps": len(trace),
        "positions": np.array(positions),
        "centers": env.centers.copy(),
        "trace": trace
    }


# ============================================================
# Aggregate results
# ============================================================

def summarize(episodes):
    summary = {}

    for key in [
        "success",
        "collision",
        "boundary",
        "qp_failure",
        "goal_distance",
        "steps"
    ]:
        summary[key] = float(
            np.mean([ep[key] for ep in episodes])
        )

    for key in [
        "imitation_mse",
        "nominal_mse",
        "feature_l1",
        "proxy_mse",
        "proxy_safe_mse",
        "oscillation",
        "progress",
        "speed",
        "clearance",
        "intervention"
    ]:
        values = [
            point[key]
            for ep in episodes
            for point in ep["trace"]
        ]

        summary[key] = (
            float(np.mean(values))
            if values else float("nan")
        )

    return summary


# ============================================================
# Matplotlib diagnostics
# ============================================================

def plot_diagnostics(examples, output):
    fields = [
        ("goal_distance", "Distance to goal [m]"),
        ("imitation_mse", "Imitation MSE"),
        ("proxy_mse", "Proxy prediction MSE"),
        ("feature_l1", "Feature alignment L1"),
        ("oscillation", "Action oscillation"),
        ("clearance", "Obstacle clearance [m]")
    ]

    fig, axes = plt.subplots(
        3, 2, figsize=(13, 10)
    )

    for ax, (field, title) in zip(
        axes.flat, fields
    ):
        for mode, episode in examples.items():
            values = [
                point[field]
                for point in episode["trace"]
            ]

            ax.plot(
                range(len(values)),
                values,
                label=mode
            )

        ax.set_title(title)
        ax.set_xlabel("Simulation step")
        ax.grid(alpha=0.25)

    axes[0, 0].legend()

    fig.tight_layout()
    fig.savefig(
        output / "diagnostics.png",
        dpi=160
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        sharex=True,
        sharey=True
    )

    for ax, (mode, episode) in zip(
        axes, examples.items()
    ):
        for center in episode["centers"]:
            ax.add_patch(
                plt.Circle(
                    center,
                    0.83,
                    color="tomato",
                    alpha=0.2
                )
            )

        p = episode["positions"]

        ax.plot(
            p[:, 0],
            p[:, 1],
            linewidth=2
        )

        ax.scatter(
            [0, GOAL[0]],
            [0, GOAL[1]],
            c=["tab:blue", "tab:green"],
            s=60
        )

        ax.set_title(
            f"{mode}\n"
            f"Goal distance: "
            f"{episode['goal_distance']:.2f} m"
        )

        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output / "student_trajectories.png",
        dpi=160
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="drone_results"
    )

    parser.add_argument(
        "--out",
        default="drone_diagnostics"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5042
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=160
    )

    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []
    examples = {}

    for mode in [
        "sitt",
        "cbf",
        "sitt_cbf"
    ]:
        model = Policies()

        path = (
            Path(args.weights) /
            f"{mode}.pt"
        )

        model.load_state_dict(
            torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )
        )

        model.eval()

        episodes = [
            evaluate_episode(
                model,
                mode,
                args.seed + i,
                args.horizon
            )
            for i in range(args.episodes)
        ]

        examples[mode] = episodes[0]

        summary = summarize(episodes)

        summaries.append({
            "method": mode,
            **summary
        })

        print(f"\n[{mode}]")

        for key, value in summary.items():
            print(f"{key:20s}: {value:.5f}")

    with open(
        output / "diagnostic_metrics.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0])
        )

        writer.writeheader()
        writer.writerows(summaries)

    plot_diagnostics(
        examples,
        output
    )

    print(
        f"\nSaved diagnostics to "
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()
