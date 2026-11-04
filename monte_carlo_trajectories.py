import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_sitt_safe import Drone, Model, GOAL, MARGIN, MODES, cbf, tensor


def load_models(folder):
    models = {}
    for name in MODES:
        path = folder / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = Model()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models[name] = model
    return models


def simulate(model, who, seed, start_y, horizon):
    # Same seed => same obstacle layout for every method and start.
    env = Drone(seed, horizon)
    env.p = np.array([0.0, float(start_y)])
    env.v = np.zeros(2)
    env.t = 0

    if env.clearance() <= MARGIN:
        return None

    trace = [env.p.copy()]
    minimum = env.clearance()
    status = "timeout"
    cbf_correction = 0.0

    for _ in range(horizon):
        obs = env.teacher_obs() if who == "teacher" else env.student_obs()
        with torch.no_grad():
            action = model.action(who, tensor(obs)[None])[0].cpu().numpy()

        executed = action
        if who == "teacher":
            executed, feasible = cbf(env, action)
            if not feasible:
                status = "qp_failure"
                break
            cbf_correction += float(np.sum((executed - action) ** 2))

        _, done, info = env.step(executed)
        trace.append(env.p.copy())
        minimum = min(minimum, float(info["clearance"]))
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

    trajectory = np.asarray(trace)
    record = {
        "seed": seed,
        "start_x": 0.0,
        "start_y": float(start_y),
        "policy": who,
        "status": status,
        "success": int(status == "success"),
        "collision": int(status == "collision"),
        "qp_failure": int(status == "qp_failure"),
        "steps": len(trajectory) - 1,
        "goal_distance": float(np.linalg.norm(env.p - GOAL)),
        "min_clearance": float(minimum),
        "path_length": float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))),
        "cbf_correction": cbf_correction,
    }
    return record, trajectory, env.cs.copy(), env.rs.copy()


def draw_layout(seed, trajectories, obstacles, starts, output):
    cs, rs = obstacles
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(starts)))

    for j, method in enumerate(MODES):
        for i, who in enumerate(("teacher", "student")):
            ax = axes[i, j]
            for c, r in zip(cs, rs):
                ax.add_patch(plt.Circle(c, r + MARGIN, color="tomato", alpha=0.14))
                ax.add_patch(plt.Circle(c, r, ec="tomato", fill=False, lw=1.3))

            entries = trajectories[(method, who)]
            successes = sum(rec["success"] for rec, _ in entries)
            for k, (rec, path) in enumerate(entries):
                ax.plot(path[:, 0], path[:, 1], color=colors[k], lw=1.5, alpha=0.85)
                ax.scatter(path[0, 0], path[0, 1], c=[colors[k]], s=17)
                endpoint_color = "green" if rec["success"] else "red"
                ax.scatter(path[-1, 0], path[-1, 1], c=endpoint_color, s=11, alpha=0.8)

            ax.scatter(*GOAL, color="green", marker="*", s=120, zorder=6)
            ax.set_title(f"{method} / {who} | {successes}/{len(entries)} success")
            ax.set_xlim(-0.6, 10.9)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

    fig.suptitle(f"Same obstacles, multiple starts | environment seed {seed}")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("sitt_safe_results"))
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--starts", type=int, default=9)
    parser.add_argument("--min-y", type=float, default=-3.0)
    parser.add_argument("--max-y", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--out", type=Path, default=Path("mc_trajectories"))
    args = parser.parse_args()

    if args.layouts < 1 or args.starts < 1 or args.horizon < 1:
        parser.error("layouts, starts, horizon must be positive")
    if not (-3.9 <= args.min_y <= args.max_y <= 3.9):
        parser.error("starts must lie inside the y boundaries [-3.9, 3.9]")

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    models = load_models(args.models)
    starts = np.linspace(args.min_y, args.max_y, args.starts)
    records = []

    for n in range(args.layouts):
        seed = args.seed + n
        trajectories = {(method, who): [] for method in MODES for who in ("teacher", "student")}
        obstacles = None

        for j, start_y in enumerate(starts):
            for method, model in models.items():
                for who in ("teacher", "student"):
                    result = simulate(model, who, seed, start_y, args.horizon)
                    if result is None:
                        continue
                    record, path, cs, rs = result
                    record.update(method=method, start_id=j)
                    records.append(record)
                    trajectories[(method, who)].append((record, path))
                    obstacles = (cs, rs)

        if obstacles is not None:
            draw_layout(seed, trajectories, obstacles, starts,
                        args.out / f"trajectories_seed_{seed}.png")
        print(f"Layout {n + 1}/{args.layouts} completed (seed={seed})", flush=True)

    with (args.out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = []
    for method in MODES:
        for who in ("teacher", "student"):
            group = [r for r in records if r["method"] == method and r["policy"] == who]
            summary.append({
                "method": method,
                "policy": who,
                "episodes": len(group),
                "success_rate": np.mean([r["success"] for r in group]),
                "collision_rate": np.mean([r["collision"] for r in group]),
                "mean_min_clearance": np.mean([r["min_clearance"] for r in group]),
                "mean_path_length": np.mean([r["path_length"] for r in group]),
                "mean_goal_distance": np.mean([r["goal_distance"] for r in group]),
            })

    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for r in summary:
        print(f"{r['method']:15s} {r['policy']:7s} "
              f"success={r['success_rate']:.1%}, collision={r['collision_rate']:.1%}, "
              f"clearance={r['mean_min_clearance']:.3f}")
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
