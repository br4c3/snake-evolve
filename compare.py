
import argparse
import csv
import math
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.distributions import Normal


# ============================================================
# Configuration
# ============================================================

DT = 0.12
ACC = 3.0
VMAX = 2.4

GOAL = np.array([10.0, 0.0])

MARGIN = 0.20
SIGMA = 0.55

ANGLES = np.deg2rad(
    np.linspace(-50.0, 50.0, 9)
)

MODES = (
    "cbf_only",
    "cbf_risk",
    "sitt_cbf_risk"
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def tensor(x):
    return torch.as_tensor(
        np.asarray(x),
        dtype=torch.float32
    )


# ============================================================
# Drone Environment
# ============================================================

class Drone:
    def __init__(self, seed, horizon):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.reset()

    def reset(self):
        for _ in range(250):
            start = np.array([
                0.0,
                self.rng.uniform(-0.3, 0.3)
            ])

            centers = []
            radii = []

            for _ in range(7):
                for _ in range(120):
                    c = np.array([
                        self.rng.uniform(1.6, 9.1),
                        self.rng.uniform(-2.8, 2.8)
                    ])

                    r = self.rng.uniform(
                        0.43, 0.73
                    )

                    if np.linalg.norm(c - start) < r + 1.3:
                        continue

                    if np.linalg.norm(c - GOAL) < r + 1.2:
                        continue

                    if any(
                        np.linalg.norm(c - q) < r + s + 0.28
                        for q, s in zip(centers, radii)
                    ):
                        continue

                    centers.append(c)
                    radii.append(r)
                    break
                else:
                    break

            if len(centers) == 7:
                self.p = start
                self.v = np.zeros(2)

                self.cs = np.asarray(centers)
                self.rs = np.asarray(radii)

                self.t = 0
                return

        raise RuntimeError(
            "Unable to sample obstacle layout"
        )

    # --------------------------------------------------------
    # Safety and Risk
    # --------------------------------------------------------

    def clearance(self):
        return float(np.min(
            np.linalg.norm(
                self.cs - self.p,
                axis=1
            ) - self.rs
        ))

    def risk(self):
        gaps = (
            np.linalg.norm(
                self.cs - self.p,
                axis=1
            ) - self.rs
        )

        return float(
            np.exp(
                -np.maximum(gaps, 0.0) / SIGMA
            ).sum()
        )

    # --------------------------------------------------------
    # Teacher Observation
    # --------------------------------------------------------

    def teacher_obs(self):
        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / VMAX,
            np.column_stack((
                (self.cs - self.p) / 10,
                self.rs
            )).ravel()
        )).astype(np.float32)

    # --------------------------------------------------------
    # Student Observation
    # --------------------------------------------------------

    def student_obs(self):
        readings = []

        for theta in ANGLES:
            d = np.array([
                math.cos(theta),
                math.sin(theta)
            ])

            nearest = 2.5

            for c, r in zip(self.cs, self.rs):
                q = self.p - c
                b = float(d @ q)

                disc = (
                    b * b
                    - (float(q @ q) - r * r)
                )

                if disc >= 0:
                    hit = -b - math.sqrt(disc)

                    if 0 <= hit < nearest:
                        nearest = hit

            readings.append(
                nearest / 2.5
            )

        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / VMAX,
            readings
        )).astype(np.float32)

    # --------------------------------------------------------
    # Drone Dynamics
    # --------------------------------------------------------

    def step(self, action):
        u = np.clip(
            np.asarray(action, dtype=float),
            -ACC,
            ACC
        )

        before = np.linalg.norm(
            self.p - GOAL
        )

        v_next = self.v + DT * u

        if np.linalg.norm(v_next) > VMAX:
            v_next *= (
                VMAX / np.linalg.norm(v_next)
            )

        p_next = (
            self.p
            + DT * (self.v + v_next) / 2
        )

        # Swept collision checking
        sweep = (
            self.p[None]
            + np.linspace(0, 1, 11)[:, None]
            * (p_next - self.p)
        )

        clearance = float(np.min(
            np.linalg.norm(
                sweep[:, None]
                - self.cs[None],
                axis=2
            ) - self.rs[None]
        ))

        self.p = p_next
        self.v = v_next
        self.t += 1

        distance = np.linalg.norm(
            self.p - GOAL
        )

        collision = clearance <= 0

        boundary = not (
            -0.7 < self.p[0] < 10.8
            and -4.0 < self.p[1] < 4.0
        )

        success = (
            distance < 0.5
            and np.linalg.norm(self.v) < 1.0
        )

        timeout = self.t >= self.horizon

        reward = (
            4.5 * (before - distance)
            - 0.012
            - 0.003 * float(u @ u)
        )

        if success:
            reward += 75.0

        if collision or boundary:
            reward -= 45.0

        done = (
            collision
            or boundary
            or success
            or timeout
        )

        return float(reward), bool(done), {
            "success": bool(success),
            "collision": bool(collision),
            "boundary": bool(boundary),
            "clearance": clearance,
            "risk": self.risk(),
            "timeout": bool(timeout)
        }


# ============================================================
# Control Barrier Function
# ============================================================

def cbf(env, nominal):
    """
    Continuous-time HOCBF:

        p_dot = v
        v_dot = u

    h = ||p-c||^2 - (r+margin)^2

    h_ddot + 3.2*h_dot + 2*h >= 0

    Numerical sampled-data implementation.
    Not a formal discrete-time safety guarantee.
    """

    nominal = np.clip(
        np.asarray(nominal, dtype=float),
        -ACC,
        ACC
    )

    d_all = env.p[None] - env.cs

    near = (
        np.linalg.norm(d_all, axis=1)
        < env.rs + MARGIN + 3.0
    )

    if not np.any(near):
        return nominal.copy(), True

    d = d_all[near]

    radius = (
        env.rs[near] + MARGIN
    )

    h = (
        np.sum(d * d, axis=1)
        - radius**2
    )

    hdot = 2 * (d @ env.v)

    # A @ u >= b
    A = 2 * d

    b = (
        -2 * float(env.v @ env.v)
        - 3.2 * hdot
        - 2 * h
    )

    if np.all(
        A @ nominal >= b - 1e-7
    ):
        return nominal.copy(), True

    result = minimize(
        fun=lambda u:
            0.5 * float(
                np.sum((u - nominal)**2)
            ),

        x0=nominal,

        jac=lambda u:
            u - nominal,

        bounds=[
            (-ACC, ACC),
            (-ACC, ACC)
        ],

        method="SLSQP",

        constraints=[{
            "type": "ineq",
            "fun": lambda u: A @ u - b,
            "jac": lambda u: A
        }],

        options={
            "maxiter": 80,
            "ftol": 1e-9
        }
    )

    u = result.x

    feasible = (
        np.all(np.isfinite(u))
        and np.all(
            A @ u >= b - 1e-5
        )
        and np.all(
            np.abs(u) <= ACC + 1e-6
        )
    )

    if feasible:
        return u.copy(), True

    return None, False


# ============================================================
# Teacher / Student / Proxy
# ============================================================

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def enc(n):
            return nn.Sequential(
                nn.Linear(n, 96),
                nn.Tanh(),

                nn.Linear(96, 96),
                nn.Tanh(),

                nn.Linear(96, 48),
                nn.Tanh()
            )

        self.teacher = enc(25)
        self.student = enc(13)
        self.proxy = enc(25)

        # Shared action decoder
        self.decoder = nn.Linear(48, 2)

        self.critic = nn.Sequential(
            nn.Linear(25, 96),
            nn.Tanh(),

            nn.Linear(96, 96),
            nn.Tanh(),

            nn.Linear(96, 1)
        )

        self.log_std = nn.Parameter(
            torch.tensor([-1.0, -1.0])
        )

        with torch.no_grad():
            self.decoder.bias[:] = torch.tensor([
                0.18,
                0.0
            ])

    def features(self, kind, obs):
        return getattr(
            self,
            kind
        )(obs)

    def action(self, kind, obs):
        return ACC * torch.tanh(
            self.decoder(
                self.features(kind, obs)
            )
        )

    def distribution(self, obs):
        return Normal(
            self.action("teacher", obs),
            self.log_std.exp()
        )


# ============================================================
# Replay Buffer
# ============================================================

class Replay:
    """
    Teacher states and DAgger student states
    are stored separately.
    """

    def __init__(self, capacity):
        self.teacher = deque(
            maxlen=capacity
        )

        self.student = deque(
            maxlen=capacity
        )

    def add(self, ot, os, target, source):
        item = (
            ot.copy(),
            os.copy(),
            target.copy()
        )

        getattr(
            self,
            source
        ).append(item)

    def batch(self, n, student_fraction):
        ns = (
            int(n * student_fraction)
            if self.student
            else 0
        )

        if not self.teacher:
            ns = n

        nt = n - ns

        teacher_batch = (
            random.choices(
                self.teacher,
                k=nt
            )
            if nt else []
        )

        student_batch = (
            random.choices(
                self.student,
                k=ns
            )
            if ns else []
        )

        return (
            teacher_batch
            + student_batch
        )


# ============================================================
# Student / Proxy Training
# ============================================================

def student_proxy_update(
    model,
    replay,
    opt_s,
    opt_p,
    use_sitt,
    updates,
    batch_size,
    dagger_fraction
):
    if not replay.teacher and not replay.student:
        return

    for _ in range(updates):
        batch = replay.batch(
            batch_size,
            dagger_fraction
        )

        ot, os, targets = map(
            tensor,
            zip(*batch)
        )

        # ----------------------------------------------------
        # Student Behavior Cloning
        # ----------------------------------------------------

        with torch.no_grad():
            teacher_features = model.features(
                "teacher",
                ot
            )

        fs = model.features(
            "student",
            os
        )

        us = model.action(
            "student",
            os
        )

        loss_bc = (
            us - targets
        ).square().mean()

        loss_feat = (
            fs - teacher_features
        ).abs().mean()

        loss_student = (
            loss_bc
            + 0.2 * loss_feat
        )

        opt_s.zero_grad(
            set_to_none=True
        )

        loss_student.backward()
        opt_s.step()

        # ----------------------------------------------------
        # Proxy learns ACTUAL Student behavior
        # ----------------------------------------------------

        if use_sitt:
            with torch.no_grad():
                fs_target = model.features(
                    "student",
                    os
                )

                us_target = model.action(
                    "student",
                    os
                )

            fp = model.features(
                "proxy",
                ot
            )

            up = model.action(
                "proxy",
                ot
            )

            # Proxy feature alignment
            loss_proxy_feat = (
                fp - fs_target
            ).abs().mean()

            # IMPORTANT:
            # Target = Student action, NOT Teacher action
            loss_proxy_action = (
                up - us_target
            ).square().mean()

            loss_proxy = (
                loss_proxy_feat
                + 1.0 * loss_proxy_action
            )

            opt_p.zero_grad(
                set_to_none=True
            )

            loss_proxy.backward()
            opt_p.step()


# ============================================================
# Teacher PPO Rollouts
# ============================================================

def collect_teacher(
    model,
    mode,
    seed,
    episodes,
    horizon,
    replay,
    risk_weight,
    imitation_weight,
    intervention_weight
):
    env = Drone(
        seed,
        horizon
    )

    data = []
    failures = 0
    imitation_values = []

    for _ in range(episodes):
        env.reset()

        trajectory = []
        terminal = False

        for t in range(horizon):
            ot = env.teacher_obs()
            os = env.student_obs()

            with torch.no_grad():
                x = tensor(ot)[None]

                dist = model.distribution(x)
                raw = dist.sample()

                sampled = (
                    raw[0].numpy().copy()
                )

                logp = dist.log_prob(
                    raw
                ).sum().item()

                value = model.critic(
                    x
                ).item()

            nominal = np.clip(
                sampled,
                -ACC,
                ACC
            )

            # ------------------------------------------------
            # Teacher's executed safe action
            # ------------------------------------------------

            safe, ok = cbf(
                env,
                nominal
            )

            if not ok:
                failures += 1

                trajectory.append((
                    ot,
                    sampled,
                    logp,
                    value,
                    -45.0
                ))

                terminal = True
                break

            # ------------------------------------------------
            # Student-informed imitation cost
            # ------------------------------------------------

            if mode == "sitt_cbf_risk":
                with torch.no_grad():
                    proxy_action = model.action(
                        "proxy",
                        x
                    )[0].numpy()

                imitation = float(
                    np.sum(
                        (safe - proxy_action)**2
                    )
                )

                imitation_values.append(
                    imitation
                )
            else:
                imitation = 0.0

            # ------------------------------------------------
            # Environment reward
            # ------------------------------------------------

            reward, done, info = env.step(
                safe
            )

            if mode != "cbf_only":
                reward -= (
                    risk_weight
                    * info["risk"]
                )

                # Same intervention penalty for
                # both risk-based methods.
                reward -= (
                    intervention_weight
                    * float(
                        np.sum(
                            (safe - nominal)**2
                        )
                    )
                )

            if mode == "sitt_cbf_risk":
                reward -= (
                    imitation_weight
                    * min(imitation, 36.0)
                )

            trajectory.append((
                ot,
                sampled,
                logp,
                value,
                reward
            ))

            if t % 2 == 0:
                replay.add(
                    ot,
                    os,
                    safe,
                    "teacher"
                )

            terminal = done

            if done:
                break

        # ----------------------------------------------------
        # Generalized Advantage Estimation
        # ----------------------------------------------------

        with torch.no_grad():
            bootstrap = (
                0.0
                if terminal
                else model.critic(
                    tensor(
                        env.teacher_obs()
                    )[None]
                ).item()
            )

        gae = 0.0
        reverse_rows = []

        for (
            ot,
            action,
            logp,
            value,
            reward
        ) in reversed(trajectory):

            td = (
                reward
                + 0.99 * bootstrap
                - value
            )

            gae = (
                td
                + 0.99 * 0.95 * gae
            )

            reverse_rows.append((
                ot,
                action,
                logp,
                gae + value,
                gae
            ))

            bootstrap = value

        data.extend(
            reversed(reverse_rows)
        )

    mean_imitation = (
        float(np.mean(imitation_values))
        if imitation_values
        else 0.0
    )

    return (
        data,
        failures,
        mean_imitation
    )


# ============================================================
# DAgger: Student-Visited State Collection
# ============================================================

def collect_dagger(
    model,
    seed,
    episodes,
    horizon,
    replay
):
    """
    Student executes actions WITHOUT CBF.

    At each Student-visited state,
    Teacher + CBF produces a supervised label.

    DAgger labels can be poor if Teacher is poor.
    """

    env = Drone(
        seed,
        horizon
    )

    labeled = 0
    failures = 0

    for _ in range(episodes):
        env.reset()

        for _ in range(horizon):
            ot = env.teacher_obs()
            os = env.student_obs()

            with torch.no_grad():
                us = model.action(
                    "student",
                    tensor(os)[None]
                )[0].numpy()

                ut = model.action(
                    "teacher",
                    tensor(ot)[None]
                )[0].numpy()

            # Teacher labels Student-visited states.
            label, ok = cbf(
                env,
                ut
            )

            if ok:
                replay.add(
                    ot,
                    os,
                    label,
                    "student"
                )

                labeled += 1
            else:
                failures += 1

            # Crucial: execute Student action,
            # not Teacher or CBF action.
            _, done, _ = env.step(us)

            if done:
                break

    return labeled, failures


# ============================================================
# PPO Update
# ============================================================

def update_ppo(
    model,
    optimizer,
    rows,
    epochs,
    batch_size
):
    if len(rows) < 2:
        return

    (
        obs,
        actions,
        logp,
        returns,
        advantages
    ) = map(
        tensor,
        zip(*rows)
    )

    advantage = (
        (advantages - advantages.mean())
        / (
            advantages.std(unbiased=False)
            + 1e-6
        )
    )

    for _ in range(epochs):
        indices = torch.randperm(
            len(obs)
        )

        for ids in indices.split(batch_size):
            dist = model.distribution(
                obs[ids]
            )

            ratio = torch.exp(
                dist.log_prob(
                    actions[ids]
                ).sum(-1) - logp[ids]
            )

            clipped_ratio = torch.clamp(
                ratio,
                0.8,
                1.2
            )

            loss_policy = -torch.minimum(
                ratio * advantage[ids],
                clipped_ratio * advantage[ids]
            ).mean()

            loss_critic = (
                model.critic(
                    obs[ids]
                ).squeeze(-1)
                - returns[ids]
            ).square().mean()

            entropy = dist.entropy().sum(
                -1
            ).mean()

            loss = (
                loss_policy
                + 0.4 * loss_critic
                - 0.002 * entropy
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                list(model.teacher.parameters())
                + list(model.decoder.parameters())
                + list(model.critic.parameters())
                + [model.log_std],
                1.0
            )

            optimizer.step()


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    mode,
    who,
    seeds,
    horizon
):
    results = []
    example = None

    for seed in seeds:
        env = Drone(
            seed,
            horizon
        )

        path = [
            env.p.copy()
        ]

        minimum = env.clearance()

        total_risk = 0.0
        intervention = 0.0
        imitation = 0.0
        steps = 0

        info = {
            "success": False,
            "collision": False,
            "boundary": False
        }

        qpfail = False

        for _ in range(horizon):
            ot = env.teacher_obs()
            os = env.student_obs()

            with torch.no_grad():
                obs = (
                    ot
                    if who == "teacher"
                    else os
                )

                u = model.action(
                    who,
                    tensor(obs)[None]
                )[0].numpy()

            applied = u

            # CBF is applied to Teacher only.
            if who == "teacher":
                applied, ok = cbf(
                    env,
                    u
                )

                if not ok:
                    qpfail = True
                    break

                intervention += float(
                    np.sum(
                        (applied - u)**2
                    )
                )

                if mode == "sitt_cbf_risk":
                    with torch.no_grad():
                        up = model.action(
                            "proxy",
                            tensor(ot)[None]
                        )[0].numpy()

                    imitation += float(
                        np.sum(
                            (applied - up)**2
                        )
                    )

            _, done, info = env.step(
                applied
            )

            steps += 1

            path.append(
                env.p.copy()
            )

            minimum = min(
                minimum,
                info["clearance"]
            )

            total_risk += info["risk"]

            if done:
                break

        path = np.asarray(path)

        results.append({
            "success": int(
                info["success"]
            ),

            "collision": int(
                info["collision"]
            ),

            "boundary": int(
                info["boundary"]
            ),

            "qp_failure": int(
                qpfail
            ),

            "goal_distance": float(
                np.linalg.norm(
                    env.p - GOAL
                )
            ),

            "min_clearance": minimum,

            "steps": steps,

            "path_length": float(
                np.linalg.norm(
                    np.diff(path, axis=0),
                    axis=1
                ).sum()
            ),

            "mean_risk": (
                total_risk
                / max(1, steps)
            ),

            "cbf_intervention": (
                intervention
                / max(1, steps)
            ),

            "proxy_safe_mse": (
                imitation
                / max(1, steps)
            )
        })

        if example is None:
            example = (
                path,
                env.cs.copy(),
                env.rs.copy()
            )

    summary = {
        k: float(np.mean([
            r[k]
            for r in results
        ]))
        for k in results[0]
    }

    return summary, example


# ============================================================
# Proxy Accuracy
# ============================================================

def proxy_accuracy(model, replay, n=1024):
    pool = (
        list(replay.teacher)
        + list(replay.student)
    )

    if not pool:
        return float("nan")

    batch = random.choices(
        pool,
        k=min(n, len(pool))
    )

    ot, os, _ = map(
        tensor,
        zip(*batch)
    )

    with torch.no_grad():
        up = model.action(
            "proxy",
            ot
        )

        us = model.action(
            "student",
            os
        )

        return float(
            (up - us)
            .square()
            .mean()
            .item()
        )


# ============================================================
# Visualization
# ============================================================

def make_plots(
    stats,
    samples,
    history,
    out
):
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True
    )

    for j, mode in enumerate(MODES):
        for i, who in enumerate((
            "teacher",
            "student"
        )):
            ax = axs[i, j]

            path, centers, radii = samples[
                (mode, who)
            ]

            for c, r in zip(
                centers,
                radii
            ):
                ax.add_patch(
                    plt.Circle(
                        c,
                        r + MARGIN,
                        color="tomato",
                        alpha=0.14
                    )
                )

                ax.add_patch(
                    plt.Circle(
                        c,
                        r,
                        fill=False,
                        color="tomato"
                    )
                )

            ax.plot(
                path[:, 0],
                path[:, 1],
                lw=2
            )

            ax.scatter(
                *path[0],
                color="tab:blue",
                s=30
            )

            ax.scatter(
                *GOAL,
                color="green",
                marker="*",
                s=90
            )

            d = stats[(mode, who)]

            ax.set_title(
                f"{mode} / {who}\n"
                f"Success {d['success']:.0%}, "
                f"Collision {d['collision']:.0%}"
            )

            ax.set_xlim(-0.7, 10.8)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        out / "comparison.png",
        dpi=150
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Learning Curves
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    for mode, points in history.items():
        for i, ax in enumerate(axes):
            ax.plot(
                [p[0] for p in points],
                [p[i + 1] for p in points],
                label=mode,
                marker="o"
            )

    for ax, title in zip(
        axes,
        (
            "Teacher success",
            "Student success"
        )
    ):
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        out / "learning_curves.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=150
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=12
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=60
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=180
    )

    parser.add_argument(
        "--student-updates",
        type=int,
        default=50
    )

    parser.add_argument(
        "--dagger-episodes",
        type=int,
        default=4
    )

    parser.add_argument(
        "--risk-weight",
        type=float,
        default=0.22
    )

    parser.add_argument(
        "--imitation-weight",
        type=float,
        default=0.08
    )

    parser.add_argument(
        "--intervention-weight",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--out",
        default="sitt_safe_results"
    )

    args = parser.parse_args()

    if args.quick:
        args.iterations = 2
        args.train_episodes = 2
        args.eval_episodes = 4
        args.horizon = 80
        args.student_updates = 3
        args.dagger_episodes = 1

    seed_all(args.seed)

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # Same initial network weights.
    initial_model = Model()

    initial = {
        k: v.detach().clone()
        for k, v in initial_model.state_dict().items()
    }

    eval_seeds = list(range(
        args.seed + 5000,
        args.seed + 5000 + args.eval_episodes
    ))

    results = {}
    samples = {}
    histories = {}
    csv_rows = []

    # --------------------------------------------------------
    # Train 3 Methods
    # --------------------------------------------------------

    for mode in MODES:
        seed_all(args.seed)

        model = Model()
        model.load_state_dict(initial)

        # Teacher PPO optimizer
        opt_t = torch.optim.Adam(
            list(model.teacher.parameters())
            + list(model.decoder.parameters())
            + list(model.critic.parameters())
            + [model.log_std],
            lr=3e-4
        )

        # Student optimizer
        opt_s = torch.optim.Adam(
            model.student.parameters(),
            lr=7e-4
        )

        # Proxy optimizer
        opt_p = torch.optim.Adam(
            model.proxy.parameters(),
            lr=7e-4
        )

        replay = Replay(60000)

        histories[mode] = []
        failures = 0

        for iteration in range(
            args.iterations + 1
        ):
            every = max(
                1,
                args.iterations // 8
            )

            # -----------------------------------------------
            # Periodic evaluation
            # -----------------------------------------------

            if (
                iteration == 0
                or iteration == args.iterations
                or iteration % every == 0
            ):
                subset = eval_seeds[
                    :min(12, len(eval_seeds))
                ]

                te, _ = evaluate(
                    model,
                    mode,
                    "teacher",
                    subset,
                    args.horizon
                )

                st, _ = evaluate(
                    model,
                    mode,
                    "student",
                    subset,
                    args.horizon
                )

                histories[mode].append((
                    iteration,
                    te["success"],
                    st["success"]
                ))

                proxy_mse = (
                    proxy_accuracy(model, replay)
                    if mode == "sitt_cbf_risk"
                    else float("nan")
                )

                print(
                    f"{mode:15s} {iteration:4d} | "
                    f"teacher {te['success']:.0%} "
                    f"student {st['success']:.0%} "
                    f"proxy/student MSE {proxy_mse:.4f}",
                    flush=True
                )

            if iteration == args.iterations:
                break

            # -----------------------------------------------
            # Reward Scheduling
            # -----------------------------------------------

            ramp = min(
                1.0,
                (iteration + 1)
                / max(
                    1.0,
                    args.iterations * 0.30
                )
            )

            # Allow Student and Proxy to learn first.
            imitation_ramp = max(
                0.0,
                min(
                    1.0,
                    (
                        iteration / max(1, args.iterations)
                        - 0.15
                    ) / 0.25
                )
            )

            # -----------------------------------------------
            # Teacher Rollout
            # -----------------------------------------------

            data, bad, _ = collect_teacher(
                model,
                mode,
                args.seed + 10000 + iteration,
                args.train_episodes,
                args.horizon,
                replay,
                args.risk_weight * ramp,
                args.imitation_weight * imitation_ramp,
                args.intervention_weight * ramp
            )

            failures += bad

            # -----------------------------------------------
            # PPO Update
            # -----------------------------------------------

            update_ppo(
                model,
                opt_t,
                data,
                epochs=4,
                batch_size=512
            )

            # -----------------------------------------------
            # DAgger Data Collection
            # -----------------------------------------------

            labeled, bad_dagger = collect_dagger(
                model,
                args.seed + 20000 + iteration,
                args.dagger_episodes,
                args.horizon,
                replay
            )

            failures += bad_dagger

            # -----------------------------------------------
            # Student + Proxy Update
            # -----------------------------------------------

            student_proxy_update(
                model,
                replay,
                opt_s,
                opt_p,
                mode == "sitt_cbf_risk",
                args.student_updates,
                256,
                0.5
            )

        # ----------------------------------------------------
        # Save Model
        # ----------------------------------------------------

        torch.save(
            model.state_dict(),
            out / f"{mode}.pt"
        )

        # ----------------------------------------------------
        # Final Evaluation
        # ----------------------------------------------------

        for who in (
            "teacher",
            "student"
        ):
            summary, sample = evaluate(
                model,
                mode,
                who,
                eval_seeds,
                args.horizon
            )

            results[(mode, who)] = summary
            samples[(mode, who)] = sample

            csv_rows.append({
                "method": mode,
                "policy": who,
                **summary,
                "training_qp_failures": failures,

                "proxy_student_mse": (
                    proxy_accuracy(model, replay)
                    if mode == "sitt_cbf_risk"
                    else float("nan")
                )
            })

            print(
                f"FINAL {mode}/{who}: "
                f"success {summary['success']:.0%} "
                f"collision {summary['collision']:.0%}",
                flush=True
            )

    # --------------------------------------------------------
    # Save Metrics
    # --------------------------------------------------------

    with (
        out / "metrics.csv"
    ).open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(csv_rows[0])
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # --------------------------------------------------------
    # Save Plots
    # --------------------------------------------------------

    make_plots(
        results,
        samples,
        histories,
        out
    )

    print(
        "Saved results to:",
        out.resolve()
    )


if __name__ == "__main__":
    main()
