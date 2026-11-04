
import argparse
import csv
import heapq
import math
import random

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

DT   = 0.12
ACC  = 3.0
VMAX = 2.4

GOAL = np.array([10.0, 0.0])

RAYS = np.deg2rad(
    np.linspace(-50, 50, 9)
)

MODES = (
    "sitt",
    "cbf_clf_bc",
    "sitt_cbf_clf"
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def T(x):
    return torch.as_tensor(
        np.asarray(x),
        dtype=torch.float32
    )


# ============================================================
# A* Path Planning
# ============================================================

def astar(start, goal, centers, radii, step=0.3):
    lo = np.array([-0.6, -4.0])
    hi = np.array([10.9, 4.0])

    size = np.floor(
        (hi - lo) / step
    ).astype(int) + 1

    def coord(k):
        return lo + step * np.array(k)

    def index(p):
        return tuple(
            np.clip(
                np.round((p - lo) / step).astype(int),
                0,
                size - 1
            )
        )

    blocked = np.zeros(tuple(size), bool)

    for i in range(size[0]):
        for j in range(size[1]):
            blocked[i, j] = np.any(
                np.linalg.norm(
                    centers - coord((i, j)),
                    axis=1
                ) < radii + 0.33
            )

    source = index(start)
    target = index(goal)

    blocked[source] = False
    blocked[target] = False

    pq = [(0.0, source)]
    dist = {source: 0.0}
    parent = {}

    moves = [
        (i, j)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if i or j
    ]

    while pq:
        _, u = heapq.heappop(pq)

        if u == target:
            break

        for dx, dy in moves:
            v = (u[0] + dx, u[1] + dy)

            if not (
                0 <= v[0] < size[0]
                and 0 <= v[1] < size[1]
            ):
                continue

            if blocked[v]:
                continue

            if dx and dy:
                if (
                    blocked[u[0] + dx, u[1]]
                    or blocked[u[0], u[1] + dy]
                ):
                    continue

            nd = dist[u] + math.hypot(dx, dy)

            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u

                heapq.heappush(
                    pq,
                    (nd + math.dist(v, target), v)
                )

    if target != source and target not in parent:
        return None

    path = [target]

    while path[-1] != source:
        path.append(parent[path[-1]])

    pts = (
        [start.copy()]
        + [coord(k) for k in reversed(path[1:-1])]
        + [goal.copy()]
    )

    def visible(a, b):
        n = max(
            2,
            int(np.linalg.norm(b - a) / 0.08) + 1
        )

        return all(
            np.all(
                np.linalg.norm(
                    centers - (a + t * (b - a)),
                    axis=1
                ) >= radii + 0.28
            )
            for t in np.linspace(0, 1, n)
        )

    result = [pts[0]]
    i = 0

    while i < len(pts) - 1:
        for j in range(len(pts) - 1, i, -1):
            if visible(pts[i], pts[j]):
                break

        result.append(pts[j])
        i = j

    return np.array(result[1:])


# ============================================================
# Drone Environment
# ============================================================

class Env:
    def __init__(self, seed, horizon):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.reset()

    def reset(self):
        for _ in range(250):
            p = np.array([
                0.0,
                self.rng.uniform(-0.35, 0.35)
            ])

            cs = []
            rs = []

            for _ in range(7):
                for _ in range(200):
                    c = np.array([
                        self.rng.uniform(1.7, 9.2),
                        self.rng.uniform(-2.7, 2.7)
                    ])

                    r = self.rng.uniform(
                        0.45, 0.78
                    )

                    valid = (
                        np.linalg.norm(c - p) > r + 1.2
                        and np.linalg.norm(c - GOAL) > r + 1.2
                        and all(
                            np.linalg.norm(c - q) > r + s + 0.35
                            for q, s in zip(cs, rs)
                        )
                    )

                    if valid:
                        cs.append(c)
                        rs.append(r)
                        break
                else:
                    break

            if len(cs) != 7:
                continue

            cs = np.array(cs)
            rs = np.array(rs)

            path = astar(p, GOAL, cs, rs)

            if path is not None and len(path) < 25:
                self.p = p
                self.v = np.zeros(2)

                self.cs = cs
                self.rs = rs

                self.path = path
                self.wp = 0
                self.steps = 0
                return

        raise RuntimeError(
            "Could not generate navigable layout"
        )

    def waypoint(self):
        while (
            self.wp < len(self.path) - 1
            and np.linalg.norm(
                self.p - self.path[self.wp]
            ) < 0.42
        ):
            self.wp += 1

        return self.path[self.wp]

    def teacher_obs(self):
        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / VMAX,
            np.column_stack((
                (self.cs - self.p) / 10,
                self.rs
            )).ravel()
        )).astype("float32")

    def student_obs(self):
        rays = []

        for ang in RAYS:
            d = np.array([
                np.cos(ang),
                np.sin(ang)
            ])

            best = 2.5

            for c, r in zip(self.cs, self.rs):
                q = self.p - c
                b = d @ q

                disc = b * b - (q @ q - r * r)

                if disc >= 0:
                    z = -b - math.sqrt(disc)

                    if 0 <= z < best:
                        best = z

            rays.append(best / 2.5)

        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / VMAX,
            rays
        )).astype("float32")

    def step(self, u):
        u = np.clip(u, -ACC, ACC)

        before = np.linalg.norm(
            self.p - GOAL
        )

        v2 = self.v + DT * u

        if np.linalg.norm(v2) > VMAX:
            v2 *= VMAX / np.linalg.norm(v2)

        self.p += DT * (self.v + v2) / 2
        self.v = v2
        self.steps += 1

        distance = np.linalg.norm(
            self.p - GOAL
        )

        clearance = np.min(
            np.linalg.norm(
                self.cs - self.p,
                axis=1
            ) - self.rs
        )

        collision = clearance < 0

        success = (
            distance < 0.5
            and np.linalg.norm(self.v) < 1.0
        )

        boundary = not (
            -0.6 < self.p[0] < 10.9
            and -4 < self.p[1] < 4
        )

        timeout = self.steps >= self.horizon

        reward = (
            3 * (before - distance)
            - 0.008 * (u @ u)
            - 0.012
        )

        if collision or boundary:
            reward -= 30

        if success:
            reward += 50

        return (
            reward,
            bool(collision or success or boundary or timeout),
            {
                "success": bool(success),
                "collision": bool(collision),
                "clearance": float(clearance),
                "boundary": bool(boundary)
            }
        )


# ============================================================
# Expert Controller
# ============================================================

def expert(env):
    delta = env.waypoint() - env.p
    d = np.linalg.norm(delta)

    desired = (
        delta / max(d, 1e-6)
        * min(1.65, 1.5 * d)
    )

    return np.clip(
        3.5 * (desired - env.v),
        -ACC,
        ACC
    )


# ============================================================
# CLF-CBF-QP
# ============================================================

def clf_cbf_qp(
    env,
    nominal,
    rho=15.0,
    kp=1.0,
    rate=0.5
):
    """
    Decision variables: z = [ux, uy, delta]

    CBF: hard safety constraint
    CLF: soft waypoint convergence constraint

    This is a discrete simulation approximation,
    not a formal safety guarantee.
    """

    p = env.p
    v = env.v
    wp = env.waypoint()

    nominal = np.clip(
        nominal,
        -ACC,
        ACC
    )

    # --------------------------------------------------------
    # Control Lyapunov Function
    # --------------------------------------------------------

    ep = p - wp
    ev = v + kp * ep

    V = 0.5 * (
        ep @ ep
        + ev @ ev
    )

    bias = (
        ep @ v
        + kp * (ev @ v)
        + rate * V
    )

    # CLF:
    # ev.T @ u + bias <= delta

    # --------------------------------------------------------
    # Control Barrier Function
    # --------------------------------------------------------

    A = []
    b = []

    for c, r in zip(env.cs, env.rs):
        d = p - c

        h = (
            d @ d
            - (r + 0.18)**2
        )

        hdot = 2 * d @ v

        A.append(2 * d)

        b.append(
            -2 * (v @ v)
            - 3.2 * hdot
            - 2 * h
        )

    A = np.asarray(A)
    b = np.asarray(b)

    # --------------------------------------------------------
    # QP Constraints
    # --------------------------------------------------------

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z:
                z[2] - (ev @ z[:2] + bias),

            "jac": lambda z:
                np.array([
                    -ev[0],
                    -ev[1],
                    1.0
                ])
        },
        {
            "type": "ineq",
            "fun": lambda z:
                A @ z[:2] - b,

            "jac": lambda z:
                np.column_stack([
                    A,
                    np.zeros(len(A))
                ])
        }
    ]

    init = np.array([
        nominal[0],
        nominal[1],
        max(0.0, ev @ nominal + bias)
    ])

    # --------------------------------------------------------
    # Minimize control modification + CLF slack
    # --------------------------------------------------------

    sol = minimize(
        fun=lambda z:
            0.5 * np.sum(
                (z[:2] - nominal)**2
            )
            + 0.5 * rho * z[2]**2,

        x0=init,

        jac=lambda z: np.array([
            z[0] - nominal[0],
            z[1] - nominal[1],
            rho * z[2]
        ]),

        bounds=[
            (-ACC, ACC),
            (-ACC, ACC),
            (0, None)
        ],

        constraints=constraints,

        method="SLSQP",

        options={
            "ftol": 1e-8,
            "maxiter": 80
        }
    )

    z = sol.x

    feasible = (
        np.all(np.isfinite(z))
        and np.all(A @ z[:2] - b >= -1e-5)
        and z[2] >= -1e-5
        and z[2] - (ev @ z[:2] + bias) >= -1e-5
    )

    return (
        z[:2] if feasible else np.zeros(2),
        bool(feasible),
        float(z[2])
    )


# ============================================================
# Neural Network
# ============================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def enc(n):
            return nn.Sequential(
                nn.Linear(n, 64),
                nn.Tanh(),

                nn.Linear(64, 64),
                nn.Tanh(),

                nn.Linear(64, 32),
                nn.Tanh()
            )

        self.teacher = enc(25)
        self.student = enc(13)
        self.proxy = enc(25)

        self.decoder = nn.Linear(32, 2)

        self.critic = nn.Sequential(
            nn.Linear(25, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )

        self.log_std = nn.Parameter(
            torch.full((2,), -1.5)
        )

    def features(self, k, o):
        return getattr(self, k)(o)

    def mean(self, k, o):
        return ACC * torch.tanh(
            self.decoder(
                self.features(k, o)
            )
        )

    def dist(self, o):
        return Normal(
            self.mean("teacher", o),
            self.log_std.exp()
        )


# ============================================================
# Expert Demonstration Dataset
# ============================================================

def demo_dataset(seed, n, horizon):
    env = Env(seed, horizon)

    xs = []
    ys = []
    wins = 0

    for _ in range(n):
        env.reset()
        episode = []

        for _ in range(horizon):
            o = env.teacher_obs()
            u = expert(env)

            episode.append((
                o,
                u.copy()
            ))

            _, done, info = env.step(u)

            if done:
                if info["success"]:
                    wins += 1

                    xs.extend(
                        x for x, _ in episode
                    )

                    ys.extend(
                        y for _, y in episode
                    )
                break

    if not xs:
        raise RuntimeError(
            "A* expert did not finish any episodes"
        )

    return T(xs), T(ys), wins


# ============================================================
# Teacher Pretraining
# ============================================================

def pretrain(net, dataset, epochs):
    x, y = dataset

    opt = torch.optim.Adam(
        list(net.teacher.parameters())
        + list(net.decoder.parameters()),
        lr=1e-3
    )

    for _ in range(epochs):
        for ids in torch.randperm(
            len(x)
        ).split(256):

            loss = (
                net.mean("teacher", x[ids])
                - y[ids]
            ).square().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


# ============================================================
# PPO Rollout
# ============================================================

def rollout(net, mode, seed, episodes, horizon):
    env = Env(seed, horizon)

    rows = []
    pairs = []
    failures = 0

    for _ in range(episodes):
        env.reset()

        traj = []
        terminal = False

        for step in range(horizon):
            ot = env.teacher_obs()
            os = env.student_obs()

            with torch.no_grad():
                o = T(ot)[None]

                dist = net.dist(o)
                raw = dist.sample()

                u = raw.squeeze(0).numpy()

                lp = dist.log_prob(
                    raw
                ).sum().item()

                value = net.critic(o).item()

                proxy = net.mean(
                    "proxy",
                    o
                )

                if mode != "cbf_clf_bc":
                    kl = (
                        0.5 * (
                            (dist.mean - proxy)
                            / dist.stddev
                        ).square()
                    ).sum().item()
                else:
                    kl = 0.0

            applied = np.clip(
                u,
                -ACC,
                ACC
            )

            # Apply CLF-CBF-QP for safety-based methods.
            if mode != "sitt":
                applied, ok, _ = clf_cbf_qp(
                    env,
                    applied
                )

                if not ok:
                    failures += 1

                    traj.append((
                        ot, u, lp, value, -30.0
                    ))

                    terminal = True
                    break

            reward, done, _ = env.step(applied)

            # SITT alignment reward
            if mode != "cbf_clf_bc":
                reward -= 0.025 * min(kl, 30.0)

            # CBF intervention penalty
            if mode == "sitt_cbf_clf":
                reward -= 0.025 * np.sum(
                    (applied - u)**2
                )

            traj.append((
                ot,
                u,
                lp,
                value,
                reward
            ))

            if step % 2 == 0:
                pairs.append((
                    ot,
                    os,
                    applied.copy()
                ))

            terminal = done

            if done:
                break

        with torch.no_grad():
            if terminal:
                next_value = 0.0
            else:
                next_value = net.critic(
                    T(env.teacher_obs())[None]
                ).item()

        gae = 0.0
        back = []

        for ot, u, lp, val, reward in reversed(traj):
            delta = (
                reward
                + 0.99 * next_value
                - val
            )

            gae = (
                delta
                + 0.99 * 0.95 * gae
            )

            back.append((
                ot,
                u,
                lp,
                gae + val,
                gae
            ))

            next_value = val

        rows.extend(reversed(back))

    return rows, pairs, failures


# ============================================================
# PPO + SITT Updates
# ============================================================

def train_update(
    net,
    opt_t,
    opt_s,
    opt_p,
    mode,
    rows,
    pairs,
    demo
):
    # --------------------------------------------------------
    # Teacher PPO
    # --------------------------------------------------------

    if len(rows) >= 2:
        o, a, lp, ret, adv = map(
            T,
            zip(*rows)
        )

        adv = (
            (adv - adv.mean())
            / (adv.std(unbiased=False) + 1e-6)
        )

        x, y = demo

        for _ in range(3):
            for ix in torch.randperm(
                len(o)
            ).split(256):

                dist = net.dist(o[ix])

                ratio = torch.exp(
                    dist.log_prob(
                        a[ix]
                    ).sum(-1) - lp[ix]
                )

                pg = -torch.min(
                    ratio * adv[ix],
                    torch.clamp(
                        ratio,
                        0.8,
                        1.2
                    ) * adv[ix]
                ).mean()

                val_loss = (
                    net.critic(
                        o[ix]
                    ).squeeze(-1) - ret[ix]
                ).square().mean()

                loss = (
                    pg
                    + 0.35 * val_loss
                    - 0.002 * dist.entropy().sum(-1).mean()
                )

                # Expert regularization
                j = torch.randint(
                    len(x),
                    (min(256, len(ix)),)
                )

                loss += (
                    net.mean(
                        "teacher",
                        x[j]
                    ) - y[j]
                ).square().mean()

                # SITT direct KL alignment
                if mode != "cbf_clf_bc":
                    with torch.no_grad():
                        proxy = net.mean(
                            "proxy",
                            o[ix]
                        )

                    std = net.log_std.detach().exp()

                    kl_loss = (
                        0.5 * (
                            (net.mean("teacher", o[ix]) - proxy)
                            / std
                        ).square().sum(-1)
                    ).mean()

                    loss += 0.005 * kl_loss

                opt_t.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(net.teacher.parameters())
                    + list(net.decoder.parameters())
                    + list(net.critic.parameters())
                    + [net.log_std],
                    1.0
                )

                opt_t.step()

    # --------------------------------------------------------
    # Student / Proxy Alignment
    # --------------------------------------------------------

    if pairs:
        ot, os, targets = map(
            T,
            zip(*pairs)
        )

        for _ in range(3):
            for ix in torch.randperm(
                len(ot)
            ).split(256):

                with torch.no_grad():
                    ft = net.features(
                        "teacher",
                        ot[ix]
                    )

                fs = net.features(
                    "student",
                    os[ix]
                )

                pred = net.mean(
                    "student",
                    os[ix]
                )

                bc_loss = (
                    pred - targets[ix]
                ).square().mean()

                feature_loss = (
                    fs - ft
                ).abs().mean()

                loss = (
                    bc_loss
                    + 0.25 * feature_loss
                )

                opt_s.zero_grad()
                loss.backward()
                opt_s.step()

                if mode != "cbf_clf_bc":
                    with torch.no_grad():
                        target = net.features(
                            "student",
                            os[ix]
                        )

                    loss_p = (
                        net.features(
                            "proxy",
                            ot[ix]
                        ) - target
                    ).abs().mean()

                    opt_p.zero_grad()
                    loss_p.backward()
                    opt_p.step()


# ============================================================
# Evaluation
# ============================================================

def evaluate(net, mode, who, seeds, horizon):
    records = []
    sample = None

    for seed in seeds:
        env = Env(seed, horizon)

        path = [env.p.copy()]

        reward_sum = 0.0
        success = False
        collision = False
        qp_fail = False

        slack = 0.0
        correction = 0.0

        for _ in range(horizon):
            if who == "teacher":
                obs = env.teacher_obs()
            else:
                obs = env.student_obs()

            with torch.no_grad():
                u = net.mean(
                    who,
                    T(obs)[None]
                ).squeeze(0).numpy()

            applied = u

            # CBF/CLF applied to teacher only.
            if who == "teacher" and mode != "sitt":
                applied, ok, s = clf_cbf_qp(
                    env,
                    u
                )

                if not ok:
                    qp_fail = True
                    break

                slack += s

                correction += np.sum(
                    (applied - u)**2
                )

            reward, done, info = env.step(applied)

            reward_sum += reward
            path.append(env.p.copy())

            if done:
                success = info["success"]
                collision = info["collision"]
                break

        records.append({
            "success": int(success),
            "collision": int(collision),
            "qp_failure": int(qp_fail),
            "goal_distance": float(
                np.linalg.norm(env.p - GOAL)
            ),
            "steps": len(path) - 1,
            "episode_return": reward_sum,
            "cbf_intervention": correction,
            "clf_slack": slack
        })

        if sample is None:
            sample = (
                np.asarray(path),
                env.cs.copy(),
                env.rs.copy()
            )

    return {
        k: float(np.mean([
            r[k] for r in records
        ]))
        for k in records[0]
    }, sample


# ============================================================
# Visualization
# ============================================================

def plot(results, samples, curves, out):
    fig, axs = plt.subplots(
        2, 3,
        figsize=(15, 8),
        sharex=True,
        sharey=True
    )

    for j, mode in enumerate(MODES):
        for i, who in enumerate(
            ("teacher", "student")
        ):
            ax = axs[i, j]

            path, cs, rs = samples[
                (mode, who)
            ]

            for c, r in zip(cs, rs):
                ax.add_patch(
                    plt.Circle(
                        c,
                        r + 0.18,
                        color="tomato",
                        alpha=0.17
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
                path[:, 1]
            )

            ax.scatter(
                *path[0],
                c="tab:blue"
            )

            ax.scatter(
                *GOAL,
                c="green",
                marker="*",
                s=100
            )

            z = results[(mode, who)]

            ax.set_title(
                f"{mode}: {who}\n"
                f"Success {z['success']:.0%}, "
                f"Collision {z['collision']:.0%}"
            )

            ax.set_aspect("equal")
            ax.set_xlim(-0.6, 11)
            ax.set_ylim(-4, 4)
            ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        out / "comparison.png",
        dpi=150
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Learning curves
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1, 2,
        figsize=(11, 4)
    )

    for mode, points in curves.items():
        for ax, key in zip(
            axs,
            ("teacher", "student")
        ):
            ax.plot(
                [v[0] for v in points],
                [
                    v[1 if key == "teacher" else 2]
                    for v in points
                ],
                label=mode,
                marker="o"
            )

    for ax, key in zip(
        axs,
        ("Teacher", "Student")
    ):
        ax.set_title(
            f"{key} success"
        )

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
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--quick",
        action="store_true"
    )

    ap.add_argument(
        "--iterations",
        type=int,
        default=20
    )

    ap.add_argument(
        "--train-episodes",
        type=int,
        default=8
    )

    ap.add_argument(
        "--eval-episodes",
        type=int,
        default=50
    )

    ap.add_argument(
        "--expert-episodes",
        type=int,
        default=100
    )

    ap.add_argument(
        "--pretrain-epochs",
        type=int,
        default=90
    )

    ap.add_argument(
        "--horizon",
        type=int,
        default=220
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--out",
        default="drone_clf_cbf_results"
    )

    args = ap.parse_args()

    if args.quick:
        args.iterations = 2
        args.train_episodes = 3
        args.eval_episodes = 8
        args.expert_episodes = 30
        args.pretrain_epochs = 25

    seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Expert dataset
    # --------------------------------------------------------

    x, y, wins = demo_dataset(
        args.seed + 10,
        args.expert_episodes,
        args.horizon
    )

    print(
        f"A* expert: {wins}/{args.expert_episodes}, "
        f"demonstrations: {len(x)}",
        flush=True
    )

    # --------------------------------------------------------
    # Common pretrained teacher
    # --------------------------------------------------------

    initial = Net()

    pretrain(
        initial,
        (x, y),
        args.pretrain_epochs
    )

    saved = {
        k: v.detach().clone()
        for k, v in initial.state_dict().items()
    }

    seeds = list(
        range(
            args.seed + 5000,
            args.seed + 5000 + args.eval_episodes
        )
    )

    results = {}
    samples = {}
    curves = {}
    rows = []

    # --------------------------------------------------------
    # Train all comparison methods
    # --------------------------------------------------------

    for mode in MODES:
        print(
            "Training",
            mode,
            flush=True
        )

        seed_all(args.seed)

        net = Net()
        net.load_state_dict(saved)

        opt_t = torch.optim.Adam(
            list(net.teacher.parameters())
            + list(net.decoder.parameters())
            + list(net.critic.parameters())
            + [net.log_std],
            lr=2e-4
        )

        opt_s = torch.optim.Adam(
            net.student.parameters(),
            lr=8e-4
        )

        opt_p = torch.optim.Adam(
            net.proxy.parameters(),
            lr=8e-4
        )

        curves[mode] = []
        failures = 0

        checkpoints = (
            {0, 1, args.iterations}
            | set(
                range(
                    0,
                    args.iterations + 1,
                    max(1, args.iterations // 5)
                )
            )
        )

        for it in range(args.iterations + 1):
            if it in checkpoints:
                eval_subset = seeds[
                    :min(20, len(seeds))
                ]

                te, _ = evaluate(
                    net,
                    mode,
                    "teacher",
                    eval_subset,
                    args.horizon
                )

                st, _ = evaluate(
                    net,
                    mode,
                    "student",
                    eval_subset,
                    args.horizon
                )

                curves[mode].append((
                    it,
                    te["success"],
                    st["success"]
                ))

                print(
                    f"  {it}/{args.iterations} "
                    f"teacher={te['success']:.0%} "
                    f"student={st['success']:.0%}",
                    flush=True
                )

            if it == args.iterations:
                break

            experience, pairs, bad = rollout(
                net,
                mode,
                args.seed + 10000 + it,
                args.train_episodes,
                args.horizon
            )

            failures += bad

            train_update(
                net,
                opt_t,
                opt_s,
                opt_p,
                mode,
                experience,
                pairs,
                (x, y)
            )

        torch.save(
            net.state_dict(),
            out / f"{mode}.pt"
        )

        # ----------------------------------------------------
        # Final evaluation
        # ----------------------------------------------------

        for who in ("teacher", "student"):
            stats, example = evaluate(
                net,
                mode,
                who,
                seeds,
                args.horizon
            )

            results[(mode, who)] = stats
            samples[(mode, who)] = example

            rows.append({
                "method": mode,
                "policy": who,
                **stats,
                "train_qp_failures": failures
            })

            print(
                f"  final {who}: "
                f"success={stats['success']:.0%}, "
                f"collision={stats['collision']:.0%}",
                flush=True
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        out / "metrics.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Save Matplotlib plots
    # --------------------------------------------------------

    plot(
        results,
        samples,
        curves,
        out
    )

    print(
        "Saved results to",
        out.resolve()
    )


if __name__ == "__main__":
    main()
