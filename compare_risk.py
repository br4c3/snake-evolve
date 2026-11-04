
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


DT, ACC, MAX_SPEED = 0.12, 3.0, 2.4
GOAL = np.array([10.0, 0.0], dtype=float)

RAY_ANGLES = np.deg2rad(np.linspace(-50, 50, 9))
RAY_RANGE = 2.5
SAFE_MARGIN = 0.18

GRID_STEP = 0.30
RISK_SCALE = 0.55
RISK_WEIGHT = 6.0

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
# Risk-aware A* Path Planning
# ============================================================

def clearance_at(point, centers, radii):
    return float(np.min(
        np.linalg.norm(
            centers - point,
            axis=-1
        ) - radii
    ))


def segment_cost(a, b, centers, radii, risk_weight):
    distance = float(np.linalg.norm(b - a))

    if distance < 1e-9:
        return 0.0

    n = max(
        2,
        int(math.ceil(distance / 0.12))
    )

    samples = (
        a[None, :]
        + np.linspace(0, 1, n)[:, None]
        * (b - a)[None, :]
    )

    c = np.min(
        np.linalg.norm(
            samples[:, None, :]
            - centers[None, :, :],
            axis=2
        ) - radii[None, :],
        axis=1
    )

    if np.any(c < SAFE_MARGIN + 0.13):
        return float("inf")

    risk = np.exp(
        -np.maximum(c, 0.0) / RISK_SCALE
    )

    return distance * (
        1.0
        + risk_weight * float(np.mean(risk))
    )


def plan_astar(
    start,
    goal,
    centers,
    radii,
    risk_weight
):
    lo = np.array([-0.6, -4.0])
    hi = np.array([10.9, 4.0])

    width, height = (
        np.floor(
            (hi - lo) / GRID_STEP
        ).astype(int) + 1
    ).tolist()

    xs = lo[0] + GRID_STEP * np.arange(width)
    ys = lo[1] + GRID_STEP * np.arange(height)

    xx, yy = np.meshgrid(
        xs, ys, indexing="ij"
    )

    points = np.stack(
        (xx, yy),
        axis=-1
    )

    clear = np.min(
        np.linalg.norm(
            points[:, :, None, :]
            - centers[None, None, :, :],
            axis=-1
        ) - radii[None, None, :],
        axis=2
    )

    blocked = clear < SAFE_MARGIN + 0.13

    def index(p):
        q = np.round(
            (p - lo) / GRID_STEP
        ).astype(int)

        return tuple(np.clip(
            q,
            [0, 0],
            [width - 1, height - 1]
        ))

    def pos(k):
        return lo + GRID_STEP * np.array(
            k, dtype=float
        )

    source = index(start)
    target = index(goal)

    blocked[source] = False
    blocked[target] = False

    queue = [(
        float(
            np.linalg.norm(
                pos(source) - pos(target)
            )
        ),
        0.0,
        source
    )]

    distances = {source: 0.0}
    parents = {}

    moves = [
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    ]

    while queue:
        _, cost, u = heapq.heappop(queue)

        if cost > distances.get(
            u, float("inf")
        ) + 1e-9:
            continue

        if u == target:
            break

        for dx, dy in moves:
            v = (
                u[0] + dx,
                u[1] + dy
            )

            if not (
                0 <= v[0] < width
                and 0 <= v[1] < height
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

            distance = (
                GRID_STEP * math.hypot(dx, dy)
            )

            risk = math.exp(
                -max(
                    0.0,
                    min(clear[u], clear[v])
                ) / RISK_SCALE
            )

            new_cost = (
                cost
                + distance * (
                    1.0 + risk_weight * risk
                )
            )

            if new_cost + 1e-9 < distances.get(
                v, float("inf")
            ):
                distances[v] = new_cost
                parents[v] = u

                heuristic = np.linalg.norm(
                    pos(v) - pos(target)
                )

                heapq.heappush(
                    queue,
                    (
                        new_cost + heuristic,
                        new_cost,
                        v
                    )
                )

    if (
        source != target
        and target not in parents
    ):
        return None

    nodes = [target]

    while nodes[-1] != source:
        nodes.append(
            parents[nodes[-1]]
        )

    nodes.reverse()

    path = (
        [start.copy()]
        + [pos(k) for k in nodes[1:-1]]
        + [goal.copy()]
    )

    # Preserve risk-aware path cost when simplifying.
    cumulative = [0.0]

    for a, b in zip(
        path[:-1],
        path[1:]
    ):
        cumulative.append(
            cumulative[-1]
            + segment_cost(
                a, b,
                centers, radii,
                risk_weight
            )
        )

    smooth = [path[0]]
    i = 0

    while i < len(path) - 1:
        best = i + 1

        for j in range(
            len(path) - 1,
            i,
            -1
        ):
            direct = segment_cost(
                path[i],
                path[j],
                centers,
                radii,
                risk_weight
            )

            original = (
                cumulative[j]
                - cumulative[i]
            )

            if (
                np.isfinite(direct)
                and direct <= original * 1.025 + 1e-6
            ):
                best = j
                break

        smooth.append(path[best])
        i = best

    return np.asarray(
        smooth[1:],
        dtype=float
    )


# ============================================================
# Drone Environment
# ============================================================

class Env:
    def __init__(
        self,
        seed,
        horizon,
        risk_weight
    ):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.risk_weight = risk_weight
        self.reset()

    def reset(self):
        for _ in range(100):
            start = np.array([
                0.0,
                self.rng.uniform(-0.3, 0.3)
            ])

            centers = []
            radii = []

            for _ in range(7):
                for _ in range(120):
                    center = np.array([
                        self.rng.uniform(1.7, 9.2),
                        self.rng.uniform(-2.65, 2.65)
                    ])

                    radius = float(
                        self.rng.uniform(0.45, 0.78)
                    )

                    valid = (
                        np.linalg.norm(
                            center - start
                        ) > radius + 1.2

                        and np.linalg.norm(
                            center - GOAL
                        ) > radius + 1.2

                        and all(
                            np.linalg.norm(
                                center - c
                            ) > radius + r + 0.35
                            for c, r in zip(
                                centers, radii
                            )
                        )
                    )

                    if valid:
                        centers.append(center)
                        radii.append(radius)
                        break
                else:
                    break

            if len(centers) != 7:
                continue

            centers = np.asarray(centers)
            radii = np.asarray(radii)

            path = plan_astar(
                start,
                GOAL,
                centers,
                radii,
                self.risk_weight
            )

            if path is None:
                continue

            self.p = start
            self.v = np.zeros(2)

            self.cs = centers
            self.rs = radii

            self.path = path
            self.wp = 0
            self.steps = 0
            return

        raise RuntimeError(
            "Could not generate an A*-reachable scenario"
        )

    def waypoint(self):
        while (
            self.wp < len(self.path) - 1
            and np.linalg.norm(
                self.p - self.path[self.wp]
            ) < 0.40
        ):
            self.wp += 1

        return self.path[self.wp]

    def teacher_obs(self):
        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / MAX_SPEED,
            np.column_stack((
                (self.cs - self.p) / 10,
                self.rs
            )).ravel()
        )).astype("float32")

    def student_obs(self):
        readings = []

        for angle in RAY_ANGLES:
            direction = np.array((
                math.cos(angle),
                math.sin(angle)
            ))

            best = RAY_RANGE

            for center, radius in zip(
                self.cs,
                self.rs
            ):
                delta = self.p - center
                b = direction @ delta

                disc = (
                    b * b
                    - (
                        delta @ delta
                        - radius * radius
                    )
                )

                if disc >= 0:
                    hit = -b - math.sqrt(disc)

                    if 0 <= hit < best:
                        best = hit

            readings.append(
                best / RAY_RANGE
            )

        return np.concatenate((
            (GOAL - self.p) / 10,
            self.v / MAX_SPEED,
            readings
        )).astype("float32")

    def step(self, action):
        u = np.clip(
            np.asarray(action, dtype=float),
            -ACC,
            ACC
        )

        old = float(
            np.linalg.norm(self.p - GOAL)
        )

        next_v = self.v + DT * u

        if np.linalg.norm(next_v) > MAX_SPEED:
            next_v *= (
                MAX_SPEED / np.linalg.norm(next_v)
            )

        new_p = (
            self.p
            + DT * (self.v + next_v) / 2
        )

        # Swept collision checking.
        segment = np.linspace(
            self.p,
            new_p,
            7
        )

        min_clearance = float(np.min(
            np.linalg.norm(
                segment[:, None, :]
                - self.cs[None, :, :],
                axis=-1
            ) - self.rs[None, :]
        ))

        self.p = new_p
        self.v = next_v
        self.steps += 1

        distance = float(
            np.linalg.norm(self.p - GOAL)
        )

        collision = min_clearance < 0

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
            3 * (old - distance)
            - 0.008 * (u @ u)
            - 0.012
        )

        if collision or boundary:
            reward -= 30

        if success:
            reward += 50

        risk = math.exp(
            -max(min_clearance, 0) / RISK_SCALE
        )

        return (
            float(reward),
            bool(
                collision
                or success
                or boundary
                or timeout
            ),
            {
                "success": bool(success),
                "collision": bool(collision),
                "boundary": bool(boundary),
                "timeout": bool(timeout),
                "clearance": min_clearance,
                "risk": risk
            }
        )


# ============================================================
# A* Expert
# ============================================================

def expert(env):
    delta = env.waypoint() - env.p
    distance = np.linalg.norm(delta)

    desired = (
        delta / max(distance, 1e-8)
        * min(1.65, 1.5 * distance)
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
    nominal = np.clip(
        np.asarray(nominal, float),
        -ACC,
        ACC
    )

    ep = env.p - env.waypoint()
    ev = env.v + kp * ep

    V = 0.5 * (
        ep @ ep + ev @ ev
    )

    bias = (
        ep @ env.v
        + kp * (ev @ env.v)
        + rate * V
    )

    A = []
    b = []

    for center, radius in zip(
        env.cs,
        env.rs
    ):
        d = env.p - center

        h = (
            d @ d
            - (radius + SAFE_MARGIN)**2
        )

        hdot = 2 * d @ env.v

        A.append(2 * d)

        b.append(
            -2 * (env.v @ env.v)
            - 3.2 * hdot
            - 2.0 * h
        )

    A = np.asarray(A)
    b = np.asarray(b)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z:
                z[2] - (
                    ev @ z[:2] + bias
                ),
            "jac": lambda z: np.array([
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
                np.column_stack((
                    A,
                    np.zeros(len(A))
                ))
        }
    ]

    x0 = np.array([
        *nominal,
        max(0.0, ev @ nominal + bias)
    ])

    sol = minimize(
        lambda z:
            0.5 * np.sum(
                (z[:2] - nominal)**2
            )
            + 0.5 * rho * z[2]**2,

        x0,

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
            "maxiter": 70
        }
    )

    z = sol.x

    feasible = (
        np.all(np.isfinite(z))
        and np.all(A @ z[:2] - b >= -1e-5)
        and z[2] >= -1e-5
        and (
            z[2]
            - (ev @ z[:2] + bias)
        ) >= -1e-5
    )

    if feasible:
        return z[:2], True, float(z[2])

    return None, False, float("nan")


# ============================================================
# Teacher / Student / Proxy
# ============================================================

class Policies(nn.Module):
    def __init__(self):
        super().__init__()

        def encoder(n):
            return nn.Sequential(
                nn.Linear(n, 64),
                nn.Tanh(),

                nn.Linear(64, 64),
                nn.Tanh(),

                nn.Linear(64, 32),
                nn.Tanh()
            )

        self.teacher = encoder(25)
        self.student = encoder(13)
        self.proxy = encoder(25)

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

    def features(self, name, obs):
        return getattr(self, name)(obs)

    def mean(self, name, obs):
        return ACC * torch.tanh(
            self.decoder(
                self.features(name, obs)
            )
        )

    def dist(self, obs):
        return Normal(
            self.mean("teacher", obs),
            self.log_std.exp()
        )


# ============================================================
# Expert Demonstrations
# ============================================================

def demos(seed, episodes, horizon, risk_weight):
    env = Env(
        seed,
        horizon,
        risk_weight
    )

    obs = []
    actions = []
    wins = 0

    for _ in range(episodes):
        env.reset()
        episode = []

        for _ in range(horizon):
            o = env.teacher_obs()
            action = expert(env)

            episode.append((
                o,
                action.copy()
            ))

            _, done, info = env.step(action)

            if done:
                if info["success"]:
                    wins += 1

                    obs.extend(
                        x for x, _ in episode
                    )

                    actions.extend(
                        a for _, a in episode
                    )

                break

    if not obs:
        raise RuntimeError(
            "No successful expert demonstrations"
        )

    return (
        tensor(obs),
        tensor(actions),
        wins
    )


# ============================================================
# Pretraining
# ============================================================

def pretrain(model, dataset, epochs):
    X, Y = dataset

    opt = torch.optim.Adam(
        list(model.teacher.parameters())
        + list(model.decoder.parameters()),
        lr=1e-3
    )

    for _ in range(epochs):
        for idx in torch.randperm(
            len(X)
        ).split(256):

            loss = (
                model.mean(
                    "teacher",
                    X[idx]
                ) - Y[idx]
            ).square().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


# ============================================================
# Collect PPO Rollouts
# ============================================================

def collect(
    model,
    mode,
    seed,
    episodes,
    horizon,
    risk_lambda
):
    risk_weight = (
        RISK_WEIGHT
        if mode != "cbf_only"
        else 0.0
    )

    env = Env(
        seed,
        horizon,
        risk_weight
    )

    rows = []
    pairs = []
    qp_failures = 0

    for _ in range(episodes):
        env.reset()

        trajectory = []
        terminal = False

        for t in range(horizon):
            ot = env.teacher_obs()
            os = env.student_obs()

            with torch.no_grad():
                o = tensor(ot)[None]

                dist = model.dist(o)
                raw = dist.sample()

                nominal = (
                    raw.squeeze(0).numpy()
                )

                logp = dist.log_prob(
                    raw
                ).sum(-1).item()

                value = model.critic(
                    o
                ).item()

                if mode == "sitt_cbf_risk":
                    proxy = model.mean(
                        "proxy", o
                    )

                    kl = 0.5 * (
                        (
                            (dist.mean - proxy)
                            / dist.stddev
                        )**2
                    ).sum().item()
                else:
                    kl = 0.0

            safe, ok, _ = clf_cbf_qp(
                env, nominal
            )

            if not ok:
                qp_failures += 1

                trajectory.append((
                    ot,
                    nominal,
                    logp,
                    value,
                    -30.0
                ))

                terminal = True
                break

            reward, done, info = env.step(safe)

            # Risk penalty
            if mode != "cbf_only":
                reward -= (
                    risk_lambda * info["risk"]
                )

            # Student-informed alignment
            if mode == "sitt_cbf_risk":
                reward -= (
                    0.025 * min(kl, 30.0)
                )

                reward -= (
                    0.025 * float(
                        np.sum(
                            (safe - nominal)**2
                        )
                    )
                )

            trajectory.append((
                ot,
                nominal,
                logp,
                value,
                reward
            ))

            if t % 2 == 0:
                pairs.append((
                    ot.copy(),
                    os.copy(),
                    safe.copy()
                ))

            terminal = done

            if done:
                break

        with torch.no_grad():
            if terminal:
                next_value = 0.0
            else:
                next_value = model.critic(
                    tensor(
                        env.teacher_obs()
                    )[None]
                ).item()

        gae = 0.0
        temp = []

        for ot, u, lp, val, reward in reversed(
            trajectory
        ):
            delta = (
                reward
                + 0.99 * next_value
                - val
            )

            gae = (
                delta
                + 0.99 * 0.95 * gae
            )

            temp.append((
                ot,
                u,
                lp,
                gae + val,
                gae
            ))

            next_value = val

        rows.extend(reversed(temp))

    return rows, pairs, qp_failures


# ============================================================
# PPO / Student / Proxy Updates
# ============================================================

def update(model, opts, mode, rows, pairs, dataset):
    opt_t, opt_s, opt_p = opts

    # Teacher PPO
    if len(rows) >= 2:
        obs, actions, oldlp, returns, adv = map(
            tensor,
            zip(*rows)
        )

        adv = (
            (adv - adv.mean())
            / (
                adv.std(unbiased=False)
                + 1e-6
            )
        )

        X, Y = dataset

        for _ in range(3):
            for idx in torch.randperm(
                len(obs)
            ).split(256):

                dist = model.dist(obs[idx])

                ratio = torch.exp(
                    dist.log_prob(
                        actions[idx]
                    ).sum(-1)
                    - oldlp[idx]
                )

                pg = -torch.minimum(
                    ratio * adv[idx],
                    torch.clamp(
                        ratio,
                        0.8,
                        1.2
                    ) * adv[idx]
                ).mean()

                value_loss = (
                    model.critic(
                        obs[idx]
                    ).squeeze(-1)
                    - returns[idx]
                ).square().mean()

                loss = (
                    pg
                    + 0.35 * value_loss
                    - 0.002 * dist.entropy().sum(
                        -1
                    ).mean()
                )

                # Expert regularization
                ii = torch.randint(
                    len(X),
                    (min(256, len(idx)),)
                )

                loss += (
                    model.mean(
                        "teacher",
                        X[ii]
                    ) - Y[ii]
                ).square().mean()

                # SITT KL gradient
                if mode == "sitt_cbf_risk":
                    with torch.no_grad():
                        proxy = model.mean(
                            "proxy",
                            obs[idx]
                        )

                    std = (
                        model.log_std
                        .detach()
                        .exp()
                    )

                    kl_loss = (
                        0.5 * (
                            (
                                model.mean(
                                    "teacher",
                                    obs[idx]
                                ) - proxy
                            ) / std
                        ).square().sum(-1)
                    ).mean()

                    loss += 0.005 * kl_loss

                opt_t.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(model.teacher.parameters())
                    + list(model.decoder.parameters())
                    + list(model.critic.parameters())
                    + [model.log_std],
                    1.0
                )

                opt_t.step()

    # Student / Proxy alignment
    if pairs:
        ot, os, targets = map(
            tensor,
            zip(*pairs)
        )

        for _ in range(3):
            for idx in torch.randperm(
                len(ot)
            ).split(256):

                with torch.no_grad():
                    ft = model.features(
                        "teacher",
                        ot[idx]
                    )

                fs = model.features(
                    "student",
                    os[idx]
                )

                bc = (
                    model.mean(
                        "student",
                        os[idx]
                    ) - targets[idx]
                ).square().mean()

                feature_loss = (
                    fs - ft
                ).abs().mean()

                loss = (
                    bc
                    + 0.25 * feature_loss
                )

                opt_s.zero_grad()
                loss.backward()
                opt_s.step()

                if mode == "sitt_cbf_risk":
                    with torch.no_grad():
                        fs_target = model.features(
                            "student",
                            os[idx]
                        )

                    ploss = (
                        model.features(
                            "proxy",
                            ot[idx]
                        ) - fs_target
                    ).abs().mean()

                    opt_p.zero_grad()
                    ploss.backward()
                    opt_p.step()


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, mode, who, seeds, horizon):
    records = []
    example = None

    risk_weight = (
        0.0
        if mode == "cbf_only"
        else RISK_WEIGHT
    )

    for seed in seeds:
        env = Env(
            seed,
            horizon,
            risk_weight
        )

        positions = [env.p.copy()]

        success = False
        collision = False
        qp_failure = False
        boundary = False

        reward_sum = 0.0
        intervention_sum = 0.0
        slack_sum = 0.0
        risk_sum = 0.0

        min_clearance = float("inf")

        for _ in range(horizon):
            if who == "teacher":
                obs = env.teacher_obs()
            else:
                obs = env.student_obs()

            with torch.no_grad():
                nominal = model.mean(
                    who,
                    tensor(obs)[None]
                ).squeeze(0).numpy()

            applied = nominal

            if who == "teacher":
                applied, ok, slack = clf_cbf_qp(
                    env,
                    nominal
                )

                if not ok:
                    qp_failure = True
                    break

                intervention_sum += float(
                    np.sum(
                        (applied - nominal)**2
                    )
                )

                slack_sum += slack

            reward, done, info = env.step(
                applied
            )

            reward_sum += reward
            risk_sum += info["risk"]

            min_clearance = min(
                min_clearance,
                info["clearance"]
            )

            positions.append(
                env.p.copy()
            )

            if done:
                success = info["success"]
                collision = info["collision"]
                boundary = info["boundary"]
                break

        trajectory = np.asarray(positions)

        path_length = float(
            np.sum(
                np.linalg.norm(
                    np.diff(
                        trajectory,
                        axis=0
                    ),
                    axis=1
                )
            )
        )

        records.append({
            "success": int(success),
            "collision": int(collision),
            "boundary": int(boundary),
            "qp_failure": int(qp_failure),
            "goal_distance": float(
                np.linalg.norm(
                    env.p - GOAL
                )
            ),
            "steps": len(positions) - 1,
            "return_": reward_sum,
            "min_clearance": min_clearance,
            "path_length": path_length,
            "mean_risk": (
                risk_sum
                / max(1, len(positions) - 1)
            ),
            "cbf_intervention": intervention_sum,
            "clf_slack": slack_sum
        })

        if example is None:
            example = (
                trajectory,
                env.cs.copy(),
                env.rs.copy(),
                env.path.copy()
            )

    summary = {
        key: float(np.mean([
            r[key] for r in records
        ]))
        for key in records[0]
    }

    return summary, example


# ============================================================
# Matplotlib Visualization
# ============================================================

def plot_results(results, examples, curves, out):
    fig, axes = plt.subplots(
        2,
        len(MODES),
        figsize=(16, 8),
        sharex=True,
        sharey=True
    )

    for j, mode in enumerate(MODES):
        for i, who in enumerate((
            "teacher",
            "student"
        )):
            ax = axes[i, j]

            path, centers, radii, planned = examples[
                (mode, who)
            ]

            for c, r in zip(
                centers,
                radii
            ):
                ax.add_patch(
                    plt.Circle(
                        c,
                        r + SAFE_MARGIN,
                        color="tomato",
                        alpha=0.15
                    )
                )

                ax.add_patch(
                    plt.Circle(
                        c,
                        r,
                        color="tomato",
                        fill=False,
                        lw=1.2
                    )
                )

            if i == 0:
                planning = np.vstack((
                    path[0],
                    planned
                ))

                ax.plot(
                    planning[:, 0],
                    planning[:, 1],
                    "--",
                    alpha=0.6,
                    label=(
                        "Risk-aware A*"
                        if mode != "cbf_only"
                        else "Shortest A*"
                    )
                )

            ax.plot(
                path[:, 0],
                path[:, 1],
                lw=2,
                label="Executed trajectory"
            )

            ax.scatter(
                *path[0],
                s=35
            )

            ax.scatter(
                *GOAL,
                c="green",
                marker="*",
                s=110
            )

            stat = results[(mode, who)]

            ax.set_title(
                f"{mode}: {who}\n"
                f"Success {stat['success']:.0%} | "
                f"Collision {stat['collision']:.0%}"
            )

            ax.set_xlim(-0.6, 11)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

            if i == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()

    fig.savefig(
        out / "comparison.png",
        dpi=150
    )

    plt.close(fig)

    # Learning curves
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    for mode, history in curves.items():
        for ax, key in zip(
            axes,
            ("teacher", "student")
        ):
            ax.plot(
                [t[0] for t in history],
                [
                    t[
                        1 if key == "teacher" else 2
                    ]
                    for t in history
                ],
                marker="o",
                label=mode
            )

    for ax, title in zip(
        axes,
        ("Teacher success", "Student success")
    ):
        ax.set_title(title)
        ax.set_xlabel("PPO iteration")
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
        default=30
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=8
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50
    )

    parser.add_argument(
        "--expert-episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=80
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=200
    )

    parser.add_argument(
        "--risk-lambda",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--out",
        default="drone_risk_results"
    )

    args = parser.parse_args()

    if args.quick:
        args.iterations = 1
        args.train_episodes = 2
        args.eval_episodes = 4
        args.expert_episodes = 12
        args.pretrain_epochs = 15
        args.horizon = 150

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    seed_all(args.seed)

    datasets = {}
    initial_states = {}

    # --------------------------------------------------------
    # Pretrain shortest-path and risk-aware teachers
    # --------------------------------------------------------

    for planner_name, weight in [
        ("shortest", 0.0),
        ("risk", RISK_WEIGHT)
    ]:
        X, Y, wins = demos(
            args.seed + 10,
            args.expert_episodes,
            args.horizon,
            weight
        )

        print(
            f"{planner_name} expert: "
            f"{wins}/{args.expert_episodes} successes, "
            f"{len(X)} samples",
            flush=True
        )

        datasets[planner_name] = (X, Y)

        seed_all(args.seed)

        net = Policies()

        pretrain(
            net,
            (X, Y),
            args.pretrain_epochs
        )

        initial_states[planner_name] = {
            k: v.detach().clone()
            for k, v in net.state_dict().items()
        }

    seeds = list(range(
        args.seed + 5000,
        args.seed + 5000 + args.eval_episodes
    ))

    results = {}
    examples = {}
    curves = {}
    rows = []

    # --------------------------------------------------------
    # Train three methods
    # --------------------------------------------------------

    for mode in MODES:
        planner = (
            "shortest"
            if mode == "cbf_only"
            else "risk"
        )

        seed_all(args.seed)

        model = Policies()

        model.load_state_dict(
            initial_states[planner]
        )

        opt_t = torch.optim.Adam(
            list(model.teacher.parameters())
            + list(model.decoder.parameters())
            + list(model.critic.parameters())
            + [model.log_std],
            lr=2e-4
        )

        opt_s = torch.optim.Adam(
            model.student.parameters(),
            lr=8e-4
        )

        opt_p = torch.optim.Adam(
            model.proxy.parameters(),
            lr=8e-4
        )

        curves[mode] = []
        failures = 0

        interval = max(
            1,
            args.iterations // 5
        )

        checkpoints = (
            {0, 1, args.iterations}
            | set(range(
                0,
                args.iterations + 1,
                interval
            ))
        )

        print(
            f"\nTraining {mode}",
            flush=True
        )

        for it in range(
            args.iterations + 1
        ):
            if it in checkpoints:
                eval_seeds = seeds[
                    :min(12, len(seeds))
                ]

                teacher, _ = evaluate(
                    model,
                    mode,
                    "teacher",
                    eval_seeds,
                    args.horizon
                )

                student, _ = evaluate(
                    model,
                    mode,
                    "student",
                    eval_seeds,
                    args.horizon
                )

                curves[mode].append((
                    it,
                    teacher["success"],
                    student["success"]
                ))

                print(
                    f"  {it}/{args.iterations} "
                    f"Teacher={teacher['success']:.0%} "
                    f"Student={student['success']:.0%}",
                    flush=True
                )

            if it == args.iterations:
                break

            experiences, pairs, bad = collect(
                model,
                mode,
                args.seed + 10000 + it,
                args.train_episodes,
                args.horizon,
                args.risk_lambda
            )

            failures += bad

            update(
                model,
                (opt_t, opt_s, opt_p),
                mode,
                experiences,
                pairs,
                datasets[planner]
            )

        torch.save(
            model.state_dict(),
            out / f"{mode}.pt"
        )

        # Final evaluation
        for who in ("teacher", "student"):
            stats, example = evaluate(
                model,
                mode,
                who,
                seeds,
                args.horizon
            )

            results[(mode, who)] = stats
            examples[(mode, who)] = example

            rows.append({
                "method": mode,
                "policy": who,
                **stats,
                "train_qp_failures": failures
            })

            print(
                f"  final {who}: "
                f"success={stats['success']:.0%}, "
                f"collision={stats['collision']:.0%}, "
                f"clearance={stats['min_clearance']:.2f}",
                flush=True
            )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with (
        out / "metrics.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_results(
        results,
        examples,
        curves,
        out
    )

    print(
        "Saved:",
        out.resolve()
    )


if __name__ == "__main__":
    main()
