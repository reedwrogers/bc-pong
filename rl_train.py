"""
rl_train.py — Self-play REINFORCE for bc-pong
==============================================

Starts from the behavioural-cloning model and improves it via self-play
using REINFORCE (vanilla policy gradient).

Key speed trick — vectorised environments:
  64 parallel Pong games share ONE batched forward pass per frame,
  amortising PyTorch's overhead across all 64 games (~60× speed-up).

Algorithm:
  1. Run 64 self-play games in parallel (one frame at a time).
  2. When a point is scored: scorer's actions → +1, conceder's → -1.
  3. Every N games, do a policy-gradient update:
        loss = -mean( log π(a|s) × (reward - baseline) )
  4. Repeat.

Usage:
  python rl_train.py                          # defaults (5 000 games)
  python rl_train.py --games 20000            # longer training
"""

import argparse
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# Game constants  — must match bc_train.ipynb & play_vs_bc.py
# ═══════════════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 800, 600
PADDLE_W, PADDLE_H = 12, 80
BALL_SIZE = 12
PADDLE_SPEED = 5
BALL_SPEED_INIT = 5.0
BALL_SPEED_MAX = 14.0
BALL_SPEED_INCREMENT = 0.3
WINNING_SCORE = 5

FEATURES = [
    "ball_x_n", "ball_y_n", "ball_vx_n", "ball_vy_n",
    "left_center_y_n", "right_center_y_n",
    "left_ball_dy_n", "left_ball_dx_n",
    "right_ball_dy_n", "right_ball_dx_n",
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
N_ENVS = 64                     # parallel games per step


# ═══════════════════════════════════════════════════════════════════════════════
# Policy network  — identical to the BC model in bc_train.ipynb
# ═══════════════════════════════════════════════════════════════════════════════
class PongMLP(nn.Module):
    def __init__(self, input_dim=10, hidden=96, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised feature builder  — operates on (N,) arrays
# ═══════════════════════════════════════════════════════════════════════════════
def build_features_batch(ball_x, ball_y, ball_vx, ball_vy, left_y, right_y):
    lc = left_y  + PADDLE_H / 2
    rc = right_y + PADDLE_H / 2
    return np.stack([
        ball_x         / WIDTH,
        ball_y         / HEIGHT,
        ball_vx        / BALL_SPEED_MAX,
        ball_vy        / BALL_SPEED_MAX,
        lc             / HEIGHT,
        rc             / HEIGHT,
        (ball_y - lc)  / HEIGHT,
        (ball_x - 30)  / WIDTH,
        (ball_y - rc)  / HEIGHT,
        (758 - ball_x) / WIDTH,
    ], axis=-1).astype(np.float32)          # (N, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised game step  (physics only — no model call)
# ═══════════════════════════════════════════════════════════════════════════════
def _step_physics(ball_x, ball_y, ball_vx, ball_vy, ball_speed,
                  left_y, right_y, active):
    """Advance one frame for all *active* environments.  Returns updated arrays
    plus boolean masks `scored_left` / `scored_right`."""
    N = len(active)

    # --- move ball ---
    ball_x += ball_vx
    ball_y += ball_vy

    # top / bottom bounce
    below_top = ball_y - BALL_SIZE / 2 <= 0
    ball_y[below_top] = BALL_SIZE / 2
    ball_vy[below_top] = np.abs(ball_vy[below_top])

    above_bottom = ball_y + BALL_SIZE / 2 >= HEIGHT
    ball_y[above_bottom] = HEIGHT - BALL_SIZE / 2
    ball_vy[above_bottom] = -np.abs(ball_vy[above_bottom])

    # --- paddle collisions ---
    # left paddle (ball moving left)
    to_left = ball_vx < 0
    hit_left = (to_left &
                (ball_x - BALL_SIZE/2 < 30 + PADDLE_W) &
                (ball_x + BALL_SIZE/2 > 30) &
                (ball_y + BALL_SIZE/2 > left_y) &
                (ball_y - BALL_SIZE/2 < left_y + PADDLE_H))

    if hit_left.any():
        _deflect(ball_x, ball_y, ball_vx, ball_vy, ball_speed,
                 left_y, hit_left, side="left")

    # right paddle (ball moving right)
    rpx = WIDTH - 30 - PADDLE_W
    to_right = ball_vx > 0
    hit_right = (to_right &
                 (ball_x + BALL_SIZE/2 > rpx) &
                 (ball_x - BALL_SIZE/2 < rpx + PADDLE_W) &
                 (ball_y + BALL_SIZE/2 > right_y) &
                 (ball_y - BALL_SIZE/2 < right_y + PADDLE_H))

    if hit_right.any():
        _deflect(ball_x, ball_y, ball_vx, ball_vy, ball_speed,
                 right_y, hit_right, side="right")

    # --- scoring ---
    scored_left  = ball_x - BALL_SIZE / 2 > WIDTH
    scored_right = ball_x + BALL_SIZE / 2 < 0

    # Reset ball for scoring environments
    for mask, last in [(scored_left, "left"), (scored_right, "right")]:
        if mask.any():
            idx = np.where(mask)[0]
            ball_x[idx] = WIDTH / 2
            ball_y[idx] = HEIGHT / 2
            ball_speed[idx] = BALL_SPEED_INIT
            angles = np.random.uniform(-math.pi/4, math.pi/4, size=len(idx))
            direction = np.where(np.array([last == "right"]*len(idx)), 1, -1)
            ball_vx[idx] = direction * ball_speed[idx] * np.cos(angles)
            ball_vy[idx] = ball_speed[idx] * np.sin(angles)

    return (ball_x, ball_y, ball_vx, ball_vy, ball_speed,
            scored_left, scored_right)


def _deflect(bx, by, bvx, bvy, bs, py, mask, side):
    """In-place paddle deflection for masked environments."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    centre_y = py[idx] + PADDLE_H / 2
    offset = (by[idx] - centre_y) / (PADDLE_H / 2)
    offset = np.clip(offset, -1.0, 1.0)
    angle = offset * (math.pi / 3)

    bs[idx] = np.minimum(bs[idx] + BALL_SPEED_INCREMENT, BALL_SPEED_MAX)

    if side == "left":
        bvx[idx] =  bs[idx] * np.cos(angle)
        bx[idx]  = 30 + PADDLE_W + BALL_SIZE / 2 + 1
    else:
        bvx[idx] = -bs[idx] * np.cos(angle)
        bx[idx]  = WIDTH - 30 - PADDLE_W - BALL_SIZE / 2 - 1
    bvy[idx] = bs[idx] * np.sin(angle)


# ═══════════════════════════════════════════════════════════════════════════════
# Batched self-play  —  runs N_ENVS games concurrently
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_selfplay_batch(model: nn.Module, temperature: float = 1.0):
    """
    Run N_ENVS parallel games until *all* finish.  One forward pass
    per frame shared across all active environments.

    Returns:
        transitions:  flat list of (features_np, action_idx, reward)
        left_wins, right_wins:  counts
        total_frames:  sum of frames across all envs
    """
    N = N_ENVS

    # --- initialise all environments ---
    left_y  = np.full(N, HEIGHT / 2 - PADDLE_H / 2, dtype=np.float32)
    right_y = np.full(N, HEIGHT / 2 - PADDLE_H / 2, dtype=np.float32)
    ball_x  = np.full(N, WIDTH / 2, dtype=np.float32)
    ball_y  = np.full(N, HEIGHT / 2, dtype=np.float32)
    speed   = np.full(N, BALL_SPEED_INIT, dtype=np.float32)
    angle   = np.random.uniform(-math.pi/4, math.pi/4, size=N)
    ball_vx = speed * np.cos(angle)   # always start to the right
    ball_vy = speed * np.sin(angle)

    score_left  = np.zeros(N, dtype=np.int32)
    score_right = np.zeros(N, dtype=np.int32)
    active = np.ones(N, dtype=bool)          # True = game still going

    # per-environment point buffers
    point_bufs = [[] for _ in range(N)]      # each: list of (feats, l_idx, r_idx)

    all_transitions = []                      # flat output
    total_frames = 0
    MAX_FRAMES = 10_000

    while active.any():
        active_idx = np.where(active)[0]
        n_active = len(active_idx)

        # --- 1. features for active envs only ---
        feats = build_features_batch(
            ball_x[active_idx], ball_y[active_idx],
            ball_vx[active_idx], ball_vy[active_idx],
            left_y[active_idx], right_y[active_idx],
        )                                            # (n_active, 10)

        # --- 2. ONE forward pass for all active envs ---
        logits = model(torch.as_tensor(feats)) / temperature
        dist = torch.distributions.Categorical(logits=logits)
        li = dist.sample().numpy()                   # left action indices  (n_active,)
        ri = dist.sample().numpy()                   # right action indices (n_active,)

        # --- 3. store (state, left_idx, right_idx) in per-env buffers ---
        for j, ei in enumerate(active_idx):
            point_bufs[ei].append((feats[j].copy(), int(li[j]), int(ri[j])))

        # --- 4. move paddles ---
        left_dy  = (li - 1) * PADDLE_SPEED           # -1→-5, 0→0, 1→5
        right_dy = (ri - 1) * PADDLE_SPEED
        left_y[active_idx]  = np.clip(left_y[active_idx] + left_dy,
                                      0, HEIGHT - PADDLE_H)
        right_y[active_idx] = np.clip(right_y[active_idx] + right_dy,
                                      0, HEIGHT - PADDLE_H)

        # --- 5. physics step ---
        (ball_x, ball_y, ball_vx, ball_vy, speed,
         scored_left, scored_right) = _step_physics(
            ball_x, ball_y, ball_vx, ball_vy, speed,
            left_y, right_y, active)

        total_frames += n_active

        # --- 6. handle scoring ---
        for smask, side in [(scored_left, "left"), (scored_right, "right")]:
            if not smask.any():
                continue
            for ei in np.where(smask & active)[0]:
                if side == "left":
                    score_left[ei] += 1
                    for (f, l_i, r_i) in point_bufs[ei]:
                        all_transitions.append((f, l_i,  1.0))
                        all_transitions.append((f, r_i, -1.0))
                else:
                    score_right[ei] += 1
                    for (f, l_i, r_i) in point_bufs[ei]:
                        all_transitions.append((f, r_i,  1.0))
                        all_transitions.append((f, l_i, -1.0))
                point_bufs[ei].clear()

        # --- 7. mark finished games inactive ---
        done = (score_left >= WINNING_SCORE) | (score_right >= WINNING_SCORE)
        # safety: also deactivate games exceeding frame limit (shouldn't happen)
        # close games that just finished (flush any remaining point buffer)
        for ei in np.where(done & active)[0]:
            # discard un-scored frames in the current point buffer
            point_bufs[ei].clear()
        active[done] = False

        if total_frames > MAX_FRAMES * N:
            break

    left_wins  = int((score_left  >= WINNING_SCORE).sum())
    right_wins = int((score_right >= WINNING_SCORE).sum())

    return all_transitions, left_wins, right_wins, total_frames


# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCE update
# ═══════════════════════════════════════════════════════════════════════════════
def reinforce_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                     transitions: list, baseline: float = 0.0):
    """
    loss =  - mean( log π(a|s) × (reward − baseline) )

    Positive advantage → push action probability UP.
    Negative advantage → push action probability DOWN.
    """
    if not transitions:
        return 0.0, 0.0

    feats_b = np.stack([t[0] for t in transitions])
    acts_b  = np.array([t[1] for t in transitions], dtype=np.int64)
    rews_b  = np.array([t[2] for t in transitions], dtype=np.float32)

    feats_t = torch.as_tensor(feats_b, dtype=torch.float32)
    acts_t  = torch.as_tensor(acts_b,  dtype=torch.long)
    rews_t  = torch.as_tensor(rews_b,  dtype=torch.float32)

    logits = model(feats_t)
    log_probs = F.log_softmax(logits, dim=-1)[range(len(acts_t)), acts_t]

    advantage = rews_t - baseline
    loss = -(log_probs * advantage).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rews_t.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation  —  deterministic (greedy) self-play, no training
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: nn.Module, n_games: int = 50):
    """
    Play *n_games* with argmax actions.  Report win counts and avg rally length.
    """
    left_wins = 0
    right_wins = 0
    rally_lengths = []

    for _ in range(n_games):
        ly = HEIGHT / 2 - PADDLE_H / 2
        ry = HEIGHT / 2 - PADDLE_H / 2
        bx, by = WIDTH / 2, HEIGHT / 2
        sp = BALL_SPEED_INIT
        angle = random.uniform(-math.pi/4, math.pi/4)
        bvx = sp * math.cos(angle)
        bvy = sp * math.sin(angle)
        sl = sr = 0
        total_frames = 0
        runs = 0
        while sl < WINNING_SCORE and sr < WINNING_SCORE and total_frames < 20_000:
            total_frames += 1
            runs += 1
            # features
            lc = ly + PADDLE_H / 2
            rc = ry + PADDLE_H / 2
            f = np.array([
                bx / WIDTH, by / HEIGHT, bvx / BALL_SPEED_MAX, bvy / BALL_SPEED_MAX,
                lc / HEIGHT, rc / HEIGHT,
                (by - lc) / HEIGHT, (bx - 30) / WIDTH,
                (by - rc) / HEIGHT, (758 - bx) / WIDTH,
            ], dtype=np.float32)

            x = torch.as_tensor(f).unsqueeze(0)
            logits = model(x) / 0.01              # near-greedy
            dist = torch.distributions.Categorical(logits=logits)
            li = dist.sample().item()
            ri = dist.sample().item()
            la, ra = li - 1, ri - 1

            ly = max(0, min(HEIGHT - PADDLE_H, ly + la * PADDLE_SPEED))
            ry = max(0, min(HEIGHT - PADDLE_H, ry + ra * PADDLE_SPEED))

            # physics (inline for speed)
            bx += bvx; by += bvy
            if by - BALL_SIZE/2 <= 0:
                by = BALL_SIZE/2; bvy = abs(bvy)
            if by + BALL_SIZE/2 >= HEIGHT:
                by = HEIGHT - BALL_SIZE/2; bvy = -abs(bvy)

            # left paddle
            if bvx < 0 and bx - BALL_SIZE/2 < 30 + PADDLE_W and bx + BALL_SIZE/2 > 30 \
               and by + BALL_SIZE/2 > ly and by - BALL_SIZE/2 < ly + PADDLE_H:
                off = max(-1.0, min(1.0, (by - (ly + PADDLE_H/2)) / (PADDLE_H/2)))
                ang = off * (math.pi / 3)
                sp = min(sp + BALL_SPEED_INCREMENT, BALL_SPEED_MAX)
                bvx = sp * math.cos(ang)
                bvy = sp * math.sin(ang)
                bx = 30 + PADDLE_W + BALL_SIZE/2 + 1

            # right paddle
            rpx = WIDTH - 30 - PADDLE_W
            if bvx > 0 and bx + BALL_SIZE/2 > rpx and bx - BALL_SIZE/2 < rpx + PADDLE_W \
               and by + BALL_SIZE/2 > ry and by - BALL_SIZE/2 < ry + PADDLE_H:
                off = max(-1.0, min(1.0, (by - (ry + PADDLE_H/2)) / (PADDLE_H/2)))
                ang = off * (math.pi / 3)
                sp = min(sp + BALL_SPEED_INCREMENT, BALL_SPEED_MAX)
                bvx = -sp * math.cos(ang)
                bvy = sp * math.sin(ang)
                bx = rpx - BALL_SIZE/2 - 1

            if bx + BALL_SIZE/2 < 0:
                sr += 1; rally_lengths.append(runs); runs = 0
                bx, by = WIDTH/2, HEIGHT/2; sp = BALL_SPEED_INIT
                ang = random.uniform(-math.pi/4, math.pi/4)
                bvx = sp * math.cos(ang); bvy = sp * math.sin(ang)
            elif bx - BALL_SIZE/2 > WIDTH:
                sl += 1; rally_lengths.append(runs); runs = 0
                bx, by = WIDTH/2, HEIGHT/2; sp = BALL_SPEED_INIT
                ang = random.uniform(-math.pi/4, math.pi/4)
                bvx = -sp * math.cos(ang); bvy = sp * math.sin(ang)

        if sl >= WINNING_SCORE:
            left_wins += 1
        else:
            right_wins += 1

    avg_rally = np.mean(rally_lengths) if rally_lengths else 0.0
    return left_wins, right_wins, avg_rally


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════
def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = PongMLP()
    bc_path = os.path.join(MODEL_DIR, "bc_model.pt")
    if os.path.exists(bc_path):
        model.load_state_dict(torch.load(bc_path, map_location="cpu"))
        print(f"Loaded BC checkpoint  ←  {bc_path}")
    else:
        print("[WARN] No bc_model.pt found — starting from random weights.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Running baseline (variance reduction) ────────────────────────────────
    baseline = 0.0
    bl_alpha = 0.05

    # ── Trackers ─────────────────────────────────────────────────────────────
    recent_left  = deque(maxlen=100)
    recent_right = deque(maxlen=100)
    recent_loss  = deque(maxlen=100)

    batches_per_log = max(1, args.log_every // N_ENVS)
    batches_per_eval = max(1, args.eval_every // N_ENVS)
    batches_per_save = max(1, args.save_every // N_ENVS)

    total_games = 0
    batch = 0

    print(f"\n{'='*60}")
    print(f"Self-play REINFORCE  |  {N_ENVS} parallel envs  |  cpu")
    print(f"Games: {args.games}  |  LR: {args.lr}  |  init temp: {args.temperature}")
    print(f"{'='*60}\n")

    while total_games < args.games:
        batch += 1
        progress = total_games / args.games
        temp = max(0.05, args.temperature * (1.0 - progress))

        # ── 1. Collect 64 parallel episodes ──────────────────────────────
        model.eval()           # dropout off for collection
        transitions, lw, rw, frames = run_selfplay_batch(model, temperature=temp)
        total_games += lw + rw

        # ── 2. Policy-gradient update ────────────────────────────────────
        model.train()          # dropout on for training
        loss_val, mean_r = reinforce_update(model, optimizer, transitions, baseline)

        if transitions:
            baseline = (1 - bl_alpha) * baseline + bl_alpha * mean_r

        recent_left.append(lw)
        recent_right.append(rw)
        recent_loss.append(loss_val)

        # ── 3. Logging ───────────────────────────────────────────────────
        if batch % batches_per_log == 0 or batch == 1:
            avg_loss = np.mean(recent_loss) if recent_loss else 0.0
            lw_tot = sum(recent_left)
            rw_tot = sum(recent_right)
            n = max(1, len(recent_left) * N_ENVS)
            print(
                f"games {total_games:>7}  │  loss {avg_loss:.4f}  │  bl {baseline:+.4f}"
                f"  │  L {lw_tot}/{len(recent_left)*N_ENVS}"
                f"  R {rw_tot}/{len(recent_left)*N_ENVS}"
                f"  │  fr/batch {frames}  │  T {temp:.3f}"
            )

        # ── 4. Evaluation ────────────────────────────────────────────────
        if batch % batches_per_eval == 0:
            model.eval()
            lw_e, rw_e, rally = evaluate(model, n_games=50)
            print(
                f"  eval →  L {lw_e}/50  R {rw_e}/50"
                f"  |  avg rally {rally:.1f} frames"
            )

        # ── 5. Checkpoint ────────────────────────────────────────────────
        if batch % batches_per_save == 0:
            ckpt = os.path.join(MODEL_DIR, f"rl_g{total_games}.pt")
            torch.save({
                "games": total_games,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "baseline": baseline,
            }, ckpt)
            print(f"  → checkpoint  {ckpt}")

    # ── Final save ───────────────────────────────────────────────────────────
    final = os.path.join(MODEL_DIR, "rl_model.pt")
    torch.save(model.state_dict(), final)
    print(f"\nFinal model  →  {final}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-play REINFORCE training for bc-pong"
    )
    parser.add_argument("--games",      type=int,   default=5000,
                        help="Total self-play games (default: 5000)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial action softmax temperature (default: 1.0)")
    parser.add_argument("--log-every",  type=int,   default=500,
                        help="Print stats every N games (default: 500)")
    parser.add_argument("--eval-every", type=int,   default=2000,
                        help="Run 50-game eval every N games (default: 2000)")
    parser.add_argument("--save-every", type=int,   default=5000,
                        help="Save checkpoint every N games (default: 5000)")
    main(parser.parse_args())
