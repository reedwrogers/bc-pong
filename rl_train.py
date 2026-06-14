"""
rl_train.py — Self-play REINFORCE for bc-pong
==============================================

Starts from the behavioural-cloning model and improves it by having it play
Pong against itself.  Designed to run headless (no pygame) so it works on
Google Colab / any GPU machine.

───────────────────────────────────────────────────────────────────────────
Algorithm:  REINFORCE  (Vanilla Policy Gradient)
───────────────────────────────────────────────────────────────────────────
This is the simplest policy-gradient RL algorithm.  The big idea:

  1.  Play an episode (game) using the current policy π(a|s).
  2.  Record every (state, action) pair.
  3.  Assign a reward:
        +1  to the paddle that scored a point
        -1  to the paddle that conceded it
  4.  Policy-gradient update:
        loss = - log π(action | state) * (reward - baseline)
  5.  Repeat thousands of times → the policy gradually improves.

The "baseline" is a running average of past rewards.  Subtracting it reduces
the variance of the gradient estimate without changing its expected direction.

Why self-play works:
  - The model plays both paddles, so it is always facing an opponent of
    equal strength.
  - When it learns a new trick (e.g. hitting the ball at a sharper angle),
    it now faces an opponent that also knows that trick → arms race.
  - Over many iterations the policy converges to strong play.

───────────────────────────────────────────────────────────────────────────
Usage
───────────────────────────────────────────────────────────────────────────
  python rl_train.py                          # defaults (5 000 episodes)
  python rl_train.py --episodes 20000         # longer training
  python rl_train.py --lr 5e-5 --episodes 10000

On Colab, clone the repo then:
  !python rl_train.py --episodes 10000
"""

import argparse
import math
import os
import random
from collections import deque
from datetime import datetime

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


# ═══════════════════════════════════════════════════════════════════════════════
# Policy network  — identical to the BC model in bc_train.ipynb
# ═══════════════════════════════════════════════════════════════════════════════
class PongMLP(nn.Module):
    """
    Input : 10 normalised game features (position, velocity, relative distances)
    Output:  3 logits →  up (0)  /  stay (1)  /  down (2)
    """
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
# Feature builder  — same normalisation as the BC notebook
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(ball_x, ball_y, ball_vx, ball_vy, left_y, right_y):
    lc = left_y  + PADDLE_H / 2   # left  paddle centre
    rc = right_y + PADDLE_H / 2   # right paddle centre
    feats = {
        "ball_x_n":         ball_x         / WIDTH,
        "ball_y_n":         ball_y         / HEIGHT,
        "ball_vx_n":        ball_vx        / BALL_SPEED_MAX,
        "ball_vy_n":        ball_vy        / BALL_SPEED_MAX,
        "left_center_y_n":  lc             / HEIGHT,
        "right_center_y_n": rc             / HEIGHT,
        "left_ball_dy_n":   (ball_y - lc)  / HEIGHT,
        "left_ball_dx_n":   (ball_x - 30)  / WIDTH,
        "right_ball_dy_n":  (ball_y - rc)  / HEIGHT,
        "right_ball_dx_n":  (758 - ball_x) / WIDTH,
    }
    return np.array([feats[f] for f in FEATURES], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Headless game engine  — no pygame dependency, fast enough for training
# ═══════════════════════════════════════════════════════════════════════════════
def _aabb(x1, y1, w1, h1, x2, y2, w2, h2):
    """Axis-aligned bounding-box overlap test."""
    return (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2)


class Paddle:
    __slots__ = ("x", "y")
    def __init__(self, x: float):
        self.x = x
        self.y = HEIGHT / 2 - PADDLE_H / 2

    def move(self, action: int):
        """action ∈ {-1, 0, 1}"""
        self.y = max(0, min(HEIGHT - PADDLE_H, self.y + action * PADDLE_SPEED))

    def centre_y(self) -> float:
        return self.y + PADDLE_H / 2


class Ball:
    __slots__ = ("x", "y", "vx", "vy", "speed")
    def __init__(self):
        self.reset()

    def reset(self, last_scorer: str | None = None):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.speed = BALL_SPEED_INIT
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = 1 if (last_scorer is None or last_scorer == "right") else -1
        self.vx = direction * self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.y - BALL_SIZE / 2 <= 0:
            self.y = BALL_SIZE / 2
            self.vy = abs(self.vy)
        if self.y + BALL_SIZE / 2 >= HEIGHT:
            self.y = HEIGHT - BALL_SIZE / 2
            self.vy = -abs(self.vy)

    def hit_paddle(self, paddle: Paddle):
        offset = (self.y - paddle.centre_y()) / (PADDLE_H / 2)
        offset = max(-1.0, min(1.0, offset))
        angle = offset * (math.pi / 3)
        self.speed = min(self.speed + BALL_SPEED_INCREMENT, BALL_SPEED_MAX)
        if self.vx < 0:
            self.vx =  self.speed * math.cos(angle)
        else:
            self.vx = -self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        if paddle.x < WIDTH / 2:
            self.x = paddle.x + PADDLE_W + BALL_SIZE / 2 + 1
        else:
            self.x = paddle.x - BALL_SIZE / 2 - 1


def ball_hits_paddle(ball: Ball, paddle: Paddle) -> bool:
    return _aabb(ball.x - BALL_SIZE/2, ball.y - BALL_SIZE/2,
                 BALL_SIZE, BALL_SIZE,
                 paddle.x, paddle.y, PADDLE_W, PADDLE_H)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy helpers
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def sample_action(model: nn.Module, features_np: np.ndarray,
                  device: torch.device, temperature: float = 1.0):
    """
    Draw an action from the categorical distribution π(·|s).

    temperature  controls randomness:
      · 1.0   →  standard sampling
      · < 1.0  →  more greedy  (exploit)
      · > 1.0  →  more uniform (explore)

    Returns (action, action_idx, log_prob_of_sampled_action).
      action     ∈  {-1, 0, 1}
      action_idx ∈  { 0, 1, 2}
    """
    x = torch.as_tensor(features_np, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model(x) / temperature
    dist = torch.distributions.Categorical(logits=logits)
    idx = dist.sample().item()
    return idx - 1, idx, dist.log_prob(torch.tensor(idx, device=device)).item()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-play episode
# ═══════════════════════════════════════════════════════════════════════════════
def run_selfplay_episode(model: nn.Module, device: torch.device,
                         temperature: float = 1.0):
    """
    One game of self-play  →  returns (transitions, score_left, score_right, frames).

    Both paddles are controlled by the *same* model.  We split the game into
    "mini-episodes": each time a point is scored we look at all the (state,
    action) pairs since the last point.

      · actions by the scoring paddle  →  +1 reward
      · actions by the conceding paddle →  -1 reward

    The returned list of transitions is the union of both paddles' experiences;
    each entry is (features_np, action_idx, reward).
    """
    left  = Paddle(30)
    right = Paddle(WIDTH - 30 - PADDLE_W)
    ball  = Ball()

    score_left = 0
    score_right = 0
    total_frames = 0
    MAX_FRAMES = 20_000

    all_transitions = []          # accumulation across the whole game

    while score_left < WINNING_SCORE and score_right < WINNING_SCORE:
        # ——— mini-episode (one point) ——————————————————————————————————————
        point_buf = []            # (features, left_action_idx, right_action_idx)
        while True:
            total_frames += 1
            if total_frames > MAX_FRAMES:
                return all_transitions, score_left, score_right, total_frames

            feats = build_features(ball.x, ball.y, ball.vx, ball.vy,
                                   left.y, right.y)

            la, la_idx, _ = sample_action(model, feats, device, temperature)
            ra, ra_idx, _ = sample_action(model, feats, device, temperature)

            point_buf.append((feats.copy(), la_idx, ra_idx))

            left.move(la)
            right.move(ra)
            ball.step()

            # Paddle collisions
            if ball.vx < 0 and ball_hits_paddle(ball, left):
                ball.hit_paddle(left)
            elif ball.vx > 0 and ball_hits_paddle(ball, right):
                ball.hit_paddle(right)

            # ——— scoring ———————————————————————————————————————————————
            if ball.x + BALL_SIZE / 2 < 0:            # right scores
                score_right += 1
                for (f, l_i, r_i) in point_buf:
                    all_transitions.append((f, r_i,  1.0))   # good paddle
                    all_transitions.append((f, l_i, -1.0))   # bad  paddle
                ball.reset(last_scorer="right")
                break

            if ball.x - BALL_SIZE / 2 > WIDTH:          # left scores
                score_left += 1
                for (f, l_i, r_i) in point_buf:
                    all_transitions.append((f, l_i,  1.0))
                    all_transitions.append((f, r_i, -1.0))
                ball.reset(last_scorer="left")
                break

    return all_transitions, score_left, score_right, total_frames


# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCE update
# ═══════════════════════════════════════════════════════════════════════════════
def reinforce_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                     transitions: list, device: torch.device,
                     baseline: float = 0.0):
    """
    Vanilla policy-gradient step on one batch of (s, a, r) tuples.

    loss =  - mean( log π(a|s) * (reward - baseline) )

    · (reward - baseline) is the *advantage* — how much better/worse this
      action was compared to our expectation.
    · Multiplying by log_prob pushes the probability UP for positive advantage
      and DOWN for negative advantage.
    """
    if not transitions:
        return 0.0, 0.0

    feats_b = np.stack([t[0] for t in transitions])          # (N, 10)
    acts_b  = np.array([t[1] for t in transitions], dtype=np.int64)
    rews_b  = np.array([t[2] for t in transitions], dtype=np.float32)

    feats_t = torch.as_tensor(feats_b, dtype=torch.float32, device=device)
    acts_t  = torch.as_tensor(acts_b,  dtype=torch.long,     device=device)
    rews_t  = torch.as_tensor(rews_b,  dtype=torch.float32,  device=device)

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
def evaluate(model: nn.Module, device: torch.device, n_games: int = 50):
    """
    Play *n_games* of self-play with argmax actions (no exploration).
    Reports average rally length and score distribution.
    """
    left_wins = 0
    right_wins = 0
    rally_lengths = []

    for _ in range(n_games):
        left  = Paddle(30)
        right = Paddle(WIDTH - 30 - PADDLE_W)
        ball  = Ball()
        s_l = s_r = 0
        total_frames = 0
        runs = 0
        while s_l < WINNING_SCORE and s_r < WINNING_SCORE and total_frames < 20_000:
            total_frames += 1
            runs += 1
            feats = build_features(ball.x, ball.y, ball.vx, ball.vy,
                                   left.y, right.y)
            # greedy actions
            _, la_idx, _ = sample_action(model, feats, device, temperature=0.01)
            _, ra_idx, _ = sample_action(model, feats, device, temperature=0.01)

            left.move(la_idx - 1)
            right.move(ra_idx - 1)
            ball.step()

            if ball.vx < 0 and ball_hits_paddle(ball, left):
                ball.hit_paddle(left)
            elif ball.vx > 0 and ball_hits_paddle(ball, right):
                ball.hit_paddle(right)

            if ball.x + BALL_SIZE / 2 < 0:
                s_r += 1
                rally_lengths.append(runs)
                runs = 0
                ball.reset(last_scorer="right")
            elif ball.x - BALL_SIZE / 2 > WIDTH:
                s_l += 1
                rally_lengths.append(runs)
                runs = 0
                ball.reset(last_scorer="left")

        if s_l >= WINNING_SCORE:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────────────
    model = PongMLP().to(device)
    bc_path = os.path.join(MODEL_DIR, "bc_model.pt")
    if os.path.exists(bc_path):
        model.load_state_dict(torch.load(bc_path, map_location=device))
        print(f"Loaded BC checkpoint  ←  {bc_path}")
    else:
        print("[WARN] No bc_model.pt found — starting from random weights.")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Running baseline (variance reduction) ────────────────────────────────
    baseline = 0.0
    bl_alpha = 0.05              # EMA smoothing factor

    # ── Trackers ─────────────────────────────────────────────────────────────
    recent_left  = deque(maxlen=100)
    recent_right = deque(maxlen=100)
    recent_loss  = deque(maxlen=100)

    print(f"\n{'='*60}")
    print(f"Self-play REINFORCE  |  device: {device}")
    print(f"Episodes: {args.episodes}  |  LR: {args.lr}  |  init temp: {args.temperature}")
    print(f"{'='*60}\n")

    for ep in range(1, args.episodes + 1):
        # Linearly anneal temperature: explore a lot early, exploit later
        progress = ep / args.episodes
        temp = max(0.05, args.temperature * (1.0 - progress))

        # ── 1. Collect experience ────────────────────────────────────────
        transitions, s_l, s_r, frames = run_selfplay_episode(
            model, device, temperature=temp
        )

        # ── 2. Policy-gradient update ────────────────────────────────────
        loss_val, mean_r = reinforce_update(
            model, optimizer, transitions, device, baseline
        )

        # ── 3. Track statistics ──────────────────────────────────────────
        if transitions:
            baseline = (1 - bl_alpha) * baseline + bl_alpha * mean_r

        recent_left.append(1 if s_l >= WINNING_SCORE else 0)
        recent_right.append(1 if s_r >= WINNING_SCORE else 0)
        recent_loss.append(loss_val)

        # ── 4. Logging ───────────────────────────────────────────────────
        if ep % args.log_every == 0 or ep == 1:
            avg_loss = np.mean(recent_loss) if recent_loss else 0.0
            lw = sum(recent_left)
            rw = sum(recent_right)
            n  = max(1, len(recent_left))
            print(
                f"ep {ep:>6}  │  loss {avg_loss:.4f}  │  bl {baseline:+.4f}"
                f"  │  L {lw}/{n} ({100*lw/n:3.0f}%)  R {rw}/{n} ({100*rw/n:3.0f}%)"
                f"  │  fr {frames}  │  T {temp:.3f}"
            )

        # ── 5. Evaluation ────────────────────────────────────────────────
        if ep % args.eval_every == 0:
            model.eval()
            lw_e, rw_e, rally = evaluate(model, device, n_games=50)
            model.train()
            print(
                f"  eval →  L {lw_e}/50  R {rw_e}/50"
                f"  |  avg rally {rally:.1f} frames"
            )

        # ── 6. Checkpoint ────────────────────────────────────────────────
        if ep % args.save_every == 0:
            ckpt = os.path.join(MODEL_DIR, f"rl_ep{ep}.pt")
            torch.save({
                "episode": ep,
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
    parser.add_argument("--episodes",  type=int,   default=5000,
                        help="Number of self-play games (default: 5000)")
    parser.add_argument("--lr",        type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial action softmax temperature (default: 1.0)")
    parser.add_argument("--log-every", type=int,   default=100,
                        help="Print stats every N episodes (default: 100)")
    parser.add_argument("--eval-every", type=int,  default=500,
                        help="Run 50-game deterministic eval every N episodes (default: 500)")
    parser.add_argument("--save-every", type=int,  default=1000,
                        help="Save checkpoint every N episodes (default: 1000)")
    main(parser.parse_args())
