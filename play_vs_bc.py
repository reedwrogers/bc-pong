"""
play_vs_bc.py

Human (right paddle, arrow keys) vs a trained model (left paddle).

Usage:
    python play_vs_bc.py                              # default: BC model
    python play_vs_bc.py --model models/rl_model.pt    # RL-trained model
"""

import argparse
import pygame
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os

# ---------------------------------------------------------------------------
# Must match bc_train.ipynb exactly
# ---------------------------------------------------------------------------
WIDTH, HEIGHT  = 800, 600
PADDLE_W       = 12
PADDLE_H       = 80
BALL_SIZE      = 12
FPS            = 60
PADDLE_SPEED   = 5
BALL_SPEED_INIT = 5.0
BALL_SPEED_MAX  = 14.0
BALL_SPEED_INCREMENT = 0.3
WINNING_SCORE  = 5

FEATURES = [
    "ball_x_n", "ball_y_n", "ball_vx_n", "ball_vy_n",
    "left_center_y_n", "right_center_y_n",
    "left_ball_dy_n", "left_ball_dx_n",
    "right_ball_dy_n", "right_ball_dx_n",
]

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "models", "bc_model.pt")

# ---------------------------------------------------------------------------
# Model (must match training definition)
# ---------------------------------------------------------------------------
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


def load_model(path):
    model = PongMLP()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"Loaded model from {path}")
    return model


# ---------------------------------------------------------------------------
# Feature builder — matches engineer_features in the notebook
# ---------------------------------------------------------------------------
def build_features(ball_x, ball_y, ball_vx, ball_vy, left_y, right_y):
    left_center_y  = left_y  + PADDLE_H / 2
    right_center_y = right_y + PADDLE_H / 2

    feats = {
        "ball_x_n":         ball_x         / WIDTH,
        "ball_y_n":         ball_y         / HEIGHT,
        "ball_vx_n":        ball_vx        / BALL_SPEED_MAX,
        "ball_vy_n":        ball_vy        / BALL_SPEED_MAX,
        "left_center_y_n":  left_center_y  / HEIGHT,
        "right_center_y_n": right_center_y / HEIGHT,
        "left_ball_dy_n":   (ball_y - left_center_y)  / HEIGHT,
        "left_ball_dx_n":   (ball_x - 30)             / WIDTH,
        "right_ball_dy_n":  (ball_y - right_center_y) / HEIGHT,
        "right_ball_dx_n":  (758 - ball_x)            / WIDTH,
    }
    return np.array([feats[f] for f in FEATURES], dtype=np.float32)


def model_action(model, features):
    """Returns -1 (up), 0 (stay), or 1 (down)."""
    with torch.no_grad():
        x      = torch.tensor(features).unsqueeze(0)
        logits = model(x)
        pred   = logits.argmax(1).item()   # 0, 1, 2
    return pred - 1                        # remap to -1, 0, 1


# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------
class Paddle:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_H // 2

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, PADDLE_W, PADDLE_H)

    def move(self, action):
        self.y += action * PADDLE_SPEED
        self.y  = max(0, min(HEIGHT - PADDLE_H, self.y))

    def center_y(self):
        return self.y + PADDLE_H // 2


class Ball:
    def __init__(self):
        self.reset()

    def reset(self, last_scorer=None):
        self.x     = WIDTH / 2
        self.y     = HEIGHT / 2
        self.speed = BALL_SPEED_INIT
        angle      = random.uniform(-math.pi / 4, math.pi / 4)
        direction  = 1 if (last_scorer is None or last_scorer == "right") else -1
        self.vx    = direction * self.speed * math.cos(angle)
        self.vy    = self.speed * math.sin(angle)

    @property
    def rect(self):
        return pygame.Rect(int(self.x - BALL_SIZE/2), int(self.y - BALL_SIZE/2),
                           BALL_SIZE, BALL_SIZE)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.y - BALL_SIZE/2 <= 0:
            self.y  = BALL_SIZE/2
            self.vy = abs(self.vy)
        if self.y + BALL_SIZE/2 >= HEIGHT:
            self.y  = HEIGHT - BALL_SIZE/2
            self.vy = -abs(self.vy)

    def hit_paddle(self, paddle):
        offset    = (self.y - paddle.center_y()) / (PADDLE_H / 2)
        offset    = max(-1.0, min(1.0, offset))
        angle     = offset * (math.pi / 3)
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


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BLACK  = ( 10,  10,  10)
WHITE  = (240, 240, 240)
GREY   = (120, 120, 120)
GREEN  = ( 80, 200, 120)   # AI  (left)
BLUE   = ( 80, 140, 220)   # You (right)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(model_path):
    model = load_model(model_path)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("bc-pong  |  YOU (↑/↓) vs BC Model  |  ESC to quit")
    clock      = pygame.time.Clock()
    font_big   = pygame.font.SysFont("monospace", 48, bold=True)
    font_small = pygame.font.SysFont("monospace", 20)

    left  = Paddle(30)                      # AI
    right = Paddle(WIDTH - 30 - PADDLE_W)   # Human
    ball  = Ball()

    score_ai    = 0
    score_human = 0
    running     = True

    while running:
        # --------------------------------------------------------------------
        # Input
        # --------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            human_action = -1
        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            human_action = 1
        else:
            human_action = 0

        # --------------------------------------------------------------------
        # AI action
        # --------------------------------------------------------------------
        features   = build_features(ball.x, ball.y, ball.vx, ball.vy,
                                     left.y, right.y)
        ai_action  = model_action(model, features)

        left.move(ai_action)
        right.move(human_action)

        # --------------------------------------------------------------------
        # Physics
        # --------------------------------------------------------------------
        ball.step()

        if ball.vx < 0 and ball.rect.colliderect(left.rect):
            ball.hit_paddle(left)
        if ball.vx > 0 and ball.rect.colliderect(right.rect):
            ball.hit_paddle(right)

        # --------------------------------------------------------------------
        # Scoring
        # --------------------------------------------------------------------
        if ball.x + BALL_SIZE / 2 < 0:
            score_human += 1
            ball.reset(last_scorer="right")
            pygame.time.delay(500)
        elif ball.x - BALL_SIZE / 2 > WIDTH:
            score_ai += 1
            ball.reset(last_scorer="left")
            pygame.time.delay(500)

        # --------------------------------------------------------------------
        # Draw
        # --------------------------------------------------------------------
        screen.fill(BLACK)

        for y in range(0, HEIGHT, 20):
            pygame.draw.rect(screen, GREY, (WIDTH//2 - 1, y, 2, 10))

        pygame.draw.rect(screen, GREEN, left.rect)   # AI
        pygame.draw.rect(screen, BLUE,  right.rect)  # Human
        pygame.draw.ellipse(screen, WHITE, ball.rect)

        # Scores
        ai_surf  = font_big.render(str(score_ai),    True, GREEN)
        you_surf = font_big.render(str(score_human), True, BLUE)
        screen.blit(ai_surf,  (WIDTH//2 - 80, 20))
        screen.blit(you_surf, (WIDTH//2 + 40, 20))

        # Labels
        screen.blit(font_small.render("BC MODEL", True, GREEN), (20,  HEIGHT - 28))
        screen.blit(font_small.render("YOU",      True, BLUE),  (WIDTH - 70, HEIGHT - 28))

        pygame.display.flip()

        # --------------------------------------------------------------------
        # Win check
        # --------------------------------------------------------------------
        if score_ai >= WINNING_SCORE or score_human >= WINNING_SCORE:
            winner     = "BC MODEL" if score_ai >= WINNING_SCORE else "YOU"
            win_color  = GREEN if score_ai >= WINNING_SCORE else BLUE
            win_surf   = font_big.render(f"{winner} WINS!", True, win_color)
            screen.blit(win_surf, win_surf.get_rect(center=(WIDTH//2, HEIGHT//2)))
            pygame.display.flip()
            pygame.time.delay(3000)
            score_ai = score_human = 0
            ball.reset()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Pong against a trained model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to model checkpoint (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(args.model)
