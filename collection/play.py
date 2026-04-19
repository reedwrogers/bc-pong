"""
collection/play.py

Two-player Pong with automatic gameplay logging.
  Left paddle  : W (up) / S (down)
  Right paddle : UP arrow / DOWN arrow

Each frame logs:
  ball_x, ball_y, ball_vx, ball_vy,
  left_y, right_y,
  left_action, right_action,   # -1 up | 0 stay | 1 down
  left_scored, right_scored    # 1 on the frame a point is awarded, else 0

Output: data/session_<timestamp>.csv
"""

import pygame
import csv
import os
import time
import math
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
PADDLE_W, PADDLE_H = 12, 80
BALL_SIZE = 12
FPS = 60

PADDLE_SPEED = 5
BALL_SPEED_INIT = 5.0
BALL_SPEED_MAX = 14.0
BALL_SPEED_INCREMENT = 0.3   # added to speed after each paddle hit

WINNING_SCORE = 5            # first to this wins the session

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Colors
BLACK  = (10,  10,  10)
WHITE  = (240, 240, 240)
GREY   = (120, 120, 120)
GREEN  = ( 80, 200, 120)
RED    = (220,  80,  80)

# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------
class Paddle:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_H // 2
        self.action = 0   # -1 | 0 | 1

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, PADDLE_W, PADDLE_H)

    def move(self, action):
        """action: -1 up, 0 stay, 1 down"""
        self.action = action
        self.y += action * PADDLE_SPEED
        self.y = max(0, min(HEIGHT - PADDLE_H, self.y))

    def center_y(self):
        return self.y + PADDLE_H // 2


class Ball:
    def __init__(self):
        self.reset()

    def reset(self, last_scorer=None):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.speed = BALL_SPEED_INIT
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = 1 if (last_scorer is None or last_scorer == "right") else -1
        self.vx = direction * self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)

    @property
    def rect(self):
        return pygame.Rect(int(self.x - BALL_SIZE/2), int(self.y - BALL_SIZE/2),
                           BALL_SIZE, BALL_SIZE)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        # wall bounce (top / bottom)
        if self.y - BALL_SIZE/2 <= 0:
            self.y = BALL_SIZE/2
            self.vy = abs(self.vy)
        if self.y + BALL_SIZE/2 >= HEIGHT:
            self.y = HEIGHT - BALL_SIZE/2
            self.vy = -abs(self.vy)

    def hit_paddle(self, paddle):
        """Deflect ball off paddle, speed up slightly."""
        offset = (self.y - paddle.center_y()) / (PADDLE_H / 2)
        offset = max(-1.0, min(1.0, offset))
        angle = offset * (math.pi / 3)   # max 60° deflection
        self.speed = min(self.speed + BALL_SPEED_INCREMENT, BALL_SPEED_MAX)
        if self.vx < 0:                  # came from right, now go right
            self.vx =  self.speed * math.cos(angle)
        else:
            self.vx = -self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        # push ball out of paddle so it doesn't double-collide
        if paddle.x < WIDTH / 2:
            self.x = paddle.x + PADDLE_W + BALL_SIZE / 2 + 1
        else:
            self.x = paddle.x - BALL_SIZE / 2 - 1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("bc-pong  |  W/S  vs  ↑/↓  |  ESC to quit")
    clock = pygame.time.Clock()
    font_big  = pygame.font.SysFont("monospace", 48, bold=True)
    font_small = pygame.font.SysFont("monospace", 22)

    left  = Paddle(30)
    right = Paddle(WIDTH - 30 - PADDLE_W)
    ball  = Ball()

    score_left  = 0
    score_right = 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(DATA_DIR, f"session_{timestamp}.csv")
    fieldnames = [
        "ball_x", "ball_y", "ball_vx", "ball_vy",
        "left_y", "right_y",
        "left_action", "right_action",
        "left_scored", "right_scored",
    ]

    frames_logged = 0
    running = True

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while running:
            # ----------------------------------------------------------------
            # Input
            # ----------------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()

            # Left paddle (W/S)
            if keys[pygame.K_w] and not keys[pygame.K_s]:
                left_action = -1
            elif keys[pygame.K_s] and not keys[pygame.K_w]:
                left_action = 1
            else:
                left_action = 0

            # Right paddle (arrows)
            if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                right_action = -1
            elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                right_action = 1
            else:
                right_action = 0

            left.move(left_action)
            right.move(right_action)

            # ----------------------------------------------------------------
            # Physics
            # ----------------------------------------------------------------
            ball.step()

            # Paddle collisions
            if ball.vx < 0 and ball.rect.colliderect(left.rect):
                ball.hit_paddle(left)
            if ball.vx > 0 and ball.rect.colliderect(right.rect):
                ball.hit_paddle(right)

            # Scoring
            left_scored_frame  = 0
            right_scored_frame = 0

            if ball.x + BALL_SIZE / 2 < 0:        # right scores
                score_right += 1
                right_scored_frame = 1
                ball.reset(last_scorer="right")
                pygame.time.delay(500)

            elif ball.x - BALL_SIZE / 2 > WIDTH:   # left scores
                score_left += 1
                left_scored_frame = 1
                ball.reset(last_scorer="left")
                pygame.time.delay(500)

            # ----------------------------------------------------------------
            # Log
            # ----------------------------------------------------------------
            writer.writerow({
                "ball_x":       round(ball.x, 3),
                "ball_y":       round(ball.y, 3),
                "ball_vx":      round(ball.vx, 3),
                "ball_vy":      round(ball.vy, 3),
                "left_y":       left.y,
                "right_y":      right.y,
                "left_action":  left_action,
                "right_action": right_action,
                "left_scored":  left_scored_frame,
                "right_scored": right_scored_frame,
            })
            frames_logged += 1

            # ----------------------------------------------------------------
            # Draw
            # ----------------------------------------------------------------
            screen.fill(BLACK)

            # Center line
            for y in range(0, HEIGHT, 20):
                pygame.draw.rect(screen, GREY, (WIDTH//2 - 1, y, 2, 10))

            # Paddles
            pygame.draw.rect(screen, GREEN, left.rect)
            pygame.draw.rect(screen, RED,   right.rect)

            # Ball
            pygame.draw.ellipse(screen, WHITE, ball.rect)

            # Scores
            left_surf  = font_big.render(str(score_left),  True, GREEN)
            right_surf = font_big.render(str(score_right), True, RED)
            screen.blit(left_surf,  (WIDTH//2 - 80, 20))
            screen.blit(right_surf, (WIDTH//2 + 40, 20))

            # HUD
            hud = font_small.render(
                f"frames logged: {frames_logged}  |  ESC to quit",
                True, GREY
            )
            screen.blit(hud, (10, HEIGHT - 30))

            pygame.display.flip()

            # ----------------------------------------------------------------
            # Win check
            # ----------------------------------------------------------------
            if score_left >= WINNING_SCORE or score_right >= WINNING_SCORE:
                winner = "LEFT (W/S)" if score_left >= WINNING_SCORE else "RIGHT (↑/↓)"
                win_surf = font_big.render(f"{winner} WINS!", True, WHITE)
                screen.blit(win_surf, win_surf.get_rect(center=(WIDTH//2, HEIGHT//2)))
                pygame.display.flip()
                pygame.time.delay(2500)
                # Reset for another game
                score_left = score_right = 0
                ball.reset()

            clock.tick(FPS)

    pygame.quit()
    print(f"\nSession saved → {csv_path}")
    print(f"Total frames logged: {frames_logged}")


if __name__ == "__main__":
    main()
