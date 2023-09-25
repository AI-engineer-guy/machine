import pygame
import math
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CAR_WIDTH, CAR_HEIGHT = 40, 20
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car")

# Initialize car position and angle
car_x, car_y = WIDTH // 2, HEIGHT - 100
car_angle = 0

# Delay and target
delay = 1000  # Delay in milliseconds
last_time = pygame.time.get_ticks()
target_x, target_y = random.randint(0, WIDTH), random.randint(0, HEIGHT)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate elapsed time
    current_time = pygame.time.get_ticks()
    elapsed_time = current_time - last_time

    # Only update the target position if the delay has passed
    if elapsed_time >= delay:
        target_x, target_y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        last_time = current_time

    # Calculate angle to target
    angle_to_target = math.atan2(target_y - car_y, target_x - car_x)
    car_angle = angle_to_target

    # Move the car
    speed = 2
    car_x += speed * math.cos(car_angle)
    car_y += speed * math.sin(car_angle)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the car
    car_rect = pygame.Rect(int(car_x - CAR_WIDTH / 2), int(car_y - CAR_HEIGHT / 2), CAR_WIDTH, CAR_HEIGHT)
    pygame.draw.rect(screen, GREEN, car_rect)

    # Round the target coordinates to integers
    target_x, target_y = int(target_x), int(target_y)
    pygame.draw.line(screen, RED, (int(car_x), int(car_y)), (target_x, target_y), 2)

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
