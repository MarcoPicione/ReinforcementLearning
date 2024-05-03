#!/usr/bin/env python
import exercises.snake.snake as s
import random
import time
from pynput import keyboard
import pygame

global listener

def map_action():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Movimento del rettangolo
                if event.key == pygame.K_LEFT:
                    return 1
                elif event.key == pygame.K_RIGHT:
                    return 2
                elif event.key == pygame.K_UP:
                    return 0

def main():
    snake = s.Snake(12, 12, seed = 123)
    pygame.init()
    win = pygame.display.set_mode((500,500))

    while(True):
        snake.render()
        time.sleep(0.1)
        print(snake.build_state())
        snake.perform_action(s.SnakeAction(map_action()))
        

if __name__== '__main__':
    main()