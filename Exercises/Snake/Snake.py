from enum import Enum
import random
import numpy as np

class SnakeAction(Enum):
    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3

class GridTile(Enum):
    _FLOOR = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2
    FOOD = 3
    OBSTACLE = 4

    def __str__(self):
        return self.name[:1]
    
class Snake():
    def __init__(self, rows, cols, seed = None):
        self.rows = rows
        self.cols = cols
        self.snake_body_max = np.inf
        self.reset(seed)

    def reset(self, seed=None):
        # Initialize Snake's head starting position
        random.seed(seed)
        self.snake = [[
            random.randint(1, self.rows-2),
            random.randint(1, self.cols-2)
        ]]
        self.pos_prev = self.snake[-1].copy()
        self.body_dim = 0
        self.direction = SnakeAction(np.random.randint(0, 3)) #cannot ne 4 but 3
        self.score = 0

        # Initialize food starting position
        self.food_pos = [
            random.randint(1, self.rows-2),
            random.randint(1, self.cols-2)
        ]

        # Build obstacles
        self.obstacles = [[0, i] for i in range (self.cols)] + [[self.rows - 1, i] for i in range (self.cols)] + \
                         [[i, 0] for i in range (1, self.rows - 1)] + [[i, self.cols - 1] for i in range (1, self.rows - 1)]

    def perform_action(self, snake_action:SnakeAction) -> bool:
        self.pos_prev = self.snake[-1].copy()
        if self.body_dim > 0:
            self.snake[:-1][0] = self.snake[-1].copy()

        clock_wise = [SnakeAction.RIGHT, SnakeAction.DOWN, SnakeAction.LEFT, SnakeAction.UP]
        idx = clock_wise.index(self.direction)
        if snake_action == SnakeAction.UP: # straight
            new_dir = clock_wise[idx]  
        elif snake_action == SnakeAction.RIGHT: #right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  
        else:  #left turn 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  
        self.direction = new_dir

        new_position = self.snake[-1].copy()
        if self.direction == SnakeAction.LEFT:
            if self.snake[-1][1]>0:
                new_position[1]-=1
        elif self.direction == SnakeAction.RIGHT:
            if self.snake[-1][1]<self.cols-1:
                new_position[1]+=1
        elif self.direction == SnakeAction.UP:
            if self.snake[-1][0]>0:
                new_position[0]-=1
        elif self.direction == SnakeAction.DOWN:
            if self.snake[-1][0]<self.rows-1:
                new_position[0]+=1
        
        self.snake.append(new_position)

        food_found = False
        if self.snake[-1] == self.food_pos:
            food_found = True
            self.score += 1
            self.food_pos = [
                random.randint(1, self.rows-2),
                random.randint(1, self.cols-2)
            ]

        if not food_found: self.snake.pop(0)

        return food_found, self.snake[-1] in self.snake[:-1], self.snake[-1] in self.obstacles

    def build_state(self):

        up = [self.snake[-1][0] -1, self.snake[-1][1]]
        down = [self.snake[-1][0] + 1, self.snake[-1][1]]
        left = [self.snake[-1][0], self.snake[-1][1] - 1]
        right = [self.snake[-1][0], self.snake[-1][1] + 1]

        dir_l = self.direction == SnakeAction.LEFT
        dir_r = self.direction == SnakeAction.RIGHT
        dir_u = self.direction == SnakeAction.UP
        dir_d = self.direction == SnakeAction.DOWN

        collision_u = (up in self.snake[:-1]) | (up in self.obstacles)
        collision_d = (down in self.snake[:-1]) | (down in self.obstacles)
        collision_l = (left in self.snake[:-1]) | (left in self.obstacles)
        collision_r = (right in self.snake[:-1]) | (right in self.obstacles)

        danger_straight =   (dir_r and collision_r) or \
                            (dir_l and collision_l) or \
                            (dir_u and collision_u) or \
                            (dir_d and collision_d)
        
        danger_right =      (dir_r and collision_d) or \
                            (dir_l and collision_u) or \
                            (dir_u and collision_r) or \
                            (dir_d and collision_l)
        
        danger_left =       (dir_r and collision_u) or \
                            (dir_l and collision_d) or \
                            (dir_u and collision_l) or \
                            (dir_d and collision_r)

        food_up = self.food_pos[0] < self.snake[-1][0]
        food_down = self.food_pos[0] > self.snake[-1][0]
        food_right = self.food_pos[1] > self.snake[-1][1]
        food_left = self.food_pos[1] < self.snake[-1][1]

        state = np.array([danger_straight, danger_right, danger_left, dir_l, dir_r, dir_u, dir_d, food_up, food_down, food_right, food_left], dtype = np.int_)
        state_str = ""
        for i in state: state_str += str(i)
        return int(state_str, 2)

    def render(self):
        print("\033c")
        for r in range(self.rows):
            for c in range(self.cols):

                if([r,c] == self.snake[-1]):
                    print(GridTile.SNAKE_HEAD, end=' ')
                elif([r,c] == self.food_pos):
                    print(GridTile.FOOD, end=' ')
                elif([r,c] in self.snake[:-1]):
                    print('+', end=' ')
                elif([r,c] in self.obstacles):
                    print(GridTile.OBSTACLE, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print()
        print()

