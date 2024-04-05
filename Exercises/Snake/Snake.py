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

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        # if(self.value == 1 or self.value == 2):
        #     return self.name[:7]
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
        self.snake_head_pos = [
            random.randint(1, self.rows-2),
            random.randint(1, self.cols-2)
        ]
        self.snake_head_pos_prev = [0, 0]
        self.snake_body = []
        self.body_dim = 0
        self.direction = SnakeAction(np.random.randint(0,4))
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
        self.snake_head_pos_prev = self.snake_head_pos.copy()
        if self.body_dim > 1:
            snake_body_reversed = self.snake_body.copy()
            snake_body_reversed.reverse()
            for i in range(len(snake_body_reversed)-1):
                snake_body_reversed[i] = snake_body_reversed[i + 1]
            snake_body_reversed.reverse()
            self.snake_body = snake_body_reversed.copy()
        
        if self.body_dim > 0:
            self.snake_body[0] = self.snake_head_pos.copy()

        clock_wise = [SnakeAction.RIGHT, SnakeAction.DOWN, SnakeAction.LEFT, SnakeAction.UP]
        idx = clock_wise.index(self.direction)
        if snake_action == SnakeAction.UP: # straight
            new_dir = clock_wise[idx]  
        elif snake_action == SnakeAction.RIGHT: #right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  
        else:  #[0,0,1] aka left turn 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  
        self.direction = new_dir

        if self.direction == SnakeAction.LEFT:
            if self.snake_head_pos[1]>0:
                self.snake_head_pos[1]-=1
        elif self.direction == SnakeAction.RIGHT:
            if self.snake_head_pos[1]<self.cols-1:
                self.snake_head_pos[1]+=1
        elif self.direction == SnakeAction.UP:
            if self.snake_head_pos[0]>0:
                self.snake_head_pos[0]-=1
        elif self.direction == SnakeAction.DOWN:
            if self.snake_head_pos[0]<self.rows-1:
                self.snake_head_pos[0]+=1

        food_found = False
        if self.snake_head_pos == self.food_pos:
            food_found = True
            self.score += 1
            self.food_pos = [
                random.randint(1, self.rows-2),
                random.randint(1, self.cols-2)
            ]
            if(self.body_dim < self.snake_body_max):
                self.add_body()

        return food_found, self.snake_head_pos in self.snake_body, self.snake_head_pos in self.obstacles

    def add_body(self):
        if self.body_dim == 0:
            self.snake_body = [[self.snake_head_pos[0], self.snake_head_pos[1] + 1]]
        else:
            self.snake_body.append([self.snake_body[-1][0], self.snake_body[-1][1] + 1])
        self.body_dim += 1

    def build_state(self):
        up = [self.snake_head_pos[0], self.snake_head_pos[1] - 1]
        down = [self.snake_head_pos[0], self.snake_head_pos[1] + 1]
        left = [self.snake_head_pos[0] - 1, self.snake_head_pos[1]]
        right = [self.snake_head_pos[0] + 1, self.snake_head_pos[1]]

        dir_l = self.direction == SnakeAction.LEFT
        dir_r = self.direction == SnakeAction.RIGHT
        dir_u = self.direction == SnakeAction.UP
        dir_d = self.direction == SnakeAction.DOWN

        collision_s = (up in self.snake_body) | (up in self.obstacles)
        collision_d = (down in self.snake_body) | (down in self.obstacles)
        collision_l = (left in self.snake_body) | (left in self.obstacles)
        collision_r = (right in self.snake_body) | (right in self.obstacles)

        danger_straight =   (dir_r and collision_r) or \
                            (dir_l and collision_l) or \
                            (dir_u and collision_s) or \
                            (dir_d and collision_d)
        
        danger_right =      (dir_r and collision_d) or \
                            (dir_l and collision_s) or \
                            (dir_u and collision_r) or \
                            (dir_d and collision_l)
        
        danger_left =       (dir_r and collision_s) or \
                            (dir_l and collision_d) or \
                            (dir_u and collision_l) or \
                            (dir_d and collision_r)

        food_up = self.food_pos[0] < self.snake_head_pos[0]
        food_down = self.food_pos[0] > self.snake_head_pos[0]
        food_right = self.food_pos[1] > self.snake_head_pos[1]
        food_left = self.food_pos[1] < self.snake_head_pos[1]

        return np.array([danger_straight, danger_right, danger_left, dir_l, dir_r, dir_u, dir_d, food_up, food_down, food_right, food_left])

    def render(self):
        # Print current state on console
        print("\033c")
        for r in range(self.rows):
            for c in range(self.cols):

                if([r,c] == self.snake_head_pos):
                    print(GridTile.SNAKE_HEAD, end=' ')
                elif([r,c] == self.food_pos):
                    print(GridTile.FOOD, end=' ')
                elif([r,c] in self.snake_body):
                    print('+', end=' ')
                elif([r,c] in self.obstacles):
                    print(GridTile.OBSTACLE, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print() # new line
        print() # new line

