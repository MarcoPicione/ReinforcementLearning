from enum import Enum
import random
import numpy as np

class VacuumCleanerAction(Enum):
    STAND = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class GridTile(Enum):
    _FLOOR = 0
    VACUUM = 1
    SNAKE_BODY = 2
    FOOD = 3
    OBSTACLE = 4

    def __str__(self):
        return self.name[:1]
    
class Vacuum_cleaner():
    def __init__(self, name, bounds, seed = None):
        self.name = name
        self.x_min = bounds['x_min']
        self.x_max = bounds['x_max']
        self.y_min = bounds['y_min']
        self.y_max = bounds['y_max']
        self.seed = seed
        self.reset(seed)

    def reset(self, seed:int = None):
        random.seed(seed)

        # Initialize Vacuum Claener position
        x = random.randint(self.x_min, self.x_max)
        y = random.randint(self.y_min, self.y_max)
        self.position = [x, y]
        self.visited_cells = set(((x, y)))

        self.reward = 0

    def perform_action(self, vc_action:VacuumCleanerAction) -> bool:
        
        if vc_action == VacuumCleanerAction.STAND:
            pass
        if vc_action == VacuumCleanerAction.UP:
            self.position[0] -= 1
        elif vc_action == VacuumCleanerAction.RIGHT:
            self.position[1] += 1
        elif vc_action == VacuumCleanerAction.DOWN:
            self.position[0] += 1
        elif vc_action == VacuumCleanerAction.LEFT:
            self.position[1] -= 1
        else:
            raise Exception("Invalid action")
        
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
        print("Score: ", self.score)

    def __str__(self):
            return self.name

