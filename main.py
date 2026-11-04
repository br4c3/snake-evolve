from dataclasses import dataclass
from collections import deque
from enum import Enum
import random
from pynput import keyboard
import time


@dataclass
class Vec2:
    x: int
    y: int
    
    def __add__(self, rhs):
        return Vec2(self.x + rhs.x, self.y + rhs.y)


class Direction(Enum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


class CellType(Enum):
    EMPTY      = " "
    WALL       = "#"
    SNAKE_HEAD = "o"
    SNAKE_BODY = "+"
    FOOD       = "&"


class Snake:
    def __init__(self, pos: Vec2):
        self.body = deque([pos])
        self.direction = random.choice(list(Direction))

    def contains(self, pos: Vec2) -> bool:
        return pos in self.body

    def get_head_pos(self) -> Vec2:
        return self.body[0]

    def get_next_head_pos(self) -> Vec2:
        return self.body[0] + self.get_direction(self.direction)

    def get_direction(self, direction: Direction) -> Vec2:
        match direction:
            case Direction.UP:
                return Vec2(0, -1)
            case Direction.DOWN:
                return Vec2(0, 1)
            case Direction.LEFT:
                return Vec2(-1, 0)
            case Direction.RIGHT:
                return Vec2(1, 0)

    def set_direction(self, direction: Direction):
        opposite = {
            Direction.UP:    Direction.DOWN,
            Direction.DOWN:  Direction.UP,
            Direction.LEFT:  Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }

        if direction != opposite[self.direction]:
            self.direction = direction

    def move(self):
        print(f"Move: {self.direction}")
        next_pos = self.get_next_head_pos()

        self.body.appendleft(next_pos)
        self.body.pop()
        print(next_pos)

    def add_tail(self):
        next_pos = self.get_next_head_pos()
        
        self.body.appendleft(next_pos)

    def check_collision(self, pos: Vec2, growing: bool = False) -> bool:
        segments = self.body if growing else list(self.body)[:-1]
        return pos in segments


class Game:
    def __init__(self, width: int, height: int):
        if width < 3 or height < 3:
            raise ValueError("grid must be at least 3x3")

        self.width  = width
        self.height = height

        self.rng = random.Random()

        self.reset()

    """
    Initialization
    """
    def init_walls(self):
        for y in range(self.height):
            self.grid[y][0] = CellType.WALL
            self.grid[y][self.width - 1] = CellType.WALL

        for x in range(self.width):
            self.grid[0][x] = CellType.WALL
            self.grid[self.height - 1][x] = CellType.WALL

    def place_food(self):
        empty = [
            Vec2(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y][x] == CellType.EMPTY and not self.snake.contains(Vec2(x, y))
        ]

        if not empty:
            return

        pos = self.rng.choice(empty)
        self.grid[pos.y][pos.x] = CellType.FOOD

    def reset(self):
        self.running = True
        self.grid = [
            [CellType.EMPTY for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self.snake = Snake(Vec2(self.width // 2, self.height // 2))

        self.init_walls()
        self.place_food()

    """
    Update
    """
    def update(self):
        next_pos = self.snake.get_next_head_pos()
        
        # Check boundaries
        if not (0 <= next_pos.x < self.width and
                0 <= next_pos.y < self.height):
            self.running = False
            return

        cell = self.grid[next_pos.y][next_pos.x]
        
        # Check wall collision
        if cell == CellType.WALL:
            self.running = False
            return
        
        growing = cell == CellType.FOOD
        
        # Check self-collision
        if self.snake.check_collision(next_pos, growing):
            self.running = False
            return
        

        if growing:
            self.snake.add_tail()
            self.grid[next_pos.y][next_pos.x] = CellType.EMPTY
            self.place_food()
        else:
            self.snake.move()

    def display(self):
        print("\033[H", end="")

        for y in range(self.height):
            for x in range(self.width):
                pos = Vec2(x, y)

                if pos == self.snake.get_head_pos():
                    print(CellType.SNAKE_HEAD.value, end="")
                elif self.snake.contains(pos):
                    print(CellType.SNAKE_BODY.value, end="")
                else:
                    print(self.grid[y][x].value, end="")

            print()

        print(flush=True)

    """
    Keyboard Interrupt
    """
    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False
        
        try:
            match key.char.lower():
                # Control keys
                case 'w':
                    self.snake.set_direction(Direction.UP)
                case 's':
                    self.snake.set_direction(Direction.DOWN)
                case 'a':
                    self.snake.set_direction(Direction.LEFT)
                case 'd':
                    self.snake.set_direction(Direction.RIGHT)
                # Reset key
                case 'r':
                    self.reset()

        except (AttributeError, TypeError):
            pass


if __name__ == "__main__":
    game = Game(20, 10)

    listener = keyboard.Listener(on_press=game.on_press)
    listener.start()

    print("\033[2J", end="")

    try:
        while game.running:
            game.display()
            game.update()
            time.sleep(0.3)

    except KeyboardInterrupt:
        pass

    finally:
        listener.stop()
        print("\033[?25h")
