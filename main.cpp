#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#define WIDTH  10
#define HEIGHT 10

#define SNAKE 2

int grid[HEIGHT][WIDTH];

/*
 * Direction
 * UP:
 * DOWN:
 * LEFT:
 * RIGHT:
 */
// int dy[4] = {-1, 1, 0, 0};
// int dx[4] = {0, 0, -1, 1};

struct Vec2 {
    int y;
    int x;

    Vec2 operator+(const Vec2& other) const {
        return {y + other.y, x + other.x};
    }

    bool operator==(const Vec2& other) const {
        return y == other.y && x == other.x;
    }
};

constexpr Vec2 UP    = {-1,  0};
constexpr Vec2 DOWN  = { 1,  0};
constexpr Vec2 LEFT  = { 0, -1};
constexpr Vec2 RIGHT = { 0,  1};


typedef struct
{
    int head_x;
    int head_y;
    Vec2 direction = RIGHT;
    int size;

    std::deque<Vec2> body;
} Snake;


void init(Snake& snake, std::pair<int, int> init_pos)
{
    snake.head_y    = init_pos.first;
    snake.head_x    = init_pos.second;
    snake.direction = RIGHT;

    snake.size = 1;

    snake.body.push_front({snake.head_y, snake.head_x});
}

void move(Snake& snake, int grid[][WIDTH]) {
    Vec2 next = snake.body.front() + snake.direction;

    snake.body.push_front(next);
    snake.body.pop_back();
}

void eat(Snake& snake, int grid[][WIDTH])
{
    Vec2 next = snake.body.front() + snake.direction;

    snake.body.push_front(next);
}

void update(Snake& snake, int grid[][WIDTH])
{
    std::cout << snake.body.empty() << '\n';

    while (!snake.body.empty()) {
        Vec2 pos = snake.body.front();
        snake.body.pop_front();

        std::cout << pos.y << ' ' << pos.x << '\n';
        grid[pos.y][pos.x] = SNAKE;
    }
}

void display(Snake& snake, int grid[][WIDTH])
{
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            std::cout << grid[i][j] << ' ';
        }
        std::cout << '\n';
    }
}

int main(void)
{
    Snake snake;
    init(snake, {1, 1});

    eat(snake, grid);
    eat(snake, grid);
    move(snake, grid);
    move(snake, grid);
    move(snake, grid);
    update(snake, grid);
    display(snake, grid);

    return 0;
}