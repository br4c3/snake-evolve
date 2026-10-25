#include <iostream>
#include <vector>
#include <queue>
#include <utility>

#define WIDTH 10
#define HEIGHT 10

#define SNAKE 2


int grid[HEIGHT][WIDTH];


int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};


typedef struct {
    int head_x;
    int head_y;
    int direction;
    std::deque<std::pair<int, int>> body;
} Snake;

void init(Snake snake, std::pair<int, int> init_pos)
{
    snake.head_y = init_pos.first;
    snake.head_x = init_pos.second;
    snake.direction = 0;

    snake.body.push_front({snake.head_y, snake.head_x});
}

void eat(Snake snake, int grid[][WIDTH])
{
    int next_y = snake.head_y + dy[snake.direction];
    int next_x = snake.head_x + dx[snake.direction];

    snake.body.push_front({next_y, next_x});
    // snake.body.pop_back();
}

void update(Snake snake, int grid[][WIDTH])
{
    std::cout << snake.body.empty() << '\n';

    while (!snake.body.empty()) {
        std::pair<int, int> pos = snake.body.front();
        snake.body.pop_front();

        std::cout << pos.first << ' ' << pos.second << '\n';
        grid[pos.first][pos.second] = SNAKE;
    }
}

void display(Snake snake, int grid[][WIDTH])
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
    // eat(snake, grid);

    update(snake, grid);
    display(snake, grid);

    return 0;
}