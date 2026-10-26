#include <algorithm>
#include <deque>
#include <iostream>

#define WIDTH  10
#define HEIGHT 10

#define EMPTY 0
#define SNAKE 2
#define FOOD  3

struct Vec2 {
    int y;
    int x;

    Vec2 operator+(const Vec2& other) const
    {
        return {y + other.y, x + other.x};
    }

    bool operator==(const Vec2& other) const
    {
        return y == other.y && x == other.x;
    }
};

constexpr Vec2 UP    = {-1, 0};
constexpr Vec2 DOWN  = {1, 0};
constexpr Vec2 LEFT  = {0, -1};
constexpr Vec2 RIGHT = {0, 1};

struct Snake {
    Vec2             direction = RIGHT;
    std::deque<Vec2> body;
};

struct Game {
    Snake snake;
    Vec2  food;
    bool  running = true;
};

/*
 * Initialize
 */

void init(Game& game, Vec2 init_pos)
{
    game.snake.direction = RIGHT;
    game.snake.body.clear();
    game.snake.body.push_front(init_pos);

    game.food    = {3, 3};
    game.running = true;
}

/*
 * Event control
 */

bool is_opposite(const Vec2& a, const Vec2& b)
{
    return a.y + b.y == 0 && a.x + b.x == 0;
}

void change_direction(Snake& snake, const Vec2& direction)
{
    if (is_opposite(snake.direction, direction)) {
        return;
    }

    snake.direction = direction;
}

void handle_input(Game& game, char key)
{
    switch (key) {
    case 'w':
        change_direction(game.snake, UP);
        break;

    case 's':
        change_direction(game.snake, DOWN);
        break;

    case 'a':
        change_direction(game.snake, LEFT);
        break;

    case 'd':
        change_direction(game.snake, RIGHT);
        break;

    case 'q':
        game.running = false;
        break;
    }
}

/*
 * Game logic
 */

bool is_inside(const Vec2& pos)
{
    return pos.y >= 0 && pos.y < HEIGHT && pos.x >= 0 && pos.x < WIDTH;
}

bool move(Game& game)
{
    Snake& snake = game.snake;

    if (snake.body.empty()) return false;

    Vec2 next = snake.body.front() + snake.direction;

    if (!is_inside(next)) {
        game.running = false;
        return false;
    }

    bool grow = (next == game.food);

    // 일반 이동이면 기존 꼬리 칸으로 진입 가능
    auto end = snake.body.end();
    if (!grow) --end;

    // 일반 이동이면, 꼬리 충돌 로직을 넣지 않아도 된다
    if (std::find(snake.body.begin(), end, next) != end) {
        game.running = false;
        return false;
    }

    snake.body.push_front(next);

    if (!grow) {
        snake.body.pop_back();
    } else {
        // 임시 먹이 위치: 실제 게임에서는 랜덤 생성 예정
        // 성장 확인을 위해 현재 먹이는 제거
        game.food = {-1, -1};
    }

    return true;
}

/*
 * Display
 */

void display(const Game& game)
{
    int grid[HEIGHT][WIDTH] = {};

    if (is_inside(game.food)) {
        grid[game.food.y][game.food.x] = FOOD;
    }

    for (const Vec2& pos : game.snake.body) {
        if (is_inside(pos)) {
            grid[pos.y][pos.x] = SNAKE;
        }
    }

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << grid[y][x] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << '\n';
}

/*
 * Main
 */

int main(void)
{
    Game game;
    init(game, {1, 1});

    display(game);

    char key;

    while (game.running && std::cin >> key) {
        handle_input(game, key);

        if (!game.running) break;

        if (!move(game)) {
            std::cout << "Game Over!\n";
            break;
        }

        display(game);
    }

    return 0;
}