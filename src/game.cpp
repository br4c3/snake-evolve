#include "game.hpp"

#include <algorithm>

void init_game(Game& game)
{
    init_snake(game.snake, {1, 1});

    game.food    = {3, 3};
    game.running = true;
    game.score   = 0;
}

bool is_inside(const Vec2& pos)
{
    return pos.y >= 0 && pos.y < HEIGHT &&
           pos.x >= 0 && pos.x < WIDTH;
}

bool move(Game& game)
{
    if (!game.running || game.snake.body.empty())
        return false;

    Snake& snake = game.snake;

    Vec2 next = snake.body.front() + snake.direction;

    bool grow = (next == game.food);

    if (!is_inside(next)) {
        game.running = false;
        return false;
    }

    auto end = snake.body.end();

    if (!grow) --end;

    if (std::find(snake.body.begin(), end, next) != end) {
        game.running = false;
        return false;
    }

    snake.body.push_front(next);

    if (!grow) {
        snake.body.pop_back();
    } else {
        game.score++;
        game.food = {-1, -1};
    }

    return true;
}