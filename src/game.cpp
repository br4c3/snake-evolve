#include "game.hpp"

#include <algorithm>
#include <vector>

bool is_inside(const Vec2& pos)
{
    return pos.y >= 0 && pos.y < HEIGHT &&
           pos.x >= 0 && pos.x < WIDTH;
}

bool is_collision(const Game& game, Vec2 next)
{
    if (!is_inside(next)) return true;

    const auto& body = game.snake.body;

    bool grow = (next == game.food);

    auto end = body.end();

    if (!grow && !body.empty()) --end;

    return std::find(body.begin(), end, next) != end;
}

bool spawn_food(Game& game, std::mt19937& rng)
{
    std::vector<Vec2> empty_cells;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Vec2 pos = {y, x};

            auto it = std::find(
                game.snake.body.begin(),
                game.snake.body.end(),
                pos
            );

            if (it == game.snake.body.end())
                empty_cells.push_back(pos);
        }
    }

    if (empty_cells.empty()) {
        game.food = {-1, -1};
        return false;
    }

    std::uniform_int_distribution<int> dist(
        0, static_cast<int>(empty_cells.size()) - 1
    );

    game.food = empty_cells[dist(rng)];

    return true;
}

void reset(Game& game, std::mt19937& rng)
{
    init_snake(game.snake, {HEIGHT / 2, WIDTH / 2});

    game.score   = 0;
    game.idle    = 0;
    game.running = true;

    spawn_food(game, rng);
}

StepResult step(Game& game, Action action, std::mt19937& rng)
{
    if (!game.running) return {0.0f, true};

    Snake& snake = game.snake;

    apply_action(snake, action);

    Vec2 next = snake.body.front() + snake.direction;

    if (is_collision(game, next)) {
        game.running = false;
        return {-10.0f, true};
    }

    bool grow = (next == game.food);

    snake.body.push_front(next);

    if (!grow)
        snake.body.pop_back();

    game.idle++;

    if (grow) {
        game.score++;
        game.idle = 0;

        if (!spawn_food(game, rng)) {
            game.running = false;
            return {20.0f, true};
        }

        return {10.0f, false};
    }

    if (game.idle >= MAX_IDLE) {
        game.running = false;
        return {-5.0f, true};
    }

    return {-0.01f, false};
}