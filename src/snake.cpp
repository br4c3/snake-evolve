#include "snake.hpp"

void init_snake(Snake& snake, Vec2 pos)
{
    snake.direction = RIGHT;
    snake.body.clear();
    snake.body.push_front(pos);
}

Vec2 rotate_left(const Vec2& direction)
{
    return {-direction.x, direction.y};
}

Vec2 rotate_right(const Vec2& direction)
{
    return {direction.x, -direction.y};
}

Vec2 get_direction(const Snake& snake, Action action)
{
    switch (action) {
    case Action::LEFT:
        return rotate_left(snake.direction);

    case Action::RIGHT:
        return rotate_right(snake.direction);

    case Action::STRAIGHT:
        return snake.direction;
    }

    return snake.direction;
}

void apply_action(Snake& snake, Action action)
{
    snake.direction = get_direction(snake, action);
}