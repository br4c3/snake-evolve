#include "snake.hpp"

void init_snake(Snake& snake, Vec2 pos)
{
    snake.direction = RIGHT;
    snake.body.clear();
    snake.body.push_front(pos);
}

bool is_opposite(const Vec2& a, const Vec2& b)
{
    return a.y + b.y == 0 && a.x + b.x == 0;
}

void change_direction(Snake& snake, const Vec2& direction)
{
    if (is_opposite(snake.direction, direction)) return;

    snake.direction = direction;
}