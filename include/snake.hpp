#pragma once

#include <deque>

#include "vec2.hpp"

enum class Action {
    STRAIGHT = 0,
    LEFT,
    RIGHT
};

struct Snake {
    Vec2 direction = RIGHT;
    std::deque<Vec2> body;
};

void init_snake(Snake& snake, Vec2 pos);

Vec2 rotate_left(const Vec2& direction);
Vec2 rotate_right(const Vec2& direction);

Vec2 get_direction(const Snake& snake, Action action);
void apply_action(Snake& snake, Action action);