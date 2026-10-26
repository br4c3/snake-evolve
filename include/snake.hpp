#pragma once

#include <deque>

#include "vec2.hpp"

struct Snake {
    Vec2 direction = RIGHT;
    std::deque<Vec2> body;
};

void init_snake(Snake& snake, Vec2 pos);

bool is_opposite(const Vec2& a, const Vec2& b);

void change_direction(Snake& snake, const Vec2& direction);