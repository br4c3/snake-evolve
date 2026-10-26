#pragma once

#include "snake.hpp"

constexpr int WIDTH  = 10;
constexpr int HEIGHT = 10;

constexpr int EMPTY = 0;
constexpr int SNAKE = 2;
constexpr int FOOD  = 3;

struct Game {
    Snake snake;
    Vec2 food;
    bool running = true;
    int score = 0;
};

void init_game(Game& game);

bool is_inside(const Vec2& pos);
bool move(Game& game);