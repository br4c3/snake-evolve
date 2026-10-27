#pragma once

#include <random>

#include "snake.hpp"

constexpr int WIDTH  = 10;
constexpr int HEIGHT = 10;

constexpr int EMPTY = 0;
constexpr int SNAKE = 2;
constexpr int FOOD  = 3;

constexpr int MAX_IDLE = 200;

struct Game {
    Snake snake;
    Vec2  food = {-1, -1};

    int score = 0;
    int idle  = 0;

    bool running = true;
};

struct StepResult {
    float reward;
    bool  done;
};

bool is_inside(const Vec2& pos);
bool is_collision(const Game& game, Vec2 next);

bool spawn_food(Game& game, std::mt19937& rng);

void reset(Game& game, std::mt19937& rng);

StepResult step(Game& game, Action action, std::mt19937& rng);