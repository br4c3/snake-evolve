#include <iostream>
#include <random>

#include "agent.hpp"
#include "game.hpp"

constexpr int EPISODES = 15000;

/*
 * Render
 */

void display(const Game& game)
{
    int grid[HEIGHT][WIDTH] = {};

    if (is_inside(game.food))
        grid[game.food.y][game.food.x] = FOOD;

    for (const Vec2& pos : game.snake.body) {
        if (is_inside(pos))
            grid[pos.y][pos.x] = SNAKE;
    }

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << grid[y][x] << ' ';
        }

        std::cout << '\n';
    }

    std::cout << "Score: " << game.score << "\n\n";
}

/*
 * Main
 */

int main(void)
{
    std::mt19937 rng(std::random_device{}());

    Agent agent;

    std::cout << "Training started...\n";

    train(agent, EPISODES, rng);

    std::cout << "\nTraining completed.\n";

    // 학습 종료 후 탐험 비활성화
    agent.epsilon = 0.0f;

    Game game;
    reset(game, rng);

    while (game.running) {
        int state = get_state(game);

        Action action = choose_action(agent, state, rng);

        step(game, action, rng);
    }

    std::cout << "\nEvaluation game:\n";

    display(game);

    std::cout << "Final Score: " << game.score << '\n';

    return 0;
}