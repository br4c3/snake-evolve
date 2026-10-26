#include <iostream>

#include "game.hpp"

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

int main(void)
{
    Game game;
    init_game(game);

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