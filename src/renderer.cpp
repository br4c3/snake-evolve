#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "raylib.h"

namespace {

constexpr int SCREEN_WIDTH = 900;
constexpr int SCREEN_HEIGHT = 620;
constexpr int CELL_SIZE = 50;
constexpr int BOARD_X = 30;
constexpr int BOARD_Y = 60;
constexpr int PANEL_X = 560;

struct Decision {
    int state = 0;
    Action action = Action::STRAIGHT;
    std::array<float, ACTION_COUNT> q_values{};
    std::array<float, ACTION_COUNT> probabilities{};
};

const char* action_name(int action)
{
    constexpr std::array<const char*, ACTION_COUNT> names = {
        "STRAIGHT", "LEFT", "RIGHT"
    };
    return names[action];
}

Decision make_decision(const Agent& agent, const Game& game,
                       std::mt19937& rng)
{
    Decision decision;
    decision.state = get_state(game);
    decision.q_values = agent.q_table[decision.state];
    decision.probabilities = action_probabilities(agent, decision.state);
    decision.action = choose_action(agent, decision.state, rng);
    return decision;
}

void draw_board(const Game& game)
{
    DrawText("Q-Learning Snake", BOARD_X, 20, 24, RAYWHITE);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            DrawRectangleLines(BOARD_X + x * CELL_SIZE,
                               BOARD_Y + y * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE, DARKGRAY);
        }
    }

    if (is_inside(game.food)) {
        DrawRectangle(BOARD_X + game.food.x * CELL_SIZE + 5,
                      BOARD_Y + game.food.y * CELL_SIZE + 5,
                      CELL_SIZE - 10, CELL_SIZE - 10, RED);
    }

    bool head = true;
    for (const Vec2& position : game.snake.body) {
        DrawRectangle(BOARD_X + position.x * CELL_SIZE + 3,
                      BOARD_Y + position.y * CELL_SIZE + 3,
                      CELL_SIZE - 6, CELL_SIZE - 6,
                      head ? LIME : GREEN);
        head = false;
    }
}

void draw_q_table(const Agent& agent, const Decision& decision)
{
    char text[128];

    DrawText("Current Q-table row", PANEL_X, 32, 22, RAYWHITE);
    std::snprintf(text, sizeof(text), "State: %d", decision.state);
    DrawText(text, PANEL_X, 70, 18, LIGHTGRAY);
    std::snprintf(text, sizeof(text), "Epsilon: %.3f", agent.epsilon);
    DrawText(text, PANEL_X, 96, 18, LIGHTGRAY);

    DrawText("Action", PANEL_X, 145, 18, GRAY);
    DrawText("Q-value", PANEL_X + 120, 145, 18, GRAY);
    DrawText("Prob.", PANEL_X + 230, 145, 18, GRAY);
    DrawLine(PANEL_X, 170, SCREEN_WIDTH - 25, 170, GRAY);

    for (int action = 0; action < ACTION_COUNT; ++action) {
        const int y = 190 + action * 55;
        const bool selected = action == static_cast<int>(decision.action);
        const Color color = selected ? YELLOW : RAYWHITE;

        DrawText(action_name(action), PANEL_X, y, 18, color);
        std::snprintf(text, sizeof(text), "% .4f",
                      decision.q_values[action]);
        DrawText(text, PANEL_X + 120, y, 18, color);
        std::snprintf(text, sizeof(text), "%5.1f%%",
                      decision.probabilities[action] * 100.0f);
        DrawText(text, PANEL_X + 225, y, 18, color);
    }

    DrawLine(PANEL_X, 350, SCREEN_WIDTH - 25, 350, GRAY);
    DrawText("Next action", PANEL_X, 375, 18, LIGHTGRAY);
    DrawText(action_name(static_cast<int>(decision.action)),
             PANEL_X, 405, 28, YELLOW);

    DrawText("State bits", PANEL_X, 465, 18, LIGHTGRAY);
    char bits[12];
    for (int index = 0; index < 11; ++index)
        bits[index] = (decision.state & (1 << (10 - index))) ? '1' : '0';
    bits[11] = '\0';
    DrawText(bits, PANEL_X, 493, 22, RAYWHITE);

    DrawText("Q-value = expected reward", PANEL_X, 535, 15, GRAY);
    DrawText("Prob. = epsilon-greedy policy", PANEL_X, 556, 15, GRAY);
}

void draw_status(const Game& game, bool paused, float steps_per_second)
{
    char text[160];
    std::snprintf(text, sizeof(text), "Score: %d   Speed: %.0f/s   %s",
                  game.score, steps_per_second,
                  game.running ? (paused ? "PAUSED" : "RUNNING") : "GAME OVER");
    DrawText(text, BOARD_X, 575, 17, RAYWHITE);
    DrawText("SPACE pause | R restart | UP/DOWN speed | ESC quit",
             BOARD_X, 598, 14, GRAY);
}

} // namespace

void run_visual_evaluation(Agent& agent, std::mt19937& rng)
{
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "snake-evolve");
    SetTargetFPS(60);

    Game game;
    reset(game, rng);
    Decision decision = make_decision(agent, game, rng);

    bool paused = false;
    float steps_per_second = 5.0f;
    float accumulator = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE) && game.running)
            paused = !paused;

        if (IsKeyPressed(KEY_R)) {
            reset(game, rng);
            decision = make_decision(agent, game, rng);
            accumulator = 0.0f;
            paused = false;
        }

        if (IsKeyPressed(KEY_UP))
            steps_per_second = std::min(30.0f, steps_per_second + 1.0f);
        if (IsKeyPressed(KEY_DOWN))
            steps_per_second = std::max(1.0f, steps_per_second - 1.0f);

        if (game.running && !paused) {
            accumulator += GetFrameTime();
            const float step_time = 1.0f / steps_per_second;

            if (accumulator >= step_time) {
                accumulator = std::fmod(accumulator, step_time);
                const StepResult result = step(game, decision.action, rng);
                if (!result.done)
                    decision = make_decision(agent, game, rng);
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        draw_board(game);
        draw_q_table(agent, decision);
        draw_status(game, paused, steps_per_second);
        EndDrawing();
    }

    CloseWindow();
}
