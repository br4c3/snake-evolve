#include "agent.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

/*
 * State
 *
 * 11 binary features
 *
 * 0~2  : danger (straight, left, right)
 * 3~6  : direction (up, down, left, right)
 * 7~10 : food (up, down, left, right)
 */

int get_state(const Game& game)
{
    const Snake& snake = game.snake;

    Vec2 head = snake.body.front();
    Vec2 dir  = snake.direction;

    Vec2 straight = head + dir;
    Vec2 left     = head + rotate_left(dir);
    Vec2 right    = head + rotate_right(dir);

    std::array<bool, 11> features = {
        is_collision(game, straight),
        is_collision(game, left),
        is_collision(game, right),

        dir == UP,
        dir == DOWN,
        dir == LEFT,
        dir == RIGHT,

        game.food.y < head.y,
        game.food.y > head.y,
        game.food.x < head.x,
        game.food.x > head.x
    };

    int state = 0;

    for (int i = 0; i < 11; i++) {
        if (features[i])
            state |= (1 << i);
    }

    return state;
}

/*
 * Action selection
 */

int best_action(const Agent& agent, int state)
{
    const auto& q = agent.q_table[state];

    return static_cast<int>(
        std::max_element(q.begin(), q.end()) - q.begin()
    );
}

Action choose_action(const Agent& agent, int state,
                     std::mt19937& rng)
{
    std::uniform_real_distribution<float> probability(0.0f, 1.0f);

    std::uniform_int_distribution<int> random_action(
        0, ACTION_COUNT - 1
    );

    if (probability(rng) < agent.epsilon)
        return static_cast<Action>(random_action(rng));

    return static_cast<Action>(best_action(agent, state));
}

/*
 * Q-Learning
 */

void learn(Agent& agent, int state, Action action,
           float reward, int next_state, bool done)
{
    int a = static_cast<int>(action);

    float& q = agent.q_table[state][a];

    float next_max = 0.0f;

    if (!done) {
        const auto& next_q = agent.q_table[next_state];

        next_max = *std::max_element(
            next_q.begin(), next_q.end()
        );
    }

    float target = reward + agent.gamma * next_max;

    q += agent.alpha * (target - q);
}

/*
 * Training
 */

void train(Agent& agent, int episodes, std::mt19937& rng)
{
    Game game;

    int total_score = 0;
    int best_score  = 0;

    for (int episode = 1; episode <= episodes; episode++) {
        reset(game, rng);

        while (game.running) {
            int state = get_state(game);

            Action action = choose_action(agent, state, rng);

            StepResult result = step(game, action, rng);

            int next_state = 0;

            if (!result.done)
                next_state = get_state(game);

            learn(agent, state, action,
                  result.reward, next_state, result.done);
        }

        total_score += game.score;
        best_score = std::max(best_score, game.score);

        agent.epsilon = std::max(
            0.02f,
            static_cast<float>(std::exp(-episode / 3500.0))
        );

        if (episode % 1000 == 0) {
            std::cout << "Episode: " << episode
                      << " | Avg Score: " << total_score / 1000.0f
                      << " | Best: " << best_score
                      << " | Epsilon: " << agent.epsilon
                      << '\n';

            total_score = 0;
        }
    }
}