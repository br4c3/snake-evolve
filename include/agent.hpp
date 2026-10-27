#pragma once

#include <array>
#include <deque>
#include <random>

#include "game.hpp"

constexpr int STATE_COUNT  = 2048;
constexpr int ACTION_COUNT = 3;

constexpr int N_STEP = 40;

struct Experience {
    int    state;
    Action action;
    float  reward;
    int    next_state;
    bool   done;
};

struct Agent {
    std::array<std::array<float, ACTION_COUNT>, STATE_COUNT> q_table{};

    float alpha   = 0.1f;
    float gamma   = 0.9f;
    float epsilon = 1.0f;
};

int get_state(const Game& game);

int best_action(const Agent& agent, int state);

std::array<float, ACTION_COUNT> action_probabilities(const Agent& agent,
                                                     int state);

Action choose_action(const Agent& agent, int state, std::mt19937& rng);

void print_q_table(const Agent& agent);

void learn(Agent& agent, int state, Action action, float reward, int next_state, bool done);

void learn_n_step(Agent& agent, std::deque<Experience>& buffer);

void train(Agent& agent, int episodes, std::mt19937& rng);
