#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <vector>

#include "game.hpp"

constexpr int INPUT_COUNT  = 11;
constexpr int HIDDEN_COUNT = 64;
constexpr int ACTION_COUNT = 3;

struct Experience {
    int state;
    Action action;
    float reward;
    int next_state;
    bool done;
};

struct Network {
    std::array<std::array<float, INPUT_COUNT>, HIDDEN_COUNT> input_weights{};
    std::array<float, HIDDEN_COUNT> hidden_biases{};
    std::array<std::array<float, HIDDEN_COUNT>, ACTION_COUNT> output_weights{};
    std::array<float, ACTION_COUNT> output_biases{};
};

struct Agent {
    Network online;
    Network target;

    std::vector<Experience> replay;
    std::size_t replay_position = 0;
    int training_steps = 0;

    float learning_rate = 0.001f;
    float gamma = 0.95f;
    float epsilon = 1.0f;
};

void initialize_agent(Agent& agent, std::mt19937& rng);

int get_state(const Game& game);

std::array<float, ACTION_COUNT> predict(const Agent& agent, int state);
int best_action(const Agent& agent, int state);
std::array<float, ACTION_COUNT> action_probabilities(const Agent& agent,
                                                     int state);
Action choose_action(const Agent& agent, int state, std::mt19937& rng);

void save_model(const Agent& agent, const char* path);
void train(Agent& agent, int episodes, std::mt19937& rng);
