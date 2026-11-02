#include <algorithm>
#include <iostream>
#include <random>
#include <string>

#include "agent.hpp"
#include "renderer.hpp"

constexpr int DEFAULT_EPISODES = 10000;

int main(int argc, char** argv)
{
    int episodes = DEFAULT_EPISODES;
    bool render = true;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--no-render") {
            render = false;
        } else if (argument == "--episodes" && index + 1 < argc) {
            episodes = std::max(1, std::stoi(argv[++index]));
        }
    }

    std::mt19937 rng(std::random_device{}());

    Agent agent;
    initialize_agent(agent, rng);

    std::cout << "Training started...\n";

    train(agent, episodes, rng);
    save_model(agent, "dqn_model.txt");

    std::cout << "\nTraining completed.\n";

    // 학습 종료 후 탐험 비활성화
    agent.epsilon = 0.0f;

    if (render) {
        std::cout << "Opening visual evaluation...\n";
        run_visual_evaluation(agent, rng);
    }

    return 0;
}
