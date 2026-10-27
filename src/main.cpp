#include <iostream>
#include <random>

#include "agent.hpp"
#include "renderer.hpp"

constexpr int EPISODES = 900000;

int main(void)
{
    std::mt19937 rng(std::random_device{}());

    Agent agent;

    std::cout << "Training started...\n";

    train(agent, EPISODES, rng);
    print_q_table(agent);

    std::cout << "\nTraining completed.\n";

    // 학습 종료 후 탐험 비활성화
    agent.epsilon = 0.0f;

    std::cout << "Opening visual evaluation...\n";
    run_visual_evaluation(agent, rng);

    return 0;
}
