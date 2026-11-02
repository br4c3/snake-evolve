#include "agent.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

constexpr std::size_t REPLAY_CAPACITY = 50000;
constexpr std::size_t BATCH_SIZE = 64;
constexpr std::size_t WARMUP_SIZE = 1000;
constexpr int TARGET_UPDATE_INTERVAL = 500;

struct ForwardPass {
    std::array<float, INPUT_COUNT> input{};
    std::array<float, HIDDEN_COUNT> hidden{};
    std::array<float, ACTION_COUNT> output{};
};

std::array<float, INPUT_COUNT> state_vector(int state)
{
    std::array<float, INPUT_COUNT> input{};
    for (int index = 0; index < INPUT_COUNT; ++index)
        input[index] = (state & (1 << index)) ? 1.0f : 0.0f;
    return input;
}

ForwardPass forward(const Network& network, int state)
{
    ForwardPass pass;
    pass.input = state_vector(state);

    for (int hidden = 0; hidden < HIDDEN_COUNT; ++hidden) {
        float value = network.hidden_biases[hidden];
        for (int input = 0; input < INPUT_COUNT; ++input)
            value += network.input_weights[hidden][input] * pass.input[input];
        pass.hidden[hidden] = std::max(0.0f, value);
    }

    for (int action = 0; action < ACTION_COUNT; ++action) {
        float value = network.output_biases[action];
        for (int hidden = 0; hidden < HIDDEN_COUNT; ++hidden)
            value += network.output_weights[action][hidden] * pass.hidden[hidden];
        pass.output[action] = value;
    }

    return pass;
}

void remember(Agent& agent, const Experience& experience)
{
    if (agent.replay.size() < REPLAY_CAPACITY) {
        agent.replay.push_back(experience);
        return;
    }

    agent.replay[agent.replay_position] = experience;
    agent.replay_position = (agent.replay_position + 1) % REPLAY_CAPACITY;
}

float train_batch(Agent& agent, std::mt19937& rng)
{
    if (agent.replay.size() < WARMUP_SIZE)
        return 0.0f;

    Network gradients{};
    float total_loss = 0.0f;
    std::uniform_int_distribution<std::size_t> sample(0, agent.replay.size() - 1);

    for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        const Experience& experience = agent.replay[sample(rng)];
        const ForwardPass current = forward(agent.online, experience.state);

        float target_value = experience.reward;
        if (!experience.done) {
            const auto next_values = forward(agent.target, experience.next_state).output;
            target_value += agent.gamma *
                *std::max_element(next_values.begin(), next_values.end());
        }

        const int action = static_cast<int>(experience.action);
        const float error = current.output[action] - target_value;
        const float output_gradient = std::clamp(error, -1.0f, 1.0f);
        total_loss += std::abs(error) < 1.0f
            ? 0.5f * error * error
            : std::abs(error) - 0.5f;

        gradients.output_biases[action] += output_gradient;
        for (int hidden = 0; hidden < HIDDEN_COUNT; ++hidden) {
            gradients.output_weights[action][hidden] +=
                output_gradient * current.hidden[hidden];

            if (current.hidden[hidden] <= 0.0f)
                continue;

            const float hidden_gradient =
                output_gradient * agent.online.output_weights[action][hidden];
            gradients.hidden_biases[hidden] += hidden_gradient;
            for (int input = 0; input < INPUT_COUNT; ++input)
                gradients.input_weights[hidden][input] +=
                    hidden_gradient * current.input[input];
        }
    }

    const float scale = agent.learning_rate / static_cast<float>(BATCH_SIZE);
    for (int hidden = 0; hidden < HIDDEN_COUNT; ++hidden) {
        agent.online.hidden_biases[hidden] -=
            scale * gradients.hidden_biases[hidden];
        for (int input = 0; input < INPUT_COUNT; ++input)
            agent.online.input_weights[hidden][input] -=
                scale * gradients.input_weights[hidden][input];
    }

    for (int action = 0; action < ACTION_COUNT; ++action) {
        agent.online.output_biases[action] -=
            scale * gradients.output_biases[action];
        for (int hidden = 0; hidden < HIDDEN_COUNT; ++hidden)
            agent.online.output_weights[action][hidden] -=
                scale * gradients.output_weights[action][hidden];
    }

    ++agent.training_steps;
    if (agent.training_steps % TARGET_UPDATE_INTERVAL == 0)
        agent.target = agent.online;

    return total_loss / static_cast<float>(BATCH_SIZE);
}

} // namespace

void initialize_agent(Agent& agent, std::mt19937& rng)
{
    const float input_limit = std::sqrt(6.0f / (INPUT_COUNT + HIDDEN_COUNT));
    const float output_limit = std::sqrt(6.0f / (HIDDEN_COUNT + ACTION_COUNT));
    std::uniform_real_distribution<float> input_weight(-input_limit, input_limit);
    std::uniform_real_distribution<float> output_weight(-output_limit, output_limit);

    for (auto& row : agent.online.input_weights)
        for (float& weight : row)
            weight = input_weight(rng);

    for (auto& row : agent.online.output_weights)
        for (float& weight : row)
            weight = output_weight(rng);

    agent.online.hidden_biases.fill(0.0f);
    agent.online.output_biases.fill(0.0f);

    agent.target = agent.online;
    agent.replay.clear();
    agent.replay.reserve(REPLAY_CAPACITY);
    agent.replay_position = 0;
    agent.training_steps = 0;
    agent.epsilon = 1.0f;
}

int get_state(const Game& game)
{
    const Snake& snake = game.snake;
    const Vec2 head = snake.body.front();
    const Vec2 direction = snake.direction;

    const std::array<bool, INPUT_COUNT> features = {
        is_collision(game, head + direction),
        is_collision(game, head + rotate_left(direction)),
        is_collision(game, head + rotate_right(direction)),
        direction == UP,
        direction == DOWN,
        direction == LEFT,
        direction == RIGHT,
        game.food.y < head.y,
        game.food.y > head.y,
        game.food.x < head.x,
        game.food.x > head.x
    };

    int state = 0;
    for (int index = 0; index < INPUT_COUNT; ++index)
        if (features[index])
            state |= 1 << index;
    return state;
}

std::array<float, ACTION_COUNT> predict(const Agent& agent, int state)
{
    return forward(agent.online, state).output;
}

int best_action(const Agent& agent, int state)
{
    const auto values = predict(agent, state);
    return static_cast<int>(
        std::max_element(values.begin(), values.end()) - values.begin());
}

std::array<float, ACTION_COUNT> action_probabilities(const Agent& agent, int state)
{
    const float random_probability = agent.epsilon / ACTION_COUNT;
    std::array<float, ACTION_COUNT> probabilities{};
    probabilities.fill(random_probability);
    probabilities[best_action(agent, state)] += 1.0f - agent.epsilon;
    return probabilities;
}

Action choose_action(const Agent& agent, int state, std::mt19937& rng)
{
    std::uniform_real_distribution<float> probability(0.0f, 1.0f);
    std::uniform_int_distribution<int> random_action(0, ACTION_COUNT - 1);

    if (probability(rng) < agent.epsilon)
        return static_cast<Action>(random_action(rng));
    return static_cast<Action>(best_action(agent, state));
}

void save_model(const Agent& agent, const char* path)
{
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Failed to save DQN model to " << path << '\n';
        return;
    }

    file << INPUT_COUNT << ' ' << HIDDEN_COUNT << ' ' << ACTION_COUNT << '\n';
    file << std::setprecision(9);
    for (const auto& row : agent.online.input_weights) {
        for (float value : row)
            file << value << ' ';
        file << '\n';
    }
    for (float value : agent.online.hidden_biases)
        file << value << ' ';
    file << '\n';
    for (const auto& row : agent.online.output_weights) {
        for (float value : row)
            file << value << ' ';
        file << '\n';
    }
    for (float value : agent.online.output_biases)
        file << value << ' ';
    file << '\n';
}

void train(Agent& agent, int episodes, std::mt19937& rng)
{
    Game game;
    int score_sum = 0;
    int best_score = 0;
    float loss_sum = 0.0f;
    int loss_count = 0;

    for (int episode = 1; episode <= episodes; ++episode) {
        reset(game, rng);

        while (game.running) {
            const int state = get_state(game);
            const Action action = choose_action(agent, state, rng);
            const StepResult result = step(game, action, rng);
            const int next_state = result.done ? 0 : get_state(game);

            remember(agent, {state, action, result.reward, next_state, result.done});
            if (agent.replay.size() >= WARMUP_SIZE) {
                loss_sum += train_batch(agent, rng);
                ++loss_count;
            }
        }

        score_sum += game.score;
        best_score = std::max(best_score, game.score);
        agent.epsilon = std::max(
            0.02f, static_cast<float>(std::exp(-episode / 2500.0)));

        if (episode % 250 == 0) {
            const float average_loss =
                loss_count > 0 ? loss_sum / loss_count : 0.0f;
            std::cout << "Episode: " << episode
                      << " | Avg Score: " << score_sum / 250.0f
                      << " | Best: " << best_score
                      << " | Loss: " << average_loss
                      << " | Epsilon: " << agent.epsilon << '\n';
            score_sum = 0;
            loss_sum = 0.0f;
            loss_count = 0;
        }
    }

    agent.target = agent.online;
}
