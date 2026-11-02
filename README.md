# snake-evolve

A C++17 Deep Q-Network Snake that trains without rendering and then opens a
raylib visualizer for the learned policy. The DQN is implemented directly in
C++ with an 11-64-3 neural network, replay memory, and a target network.

The right-hand panel shows the neural network's three Q-value outputs, the
exact epsilon-greedy action probabilities, the selected next action, and the
11 binary state features used by the agent. Trained weights are saved to
`dqn_model.txt`.

## Build and run

```sh
cmake -S . -B build
cmake --build build -j
./build/snake-evolve
```

If raylib 6.0 is not installed, CMake downloads and builds it automatically.

For a short headless training run:

```sh
./build/snake-evolve --episodes 500 --no-render
```

## Controls

- `Space`: pause or resume
- `R`: restart the evaluation game
- `Up` / `Down`: change playback speed
- `Esc`: quit
