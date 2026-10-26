# snake-evolve

A C++17 Q-learning Snake that trains without rendering and then opens a raylib
visualizer for the learned policy.

The right-hand panel shows the current Q-table row, the exact epsilon-greedy
action probabilities, the selected next action, and the 11 binary state
features used by the agent.

## Build and run

```sh
cmake -S . -B build
cmake --build build -j
./build/snake-evolve
```

If raylib 6.0 is not installed, CMake downloads and builds it automatically.

## Controls

- `Space`: pause or resume
- `R`: restart the evaluation game
- `Up` / `Down`: change playback speed
- `Esc`: quit
