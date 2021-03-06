# Taxi
Project for solving the OpenAI Gym Taxi-v3 environment

## Introduction

This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.

[Dietterich2000] T Erez, Y Tassa, E Todorov, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition", 2011.

![](images/image.png)

## Installation

Use the [docker](https://www.docker.com) to test the algoritm.

```bash
docker pull fernandofsilva/taxi
```

## Usage

Run with default parameters

```bash
docker run fernandofsilva/taxi
```

Run with custom parameters

```bash
docker run fernandofsilva/taxi --alpha 0.07 --gamma 1.0 --num_episodes 20000 --window 100
```

Each parameter correspond to:

- alpha: step-size parameter for the update step
- gamma: discount rate
- num_episodes: number of episodes of agent-environment interaction
- window: number of episodes to consider when calculating average rewards

## Output

Expected final output

```
Episode 20000/20000 || Best average reward 9.085
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)