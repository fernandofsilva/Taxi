from agent import Agent
from monitor import interact
import gym
import argparse

# Instantiate argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--alpha', nargs='?', const=1, type=float, default=0.07)
parser.add_argument('--gamma', nargs='?', const=1, type=float, default=1.0)
parser.add_argument('--num_episodes', nargs='?', const=1, type=int, default=20000)
parser.add_argument('--window', nargs='?', const=1, type=int, default=100)

# Pass args
args = parser.parse_args()


if __name__ == '__main__':

    # Create environment
    env = gym.make('Taxi-v3')

    # Instantiate agent
    agent = Agent(alpha=args.alpha, gamma=args.gamma)

    # Interact with environment
    avg_rewards, best_avg_reward = interact(
        env,
        agent,
        num_episodes=args.num_episodes,
        window=args.window
    )
