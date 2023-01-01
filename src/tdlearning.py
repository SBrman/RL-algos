#!/usr/bin/local/python3

__author__ = "Simanta Barman"
__email__ = "barma017@umn.edu"


from itertools import product
import random
import numpy as np
import gym


class RLAgent:
    def __init__(self, env_name: str, discretizer: int, init_reward: int, 
                 lower_bound: float, upper_bound: float):
        # Create environment
        self.env = gym.make(env_name)

        # State space
        self.N = discretizer    # Discretize the state space into NxN state space
        self.state_shape = self.env.observation_space.shape
        self.state_space_size = np.ones(shape=self.state_shape) * (self.N - 1)
        self.low = np.clip(self.env.observation_space.low, lower_bound, upper_bound)
        self.high = np.clip(self.env.observation_space.high, lower_bound, upper_bound)

        # Action space
        self.A = self.env.action_space.n

        # Initialize the Q values
        self.Q = {action: {state: init_reward for state in self.state_space} 
                        for action in range(self.A)}

    @property
    def state_space(self):
        all_states = [list(range(self.N)), ] * self.state_shape[0]
        for state in product(*all_states):
            yield state

    def reset_to_initial_state(self):
        return self.discretize(self.env.reset())

    def discretize(self, state: np.array) -> tuple:
        """Returns the discretized state"""
        dspace = ((state - self.low) / (self.high - self.low)) * self.state_space_size
        dspace = dspace.astype(np.int64)
        return tuple(np.clip(dspace, 0, self.N-1))

    def max_Q_a(self, state: tuple) -> tuple[float, int]:
        """Returns the maximum Q value and the best action taken at the input state """
        return max((self.Q[a][state], a) for a in range(self.A))

    def max_Q(self, state: tuple) -> float:
        """Returns the maximum Q values where decision variables are the actions."""
        return self.max_Q_a(state)[0]

    def argmax_Q(self, state: tuple) -> int:
        """Returns the best action based on the Q values."""
        return self.max_Q_a(state)[1]

    def choose_best_policy(self, epsilon: float) -> bool:
        """Returns whether best policy should be selected"""
        return True if np.random.random() <= (1 - epsilon) else False

    def epsilon_greedy(self, state: tuple, epsilon: float) -> int:
        """Returns the epsilon greedy action"""
        if self.choose_best_policy(epsilon):
            action = self.argmax_Q(state)
        else:
            action = random.sample(list(range(self.A)), 1).pop()
        return action

    def get_Q(self, state, action):
        """Returns the Q value based on the input state and action"""
        # assert action in self.Q and state in self.Q[action], "Q values are not properly initialized."
        return self.Q[action][state]

    def learn_Q(self, episodes: int, epsilon: float, alpha: float, gamma: float, render_interval: int):
        """Start learning the Q factor."""

        for episode in range(episodes):

            # Initialize state
            state = self.reset_to_initial_state()

            # Initialize terminal state not reached
            reached_terminal = False

            while not reached_terminal:
                # Epsilon greedy action selection
                action = self.epsilon_greedy(epsilon=epsilon, state=state)

                # Take epsilon greedy aciton
                new_state, reward, reached_terminal, _ = self.env.step(action)

                # Render the action (Not part of Q learning) just for visualization
                if episode % render_interval == 0:
                    self.env.render()

                # Discretize new state, s' after taking action a
                new_state = self.discretize(new_state)

                # Bellman Optimality Equation
                if reached_terminal:
                    self.Q[action][state] = reward
                else:
                    temporal_difference = reward + gamma * self.max_Q(new_state) - self.get_Q(state, action)
                    self.Q[action][state] += alpha * temporal_difference

                # Update state
                state = new_state

        self.env.close()
        print(f"Learned the Q factor with total {episodes} episodes")

    def play(self):
        """Play after learning"""

        terminal = False
        dstate = self.reset_to_initial_state()

        total_reward = 0

        while not terminal:
            self.env.render()

            action = self.argmax_Q(dstate)
            state, reward, terminal, _ = self.env.step(action)
            
            total_reward += reward

            dstate = self.discretize(state)

        self.env.close()

        return total_reward


if __name__ == "__main__":

    epsilon = 0.6
    learning_rate = 0.1
    discount = 0.95

    agent = RLAgent(env_name="MountainCar-v0", discretizer=50, init_reward=2, lower_bound=-50000, upper_bound=50000)
    agent.learn_Q(episodes=50000, epsilon=epsilon, alpha=learning_rate, gamma=discount, render_interval=500)

    for _ in range(10):
        agent.play()

    avg_rewards = 0
    k = 100

    for i in range(k):
        avg_reward = agent.play()
        avg_rewards += avg_reward

    print(f'Avg reward = {avg_rewards / k}')
