import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment
states = range(10) # Example: 10 states
actions = range(2) # Example: 2 ac@ons (e.g., leD and right)
# Ini@alize Q-table
Q = np.zeros((len(states), len(actions)))
# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
# Number of episodes for the agent to learn from
total_episodes = 1000
# Q-learning algorithm
for episode in range(total_episodes):
    state = random.choice(states) # Start with a random state
    done = False
    while not done:
        # Explora@on-exploita@on trade-off
        if random.uniform(0, 1) < exploration_rate:
            action = random.choice(actions) # Explore: choose a random ac@on
        else:
            action = np.argmax(Q[state, :]) # Exploit: choose the best known ac@on
        # Take the ac@on and observe the outcome state and reward
        # For simplicity, we assume a fixed reward and next state
        # In a real scenario, the environment would provide this
        next_state = state + 1 # Example transi@on
        reward = 1 if next_state == len(states) - 1 else 0 # Example reward
        # Update Q-table using the Bellman equa@on
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
        # Transi@on to the next state
        state = next_state
        # Check if the goal state is reached
        if state == len(states) - 1:
            done = True
    # Decay explora@on rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
# Display the final Q-table
print("Q-table aDer training:")
print(Q)

# Assume you have a list 'total_rewards' that stores the total reward for each episode
total_rewards = []
for episode in range(total_episodes):
    total_reward = 0
    state = random.choice(states)
    done = False
    while not done:
        # ... [rest of your episode loop] ...
        total_reward += reward
        total_rewards.append(total_reward)
# PloHng the total rewards per episode
plt.plot(total_rewards)
plt.title('Total Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()