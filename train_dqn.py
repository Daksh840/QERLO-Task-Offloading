from env_scheduler import TaskOffloadingEnv
from dqn_agent import DQNAgent

env = TaskOffloadingEnv("data/gml_dags/part.0.gml", num_nodes=8)
agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

episodes = 100
rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.train()

    agent.update_target()
    agent.decay_epsilon()
    
    rewards_per_episode.append(total_reward)  # ✅ log reward
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Learning Curve: Reward vs. Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_reward_curve.png")
plt.show()
