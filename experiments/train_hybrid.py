import os
import sys
import networkx as nx
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import your modules
from src.algorithms.QIPSO import QIPSO_Scheduler
from src.algorithms.dqn_agent import DQNAgent
from src.environment.scheduler_env import TaskOffloadingEnv

# Rest of your existing code...
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
# ======================
#  CONFIGURATION
# ======================
class Config:
    # Paths
    DAG_PATH = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/processed/part.10.gml"
    
    # Environment
    NUM_NODES = 6                # Start with fewer nodes for faster convergence
    MAX_TASK_PER_NODE = 3        # Prevent node overload
    
    # QIPSO Parameters
    QIPSO_PARTICLES = 15         # Reduced from 20 for faster initialization
    QIPSO_ITERATIONS = 30        # Reduced from 50
    
    # DQN Parameters
    EPISODES = 200               # Increased from 100
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.01
    EPSILON_DECAY = 0.995        # Slower decay for better exploration
    BATCH_SIZE = 64              # Standard batch size
    
    # Hybrid Strategy
    QIPSO_PROBABILITY = 0.5      # Base probability of using QIPSO
    QIPSO_PROB_DECAY = 0.99      # Decay QIPSO influence over time
    
    # Monitoring
    REWARD_WINDOW = 20           # Moving average window
    SAVE_INTERVAL = 50           # Save model every N episodes


def load_dag():
    """Load and validate the DAG file"""
    dag_path = project_root / Config.DAG_PATH
    try:
        G = nx.read_gml(dag_path)
        if not isinstance(G, nx.DiGraph):
            raise ValueError("Input file is not a directed graph")
        
        # Convert node labels to strings if needed
        if all('label' in data for _, data in G.nodes(data=True)):
            G = nx.relabel_nodes(G, {n: str(data['label']) for n, data in G.nodes(data=True)})
        
        return G
    except Exception as e:
        raise ValueError(f"Failed to load DAG: {str(e)}")
# ======================
#  TRAINING SETUP
# ======================
def setup():
    # Load DAG first
    print("Loading DAG...")
    G = load_dag()
    
    # Initialize QIPSO Scheduler with the graph
    print("Initializing QIPSO Scheduler...")
    qipso = QIPSO_Scheduler(
        graph=G,
        num_edge_nodes=Config.NUM_NODES,
        num_particles=Config.QIPSO_PARTICLES,
        max_iter=Config.QIPSO_ITERATIONS
    )
    qipso_schedule, qipso_reward = qipso.run_optimization()
    print(f"QIPSO Initialized | Best Reward: {qipso_reward:.2f}")
    
    # Initialize Environment
    env = TaskOffloadingEnv(
        dag=G,
        num_nodes=Config.NUM_NODES,
    )
    
    # Initialize DQN Agent with correct parameters
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=Config.INITIAL_EPSILON,
        epsilon_min=Config.FINAL_EPSILON,
        epsilon_decay=Config.EPSILON_DECAY,
        memory_size=10000,  # Now properly accepted
        batch_size=Config.BATCH_SIZE
    )
    
    return qipso_schedule, env, agent


# ======================
#  TRAINING LOOP
# ======================
def train():
    qipso_schedule, env, agent = setup()
    
    # Tracking variables
    rewards = []
    moving_avg = []
    reward_window = deque(maxlen=Config.REWARD_WINDOW)
    best_reward = -np.inf
    qipso_prob = Config.QIPSO_PROBABILITY
    start_time = time.time()  # Add this line to record start time
    patience = 20
    no_improvement = 0
    
    for episode in range(1, Config.EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            current_task = env.task_list[env.current_task_idx]
            
            # Hybrid action selection
            if current_task in qipso_schedule and random.random() < qipso_prob:
                action = qipso_schedule[current_task]
            else:
                action = agent.act(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Post-episode updates
        agent.decay_epsilon()
        qipso_prob *= Config.QIPSO_PROB_DECAY
        
        # Track performance
        rewards.append(total_reward)
        reward_window.append(total_reward)
        moving_avg.append(np.mean(reward_window))
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement = 0
            agent.save("best_model.pth")
        else:
            no_improvement +=1

        if no_improvement >=patience:
            print(f"Early stopping at Episode {episode}")
            break
        
        
        # Periodic logging
        if episode % Config.SAVE_INTERVAL == 0:
            agent.save(f"model_episode_{episode}.pth")
            
        print(f"Episode {episode:3d} | "
              f"Reward: {total_reward:7.2f} | "
              f"Avg: {moving_avg[-1]:7.2f} | "
              f"ε: {agent.epsilon:.3f} | "
              f"QIPSO%: {qipso_prob*100:.1f}%")
    
    # Training complete
    print(f"\nTraining completed in {(time.time()-start_time)/60:.2f} minutes")  # Now start_time is defined
    return rewards, moving_avg

# In DQNAgent.save():
def save(self, filename: Path, env):
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'input_dim': self.state_size,
        'output_dim': self.action_size,
        'env_version': '2.1',  # Update version when architecture changes
        'training_params': {
            'episodes': episode_count,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
    }, str(filename))


# ======================
#  VISUALIZATION
# ======================
def plot_results(rewards, moving_avg):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    plt.plot(moving_avg, linewidth=2, label=f'Moving Avg ({Config.REWARD_WINDOW} eps)', color='red')
    
    # Add QIPSO probability decay visualization
    qipso_probs = [Config.QIPSO_PROBABILITY * (Config.QIPSO_PROB_DECAY**ep) 
                   for ep in range(len(rewards))]
    ax2 = plt.gca().twinx()
    ax2.plot(qipso_probs, 'g--', alpha=0.5, label='QIPSO Probability')
    ax2.set_ylabel('QIPSO Usage Probability', color='green')
    
    plt.title(f"Hybrid Training Performance\n"
              f"Final Avg Reward: {moving_avg[-1]:.2f} | "
              f"Best Reward: {max(rewards):.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Reward", color='blue')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_performance.png", dpi=300, bbox_inches='tight')

# ======================
#  MAIN EXECUTION
# ======================
if __name__ == "__main__":
    rewards, moving_avg = train()
    plot_results(rewards, moving_avg)
