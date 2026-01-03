import os
import networkx as nx
from pathlib import Path
from src.algorithms.dqn_agent import DQNAgent
from src.environment.scheduler_env import TaskOffloadingEnv
import torch

def load_and_prepare_dag(dag_path):
    """Load and validate DAG"""
    G = nx.read_gml(dag_path)

    # Relabel nodes if 'label' exists
    if all('label' in data for _, data in G.nodes(data=True)):
        G = nx.relabel_nodes(G, {n: str(data['label']) for n, data in G.nodes(data=True)})

    # Add default attributes
    for node in G.nodes():
        if 'exec_time' not in G.nodes[node]:
            G.nodes[node]['exec_time'] = 1.0
        if 'comp_cost' not in G.nodes[node]:
            G.nodes[node]['comp_cost'] = 1000
    return G

def train_on_dag(dag_path, model_dir, num_nodes=8, episodes=1500):
    print(f"\n🚀 Training DQN on DAG: {dag_path.name}")
    G = load_and_prepare_dag(dag_path)
    env = TaskOffloadingEnv(G, num_nodes=num_nodes)

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=512,
        epsilon_decay=0.999,
        learning_rate=5e-5
    )

    best_makespan = float("inf")
    patience, no_improve = 200, 0

    for ep in range(episodes):
        state = env.reset()
        done, total_reward = False, 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        makespan = info.get('makespan', float("inf"))
        if makespan < best_makespan:
            best_makespan = makespan
            save_path = model_dir / f"{dag_path.stem}_best.pth"
            agent.save(save_path, env)
            no_improve = 0
            print(f"[EP {ep}] 🎯 New Best Makespan = {best_makespan:.2f}, Model Saved")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"⏹️ Early stopping at EP {ep} (no improvement)")
            break

    print(f"✅ Training Done | Best Makespan: {best_makespan:.2f}")
    return True


if __name__ == "__main__":
    dag_dir = Path("D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/data/Montage_dags")
    model_dir = Path("D:/Desktop Material/DAD_Computing/TaskOffloadingOptimization/results/Montage_Models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for dag_file in sorted(dag_dir.glob("*.gml")):
        train_on_dag(dag_file, model_dir, num_nodes=8, episodes=1500)
