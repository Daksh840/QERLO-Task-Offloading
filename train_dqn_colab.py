# train_dqn_colab.py
import os
import argparse
import networkx as nx
from pathlib import Path
from src.algorithms.dqn_agent import DQNAgent
from src.environment.scheduler_env import TaskOffloadingEnv
import torch
import numpy as np
import json
import time

def load_and_prepare_dag(dag_path):
    G = nx.read_gml(str(dag_path))
    # relabel if 'label' attribute exists
    if all('label' in data for _, data in G.nodes(data=True)):
        G = nx.relabel_nodes(G, {n: str(data['label']) for n, data in G.nodes(data=True)})
    # fill defaults
    for node in G.nodes():
        if 'exec_time' not in G.nodes[node]:
            G.nodes[node]['exec_time'] = 1.0
        else:
            G.nodes[node]['exec_time'] = float(G.nodes[node]['exec_time'])
    return G

def shaped_reward(info):
    """
    Example: reward shaped using makespan and energy. Smaller makespan -> larger reward.
    We'll return per-step reward, but as environment returns reward, this function can be used
    as additional shaping (not used blindly).
    """
    # If your env already returns reward, you might combine them here.
    makespan = info.get('makespan', None)
    energy = info.get('energy', None)
    # return simple 0 if nothing
    if makespan is None:
        return 0.0
    # Negative penalty; scaled
    reward = -makespan * 0.001
    if energy is not None:
        reward -= 0.0005 * energy
    # Or you can invert: reward = base_reward - alpha*makespan...
    return reward

def train_on_dag(dag_path, model_dir, episodes=3000,
                 batch_size=1024, memory_size=100000,
                 epsilon_decay=0.997, learning_rate=3e-4,
                 device=None, early_stop_patience=400):
    print(f"\n=== TRAINING on {dag_path.name} ===")
    G = load_and_prepare_dag(dag_path)
    env = TaskOffloadingEnv(G, num_nodes=8)

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        batch_size=batch_size,
        epsilon_decay=epsilon_decay,
        learning_rate=learning_rate,
        memory_size=memory_size,
        device=device
    )

    best_makespan = float("inf")
    no_improve = 0

    history = []
    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            # augment reward with shaping if you want
            shaped = shaped_reward(info)
            combined_reward = reward + shaped
            agent.remember(state, action, combined_reward, next_state, done)
            loss = agent.replay()
            state = next_state
            total_reward += combined_reward

        # get makespan from info (last step)
        makespan = info.get('makespan', float('inf'))
        history.append({'episode': ep, 'makespan': makespan, 'reward': total_reward, 'epsilon': agent.epsilon})

        # Checkpoint if improved
        if makespan < best_makespan:
            best_makespan = makespan
            no_improve = 0
            save_path = model_dir / f"{dag_path.stem}_best.pth"
            agent.save(str(save_path), env)
            print(f"[EP {ep}] 🎯 New best makespan = {best_makespan:.2f}. Saved: {save_path.name}")
        else:
            no_improve += 1

        # Early stop
        if no_improve >= early_stop_patience:
            print(f"⏹️ Early stopping at ep {ep} (no improve in {early_stop_patience} eps).")
            break

        # periodic checkpoint
        if ep % 250 == 0:
            save_path = model_dir / f"{dag_path.stem}_ep{ep}.pth"
            agent.save(str(save_path), env)
            print(f"[EP {ep}] Checkpoint saved: {save_path.name}")

        # log
        if ep % 50 == 0:
            print(f"[EP {ep}] makespan={makespan:.2f} | total_reward={total_reward:.2f} | eps={agent.epsilon:.4f}")

    # final save
    final_path = model_dir / f"{dag_path.stem}_final.pth"
    agent.save(str(final_path), env)
    agent.plot_metrics(str(model_dir))
    # save history
    hist_path = model_dir / f"{dag_path.stem}_train_history.json"
    with open(hist_path, 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f"✅ Finished training {dag_path.name} | best_makespan={best_makespan:.2f}")
    return {'dag': dag_path.name, 'best_makespan': best_makespan, 'episodes_trained': ep, 'model_path': str(final_path)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag_dir", type=str, default="data/Montage_dags")
    parser.add_argument("--model_dir", type=str, default="results/Montage_Models_Colab")
    parser.add_argument("--episodes", type=int, default=3000)
    args = parser.parse_args()

    dag_dir = Path(args.dag_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for dag_file in sorted(dag_dir.glob("*.gml")):
        try:
            res = train_on_dag(dag_file, model_dir, episodes=args.episodes)
            results.append(res)
        except Exception as ex:
            print(f"[ERROR] training failed for {dag_file.name}: {ex}")

    # Save a summary
    with open(model_dir / "training_summary.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    print("All training done. Summary saved.")
