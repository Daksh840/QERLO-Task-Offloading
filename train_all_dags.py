import os
import networkx as nx
from pathlib import Path
from src.algorithms.QIPSO import QIPSO_Scheduler
from src.algorithms.dqn_agent import DQNAgent
from src.environment.scheduler_env import TaskOffloadingEnv
import torch

def load_and_prepare_dag(dag_path):
    """Load and prepare DAG with comprehensive validation"""
    try:
        # Load the DAG
        G = nx.read_gml(dag_path)
            
        # Convert node labels to strings if they exist
        if all('label' in data for _, data in G.nodes(data=True)):
            G = nx.relabel_nodes(G, {n: str(data['label']) for n, data in G.nodes(data=True)})
            
            # Validate required attributes
        for node in G.nodes():
            if 'comp_cost' not in G.nodes[node] and 'exec_time' not in G.nodes[node]:
                G.nodes[node]['comp_cost'] = 1000  # Default value
                G.nodes[node]['exec_time'] = 1.0   # Default value
            elif 'exec_time' not in G.nodes[node]:
                G.nodes[node]['exec_time'] = G.nodes[node]['comp_cost'] / 1000
            
        return G
    except Exception as e:
        raise ValueError(f"Failed to process {dag_path}: {str(e)}")

def train_on_dag(dag_path, num_nodes=8, episodes=1500):
    print(f"\n🚀 Training on DAG: {os.path.basename(dag_path)}")
        
    try:
        models_dir = Path("C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/results/Montage_Models")
        models_dir.mkdir(parents=True, exist_ok=True)
        G = load_and_prepare_dag(dag_path)
        env = TaskOffloadingEnv(dag=G, num_nodes=num_nodes)

        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            batch_size=512,
            epsilon_decay=0.999,
            learning_rate=5e-5
        )

        best_makespan = float("inf")
        no_improve_counter = 0
        patience = 100  # stop after 300 episodes without improvement

        for episode in range(episodes):
            state = env.reset()
            if len(env.task_list) == 0:
                raise ValueError(f"DAG {dag_path.name} is empty or has no valid tasks.")

            done = False
            total_reward = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward

            current_makespan = info.get('makespan', float("inf"))

            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_model_path = str(models_dir / f"{Path(dag_path).stem}_best.pth")
                agent.save(best_model_path, env)
                no_improve_counter = 0
                print(f"[Episode {episode}] 🎯 New Best Makespan: {best_makespan:.2f} - Model Saved")
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                print(f"⏹️ Early stopping at episode {episode} (no improvement in {patience} episodes)")
                break


            if episode % 50 == 0:
                agent.save(str(models_dir / f"{Path(dag_path).stem}_ep{episode}.pth"), env)
                print(f"[Episode {episode}] Model checkpoint saved.")

            # Final save
        agent.save(str(models_dir / f"{Path(dag_path).stem}_final.pth"), env)
        print(f"✅ Training completed for {os.path.basename(dag_path)} | Best Makespan: {best_makespan:.2f}")
        return True

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(dag_path)}: {str(e)}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    dag_dir = Path("C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/Montage_dags")
    num_nodes = 8  # Must match your environment's num_nodes
        
    # Verify DAG folder exists
    if not dag_dir.exists():
        raise FileNotFoundError(f"DAG directory not found: {dag_dir}")
        
    # Process each DAG
    success_count = 0
    for dag_file in sorted(dag_dir.glob("*.gml")):
        if train_on_dag(dag_file, num_nodes, episodes=1500):
            success_count += 1
        
    print(f"\nTraining completed for {success_count}/{len(list(dag_dir.glob('*.gml')))} DAGs")

