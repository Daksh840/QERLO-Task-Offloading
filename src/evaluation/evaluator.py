import os
import csv
import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import logging
import sys
from time import time
from datetime import datetime

# Get the project root (TaskOffloadingOptimization/)
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to reach the project root
sys.path.append(str(project_root))

# Now import your project modules
from src.environment.scheduler_env import TaskOffloadingEnv
from src.algorithms.dqn_agent import DQNAgent
from src.algorithms.QIPSO import QIPSO_Scheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@dataclass
class EvaluationConfig:
    dag_folder: str = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/CyberShake_dags"
    model_folder: str = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/results/CyberShake_Models"
    output_dir: str = "evaluation_results"
    num_nodes: int = 8
    num_eval_episodes: int = 10  # Number of evaluation episodes
    qipso_params: Dict[str, Any] = None
    dqn_params: Dict[str, Any] = None
    
    def __post_init__(self):
        self.qipso_params = self.qipso_params or {
            'num_particles': 20, 
            'max_iter': 50,
            'early_stop_patience': 10,
            'cognitivee_weight': 1.7,
            'social_weight': 1.7,
            'mutation_prob': 0.3
        }
        self.dqn_params = self.dqn_params or {
            'episodes': 200,
            'batch_size': 64,
            'gamma': 0.99,
            'epsilon_decay': 0.995
        }

class SchedulerEvaluator:
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._setup_directories()
        self.results = []
        
    def _setup_directories(self) -> None:
        """Ensure all required directories exist."""
        Path(self.config.dag_folder).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Debug print directory contents
        logger.info("\n=== Directory Verification ===")
        logger.info(f"DAG folder: {Path(self.config.dag_folder).absolute()}")
        logger.info(f"Model folder: {Path(self.config.model_folder).absolute()}")
        logger.info(f"Files in DAG folder: {list(Path(self.config.dag_folder).glob('*.gml'))}")
        logger.info(f"Files in Model folder: {list(Path(self.config.model_folder).glob('*.pth'))}")
        logger.info("=============================")

    def _load_dag(self, dag_path: str) -> nx.DiGraph:
        """Improved DAG loading with validation."""
        try:
            dag = nx.read_gml(dag_path)
            if not isinstance(dag, nx.DiGraph):
                raise ValueError("Input must be a NetworkX DiGraph")
            if not nx.is_directed_acyclic_graph(dag):
                raise ValueError("Graph contains cycles")
                
            # Convert node labels to strings if needed
            if all('label' in data for _, data in dag.nodes(data=True)):
                dag = nx.relabel_nodes(dag, {n: str(data['label']) for n, data in dag.nodes(data=True)})
                
            # Validate and set default attributes
            for node in dag.nodes():
                if 'exec_time' not in dag.nodes[node] and 'comp_cost' not in dag.nodes[node]:
                    raise ValueError(f"Node {node} missing both exec_time and comp_cost")
                dag.nodes[node]['exec_time'] = dag.nodes[node].get('exec_time', 
                    dag.nodes[node].get('comp_cost', 1000) / 1000)
                    
            return dag
        except Exception as e:
            logger.error(f"Failed to load DAG {dag_path}: {str(e)}")
            raise

    def _run_evaluation_episode(self, env: TaskOffloadingEnv, agent: DQNAgent = None) -> Dict:
        """Run a single evaluation episode and return metrics."""
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, epsilon=0) if agent else env.action_space.sample()
            state, _, done, _ = env.step(action)
            
        metrics = env.get_metrics()
        return {
            'makespan': metrics['makespan'],
            'utilization': metrics['utilization'],
            'schedule': env.get_schedule()
        }

    def evaluate_dqn(self, dag_path: str, model_path: str) -> Dict[str, Any]:
        """Comprehensive DQN evaluation with multiple episodes."""
        try:
            dag = self._load_dag(dag_path)
            env = TaskOffloadingEnv(dag, num_nodes=self.config.num_nodes)
            
            # Load agent
            checkpoint = torch.load(model_path)
            agent = DQNAgent(
                state_size=checkpoint['input_dim'],
                action_size=checkpoint['output_dim'],
                **self.config.dqn_params
            )
            agent.load(model_path)
            
            # Run evaluation episodes
            results = []
            for _ in range(self.config.num_eval_episodes):
                results.append(self._run_evaluation_episode(env, agent))
                
            # Calculate aggregate metrics
            avg_makespan = np.mean([r['makespan'] for r in results])
            avg_utilization = np.mean([np.mean(r['utilization']) for r in results])
            
            return {
                'algorithm': 'DQN',
                'dag': Path(dag_path).stem,
                'avg_makespan': avg_makespan,
                'avg_utilization': avg_utilization,
                'node_utilization': np.mean([r['utilization'] for r in results], axis=0).tolist(),
                'episode_results': results
            }
            
        except Exception as e:
            logger.error(f"DQN evaluation failed: {str(e)}")
            return None

    def evaluate_qipso(self, dag_path: str) -> Dict[str, Any]:
        """Enhanced QIPSO evaluation with metrics collection."""
        try:
            dag = self._load_dag(dag_path)
            qipso = QIPSO_Scheduler(
                graph=dag,
                num_edge_nodes=self.config.num_nodes,
                **self.config.qipso_params
            )
            
            start_time = time()
            schedule, makespan = qipso.run_optimization()
            runtime = time() - start_time
            
            # Calculate utilization
            node_times = [0] * self.config.num_nodes
            for task, assignment in schedule.items():
                node = assignment['assigned_node']
                node_times[node] += assignment['end_time'] - assignment['start_time']
                
            utilization = [(t/makespan)*100 for t in node_times] if makespan > 0 else [0]*self.config.num_nodes
            
            return {
                'algorithm': 'QIPSO',
                'dag': Path(dag_path).stem,
                'avg_makespan': makespan,
                'avg_utilization': np.mean(utilization),
                'node_utilization': utilization,
                'runtime': runtime,
                'schedule': schedule
            }
            
        except Exception as e:
            logger.error(f"QIPSO evaluation failed: {str(e)}")
            return None

    def evaluate_baselines(self, dag_path: str) -> List[Dict[str, Any]]:
        """Evaluate baseline policies for comparison."""
        dag = self._load_dag(dag_path)
        policies = {
            'Random': None,
            # Add other baselines as needed
        }
        
        results = []
        for policy_name, _ in policies.items():
            env = TaskOffloadingEnv(dag, num_nodes=self.config.num_nodes)
            policy_results = []
            
            for _ in range(self.config.num_eval_episodes):
                policy_results.append(self._run_evaluation_episode(env))
                
            avg_makespan = np.mean([r['makespan'] for r in policy_results])
            avg_utilization = np.mean([np.mean(r['utilization']) for r in policy_results])
            
            results.append({
                'algorithm': policy_name,
                'dag': Path(dag_path).stem,
                'avg_makespan': avg_makespan,
                'avg_utilization': avg_utilization,
                'node_utilization': np.mean([r['utilization'] for r in policy_results], axis=0).tolist()
            })
            
        return results

    def batch_evaluate(self) -> None:
        """Run comprehensive evaluation across all DAGs."""
        logger.info("Starting batch evaluation")
        
        # Verify model/environment compatibility
        self._verify_compatibility()
        
        # Process each DAG
        dag_files = list(Path(self.config.dag_folder).glob("*.gml"))
        if not dag_files:
            raise FileNotFoundError(f"No DAG files found in {self.config.dag_folder}")
            
        for dag_file in dag_files:
            dag_results = []
            
            # Evaluate QIPSO
            qipso_result = self.evaluate_qipso(str(dag_file))
            if qipso_result:
                dag_results.append(qipso_result)
                
            # Evaluate DQN if model exists
            model_file = f"dqn_model_{dag_file.stem}.pth"
            model_path = Path(self.config.model_folder) / model_file
            if model_path.exists():
                dqn_result = self.evaluate_dqn(str(dag_file), str(model_path))
                if dqn_result:
                    dag_results.append(dqn_result)
                    
            # Evaluate baselines
            dag_results.extend(self.evaluate_baselines(str(dag_file)))
            
            if dag_results:
                self.results.extend(dag_results)
                self._save_dag_results(dag_file.stem, dag_results)
                self._plot_dag_results(dag_file.stem, dag_results)
                
        # Save final aggregated results
        self._save_final_results()

    def _verify_compatibility(self) -> None:
        """Check model/environment compatibility."""
        try:
            sample_dag = next(Path(self.config.dag_folder).glob("*.gml"), None)
            sample_model = next(Path(self.config.model_folder).glob("*.pth"), None)
            
            if sample_dag and sample_model:
                dag = self._load_dag(str(sample_dag))
                env = TaskOffloadingEnv(dag, num_nodes=self.config.num_nodes)
                checkpoint = torch.load(sample_model)
                
                logger.info("\n=== Model/Environment Compatibility ===")
                logger.info(f"Model expects input: {checkpoint.get('input_dim')}")
                logger.info(f"Env provides state: {env.observation_space.shape[0]}")
                logger.info(f"Model action size: {checkpoint.get('output_dim')}")
                logger.info(f"Env action space: {env.action_space.n}")
                
                if checkpoint.get('input_dim') != env.observation_space.shape[0]:
                    logger.warning("State dimension mismatch!")
                if checkpoint.get('output_dim') != env.action_space.n:
                    logger.warning("Action space mismatch!")
        except Exception as e:
            logger.warning(f"Compatibility check failed: {str(e)}")

    def _save_dag_results(self, dag_name: str, results: List[Dict]) -> None:
        """Save results for a single DAG."""
        csv_path = Path(self.config.output_dir) / f"{dag_name}_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved results for {dag_name} to {csv_path}")

    def _plot_dag_results(self, dag_name: str, results: List[Dict]) -> None:
        """Generate visualization for a single DAG's results."""
        plt.figure(figsize=(12, 6))
        
        # Makespan comparison
        plt.subplot(1, 2, 1)
        algorithms = [r['algorithm'] for r in results]
        makespans = [r['avg_makespan'] for r in results]
        plt.bar(algorithms, makespans, color=['blue', 'orange', 'green'])
        plt.title(f"Makespan Comparison\n{dag_name}")
        plt.ylabel("Makespan")
        
        # Utilization comparison
        plt.subplot(1, 2, 2)
        for i, result in enumerate(results):
            plt.plot(result['node_utilization'], label=result['algorithm'], 
                   marker='o', linestyle='--', alpha=0.7)
        plt.title("Node Utilization")
        plt.xlabel("Node ID")
        plt.ylabel("Utilization (%)")
        plt.legend()
        
        plt.tight_layout()
        plot_path = Path(self.config.output_dir) / f"{dag_name}_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization for {dag_name} to {plot_path}")

    def _save_final_results(self) -> None:
        """Save aggregated results across all DAGs."""
        if not self.results:
            logger.warning("No results to save")
            return
            
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_csv_path = Path(self.config.output_dir) / f"all_results_{timestamp}.csv"
        with open(raw_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
            
        # Save summary statistics
        summary_csv_path = Path(self.config.output_dir) / f"summary_{timestamp}.csv"
        algorithms = list(set(r['algorithm'] for r in self.results))
        summary = []
        
        for algo in algorithms:
            algo_results = [r for r in self.results if r['algorithm'] == algo]
            summary.append({
                'algorithm': algo,
                'avg_makespan': np.mean([r['avg_makespan'] for r in algo_results]),
                'std_makespan': np.std([r['avg_makespan'] for r in algo_results]),
                'avg_utilization': np.mean([r['avg_utilization'] for r in algo_results]),
                'num_dags': len(algo_results)
            })
            
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
            
        logger.info(f"\n✅ Saved final results to:\n{raw_csv_path}\n{summary_csv_path}")

if __name__ == "__main__":
    # Example configuration
    config = EvaluationConfig(
        dag_folder="C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/processed",
        model_folder="C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/results/models",
        output_dir="evaluation_results",
        num_nodes=8,
        num_eval_episodes=5,
        qipso_params={
            'num_particles': 30,
            'max_iter': 100,
            'early_stop_patience': 20
        },
        dqn_params={
            'episodes': 300,
            'batch_size': 128,
            'gamma': 0.95,
            'epsilon_decay': 0.998
        }
    )
    
    evaluator = SchedulerEvaluator(config)
    evaluator.batch_evaluate()
