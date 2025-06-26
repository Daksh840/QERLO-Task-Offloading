import gym
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any
from pathlib import Path

class TaskOffloadingEnv(gym.Env):
    def __init__(self, dag: nx.DiGraph, num_nodes: int = 8):
        super(TaskOffloadingEnv, self).__init__()

        if not isinstance(dag, nx.DiGraph):
            raise ValueError("Input must be a NetworkX DiGraph")
        try:
            self.dag = dag
            self.task_list = list(nx.topological_sort(dag))
            if not self.task_list:
                raise ValueError("Empty task list - check DAG structure")
        except nx.NetworkXUnfeasible:
            raise ValueError("Input graph contains cycles - must be a DAG")

        self.device_end_time = [0] * num_nodes
        self.num_nodes = num_nodes
        self.current_time = 0
        self.current_task_idx = 0
        self.node_available_time = [0.0] * num_nodes
        self.schedule = {}

        for task in self.task_list:
            if 'comp_cost' not in dag.nodes[task] and 'exec_time' not in dag.nodes[task]:
                raise ValueError(f"Task {task} missing both 'comp_cost' and 'exec_time'")

        self.action_space = gym.spaces.Discrete(num_nodes)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(num_nodes * 8,), dtype=np.float32)
        self.makespan_history = []
        self.utilization_history = []

    def reset(self) -> np.ndarray:
        self.device_end_time = [0] * self.num_nodes
        self.current_task_idx = 0
        self.node_available_time = [0.0] * self.num_nodes
        self.schedule = {}
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # Ensure we're within bounds of task list
        if self.current_task_idx >= len(self.task_list):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        task = self.task_list[self.current_task_idx]
        exec_time = self.dag.nodes[task].get('exec_time', 0)
        preds = len(list(self.dag.predecessors(task)))
        succs = len(list(self.dag.successors(task)))

        # Extract shared stats
        mean_node_time = np.mean(self.node_available_time)
        std_node_time = np.std(self.node_available_time)
        current_task = self.current_task_idx
        remaining_tasks = len(self.task_list) - self.current_task_idx

        state = []

        # Construct a state vector for each node (repeat global task stats per node)
        for node_time in self.node_available_time:
            state.extend([
                exec_time,           # Task exec time
                preds,               # Number of predecessors
                succs,               # Number of successors
                node_time,           # This node’s availability
                mean_node_time,      # Mean availability across all nodes
                std_node_time,       # Std deviation
                current_task,
                remaining_tasks
            ])

        return np.array(state, dtype=np.float32)


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_task_idx >= len(self.task_list):
            raise ValueError(f"[Step Error] Task index {self.current_task_idx} exceeds number of tasks {len(self.task_list)}")
        if not 0 <= action < self.num_nodes:
            raise ValueError(f"Invalid node index {action}")

        task = self.task_list[self.current_task_idx]
        try:
            comp_cost = float(self.dag.nodes[task].get('comp_cost', float(self.dag.nodes[task].get('exec_time', 10)) * 1000))
            duration = max(0.1, comp_cost / 1000)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid task duration for {task}: {str(e)}")

        ready_time = max([self.schedule.get(pred, {}).get('end_time', 0) for pred in self.dag.predecessors(task)] or [0])
        start_time = max(ready_time, self.node_available_time[action])
        end_time = start_time + duration

        self.device_end_time[action] = end_time
        self.node_available_time[action] = end_time

        self.schedule[task] = {
            'assigned_node': action,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'ready_time': ready_time
        }
        self.current_task_idx += 1

        done = self.current_task_idx >= len(self.task_list)
        reward = self._calculate_reward(end_time, done)
        next_state = self._get_state()
        info = {
            'task': task,
            'node': action,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'ready_time': ready_time,
            'makespan': max(self.device_end_time) if done else None,
            'reward': reward
        }
        if done:
            self._record_episode_metrics()
        return next_state, reward, done, info

    def _calculate_reward(self, end_time: float, done: bool) -> float:
        if done:
            makespan = max(self.device_end_time)
            load_std = np.std(self.device_end_time)
            return -makespan - 0.5 * load_std  # Penalize imbalance
        else:
            util = np.mean(self.node_available_time)
            imbalance = np.std(self.node_available_time)
            return -0.1 * end_time - 0.1 * imbalance + 0.05 * util

    def _record_episode_metrics(self):
        makespan = max(self.node_available_time)
        utilization = [t/makespan * 100 for t in self.node_available_time]
        self.makespan_history.append(makespan)
        self.utilization_history.append(utilization)
        print(f"[Env Log] ✅ Episode done. Makespan: {makespan:.2f}, Avg Util: {np.mean(utilization):.1f}%")

    def render(self, mode: str = 'human'):
        if mode == 'human':
            if self.current_task_idx > 0:
                last_task = self.task_list[self.current_task_idx - 1]
                last_assignment = self.schedule[last_task]
                print(f"Assigned Task {last_task} to Node {last_assignment['assigned_node']} | "
                      f"Start: {last_assignment['start_time']:.2f} | "
                      f"End: {last_assignment['end_time']:.2f}")
            print(f"Node Available Times: {[f'{t:.2f}' for t in self.node_available_time]}")
            if self.current_task_idx >= len(self.task_list):
                print(f"\nEpisode Complete | Makespan: {max(self.node_available_time):.2f}")

    def get_schedule(self) -> Dict:
        return self.schedule

    def get_metrics(self) -> Dict:
        return {
            'makespan': max(self.node_available_time) if self.schedule else 0,
            'utilization': [t/max(self.node_available_time)*100 for t in self.node_available_time] if self.schedule else [0]*self.num_nodes
        }

    def _get_current_task(self):
        if self.current_task_idx >= len(self.task_list):
            raise IndexError("No more tasks to schedule")
        return self.task_list[self.current_task_idx]

    def load_dag_from_gml(path: str) -> nx.DiGraph:
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"DAG file not found at {path}")
            G = nx.read_gml(path)
            if not isinstance(G, nx.DiGraph):
                raise ValueError("Input is not a directed graph")
            if all('label' in data for _, data in G.nodes(data=True)):
                G = nx.relabel_nodes(G, {n: str(data['label']) for n, data in G.nodes(data=True)})
            for node in G.nodes():
                attrs = G.nodes[node]
                if 'exec_time' not in attrs and 'comp_cost' not in attrs:
                    if 'LengthOnScheduledMachine' in attrs:
                        try:
                            exec_time = float(attrs['LengthOnScheduledMachine'])
                            G.nodes[node]['exec_time'] = exec_time
                        except Exception as e:
                            raise ValueError(f"Node {node} has invalid 'LengthOnScheduledMachine': {e}")
                    else:
                        raise ValueError(f"Node {node} missing both 'exec_time' and 'comp_cost' and no 'LengthOnScheduledMachine'")
            return G
        except Exception as e:
            raise ValueError(f"Failed to load DAG: {str(e)}")



"""if __name__ == "__main__":
    try:
        # Example usage
        dag_path = "C:/Users/palas/OneDrive/Desktop/DAD_Computing/TaskOffloadingOptimization/data/CyberShake_dags/CyberShake_1000.gml"
        G = TaskOffloadingEnv.load_dag_from_gml(dag_path)
        
        env = TaskOffloadingEnv(G, num_nodes=8)
        obs = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random policy
            obs, reward, done, info = env.step(action)
            env.render()
        
        print("\nFinal Schedule:")
        for task, assignment in env.get_schedule().items():
            print(f"Task {task}: Node {assignment['assigned_node']} | "
                  f"Start: {assignment['start_time']:.2f} | "
                  f"End: {assignment['end_time']:.2f}")
        
        metrics = env.get_metrics()
        print(f"\nMakespan: {metrics['makespan']:.2f}")
        print("Node Utilization:")
        for i, util in enumerate(metrics['utilization']):
            print(f"Node {i}: {util:.1f}%")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure:")
        print("1. The DAG file exists and is accessible")
        print("2. All nodes have 'exec_time' or 'comp_cost' attributes")
        print("3. The graph is a valid DAG (no cycles)")"""
