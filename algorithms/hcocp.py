"""
HCOCP: Heuristic Cost-Oriented Critical Path Scheduling
Prioritizes tasks on critical path while minimizing cost.
"""
import networkx as nx
import numpy as np
import numpy as np

class HCOCPScheduler:
    def __init__(self, num_edge_nodes=8, node_powers=None):
        self.num_edge_nodes = num_edge_nodes
        if node_powers is None:
            self.node_powers = [1.0 + 0.2*i for i in range(num_edge_nodes)]
        else:
            self.node_powers = node_powers

    def schedule(self, G):
        """
        Returns: dict {task: {assigned_node, start_time, end_time}}
        """
        # Compute critical path length (simplified: longest exec_time path)
        top_order = list(nx.topological_sort(G))
        longest_path = {n: float(G.nodes[n].get('exec_time', 1.0)) for n in top_order}
        for n in top_order:
            for succ in G.successors(n):
                longest_path[succ] = max(longest_path[succ],
                                         longest_path[n] + float(G.nodes[succ].get('exec_time', 1.0)))

        ready = [n for n in G.nodes() if G.in_degree(n) == 0]
        scheduled = {}
        available_time = [0.0] * self.num_edge_nodes

        while ready:
            # Prioritize tasks with highest critical-path length
            ready.sort(key=lambda x: longest_path[x], reverse=True)
            task = ready.pop(0)
            exec_time = float(G.nodes[task].get('exec_time', 1.0))

            best_cost, best_node, best_start = float('inf'), None, None
            for node in range(self.num_edge_nodes):
                start = max([scheduled[p]['end_time'] for p in G.predecessors(task)] or [0.0])
                start = max(start, available_time[node])
                end = start + exec_time

                energy = exec_time * self.node_powers[node]
                cost = end + 0.5 * energy  # Weighted: prioritize end-time & cost

                if cost < best_cost:
                    best_cost, best_node, best_start = cost, node, start

            scheduled[task] = {
                'assigned_node': best_node,
                'start_time': best_start,
                'end_time': best_start + exec_time
            }
            available_time[best_node] = scheduled[task]['end_time']

            for succ in G.successors(task):
                if all(pred in scheduled for pred in G.predecessors(succ)):
                    ready.append(succ)

        return scheduled

    def get_makespan(self, schedule):
        return max(info['end_time'] for info in schedule.values())

    def get_energy(self, schedule):
        return sum((info['end_time'] - info['start_time']) *
                   self.node_powers[info['assigned_node']] for info in schedule.values())

    def get_cost(self, schedule):
        """Add a cost metric (makespan + energy factor)"""
        makespan = self.get_makespan(schedule)
        energy = self.get_energy(schedule)
        return makespan + 0.5 * energy
