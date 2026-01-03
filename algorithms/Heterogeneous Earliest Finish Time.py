import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gml("results/tasks_graph_final.gml")

schedule_fcfs = schedule_fcfs(G, num_edge_nodes=3)
schedule_heft = schedule_heft(G, num_edge_nodes=3)
# Later: schedule_pso, schedule_qipso

