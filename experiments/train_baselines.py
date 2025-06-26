import os
import sys
from pathlib import Path
import networkx as nx
from typing import Dict, Any

# Add project root to Python path once at the start
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def read_gml_fixed(path: Path) -> nx.DiGraph:
    """Robust GML file reader that handles different NetworkX versions"""
    try:
        return nx.read_gml(path)
    except Exception as e:
        print(f"Standard GML read failed, trying alternative methods: {e}")
        try:
            with open(path, 'r') as f:
                gml_str = f.read()
            return nx.parse_gml(gml_str)
        except Exception as e:
            print(f"Alternative GML parse failed: {e}")
            raise ValueError(f"Could not parse GML file at {path}")

def load_and_validate_dag(dag_path: Path) -> nx.DiGraph:
    """Load and validate DAG with comprehensive error checking"""
    if not dag_path.exists():
        available = [f.name for f in dag_path.parent.glob('*') if f.is_file()]
        raise FileNotFoundError(
            f"DAG file not found at {dag_path}. Available files: {available}"
        )
    
    try:
        G = nx.read_gml(dag_path)
        
        # Convert node labels to proper string IDs
        if all('label' in data for _, data in G.nodes(data=True)):
            label_mapping = {node: data['label'] for node, data in G.nodes(data=True)}
            G = nx.relabel_nodes(G, label_mapping)
        
        # Validate and set required attributes
        for node, data in G.nodes(data=True):
            if 'exec_time' not in data:
                if 'comp_cost' in data:
                    data['exec_time'] = data['comp_cost'] / 1000
                else:
                    raise ValueError(f"Node {node} missing both 'exec_time' and 'comp_cost' attributes")
            data['comp_cost'] = data.get('comp_cost', data['exec_time'] * 1000)
            
        return G
        
    except Exception as e:
        print(f"Invalid DAG structure: {e}")
        print("Node example from GML file should look like:")
        print("""
        node [
            id 0
            label "-8903467868813315685" 
            exec_time 65
        ]""")
        raise

def compare_algorithms(dag_path: str, num_edge_nodes: int = 3) -> Dict[str, Any]:
    """Compare scheduling algorithms with robust error handling"""
    try:
        abs_dag_path = project_root / dag_path
        
        print(f"\n{' Starting Algorithm Comparison ':=^50}")
        print(f"Loading DAG from: {abs_dag_path}")
        
        G = load_and_validate_dag(abs_dag_path)
        print(f"DAG loaded successfully with {len(G.nodes())} nodes")
        
        # Import algorithms here to ensure path is set
        from src.algorithms.heft import HEFTScheduler
        from src.algorithms.QIPSO import QIPSO_Scheduler
        from src.algorithms.First_come_first_server import schedule_fcfs
        
        print("\nRunning algorithm comparisons...")
        
        # Initialize schedulers
        heft_scheduler = HEFTScheduler(num_edge_nodes)
        qipso_scheduler = QIPSO_Scheduler(
            graph=G,
            num_edge_nodes=num_edge_nodes,
            num_particles=30,
            max_iter=50
        )
        
        # Run algorithms and safely extract makespan values
        def get_makespan(result):
            if isinstance(result, (int, float)):
                return float(result)
            elif isinstance(result, dict):
                return float(result.get('makespan', result.get('end_time', 0)))
            elif isinstance(result, tuple):
                return float(result[1])  # Assume makespan is second element
            return 0.0
        
        results = {
            'FCFS': get_makespan(schedule_fcfs(G, num_edge_nodes)),
            'HEFT': get_makespan(heft_scheduler.schedule(G)),
            'QIPSO': get_makespan(qipso_scheduler.run_optimization())
        }
        
        print("\n=== Results ===")
        for algo, makespan in results.items():
            print(f"{algo:<5}: {makespan:.2f} time units")
            
        return results
        
    except Exception as e:
        print(f"\n{' Comparison Failed ':=^50}")
        print(f"Error: {str(e)}")
        print("\nDebugging Info:")
        print(f"Python path: {sys.path}")
        print(f"DAG exists: {abs_dag_path.exists() if 'abs_dag_path' in locals() else 'N/A'}")
        print(f"Graph type: {type(G) if 'G' in locals() else 'Not loaded'}")
        raise

if __name__ == "__main__":
    # Configuration
    dag_file = "data/processed/part.10.gml"  # Relative to project root
    num_nodes = 3
    
    try:
        results = compare_algorithms(dag_file, num_nodes)
    except Exception as e:
        print("\nTroubleshooting Tips:")
        print("1. Verify the DAG file exists at data/processed/part.0.gml")
        print("2. Check all node attributes in the GML file")
        print("3. Try loading the GML file manually with nx.read_gml()")
        print("4. Verify all scheduler implementations are up to date")
        sys.exit(1)
