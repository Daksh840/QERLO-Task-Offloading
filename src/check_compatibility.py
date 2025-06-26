import sys
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import after path setup
from algorithms.evaluation.evaluator import SchedulerEvaluator, TaskOffloadingEnv
from algorithms.evaluation.evaluation_config import EvaluationConfig
import torch

def check():
    print("\n=== Starting Compatibility Check ===")
    config = EvaluationConfig()
    evaluator = SchedulerEvaluator(config)
    
    try:
        dag_path = next(Path(config.dag_folder).glob("*.gml"))
        model_path = next(Path(config.model_folder).glob("*.pth"))
        
        print(f"Testing with:\n- DAG: {dag_path}\n- Model: {model_path}")
        
        dag = evaluator._load_dag(str(dag_path))
        env = TaskOffloadingEnv(dag, num_nodes=config.num_nodes)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        print("\nModel Specifications:")
        print(f"Input dimensions: {checkpoint.get('input_dim', 'Not found')}")
        print(f"Output dimensions: {checkpoint.get('output_dim', 'Not found')}")
        
        print("\nEnvironment Specifications:")
        print(f"State space: {env.observation_space.shape[0]}")
        print(f"Action space: {env.action_space.n}")
        
        # Verify compatibility
        if checkpoint.get('input_dim') != env.observation_space.shape[0]:
            print("\n⚠️ CRITICAL: State dimension mismatch!")
            print(f"Model expects {checkpoint['input_dim']} but env provides {env.observation_space.shape[0]}")
        else:
            print("\n✅ State dimensions match")
            
        if checkpoint.get('output_dim') != env.action_space.n:
            print("⚠️ CRITICAL: Action space mismatch!")
            print(f"Model expects {checkpoint['output_dim']} but env provides {env.action_space.n}")
        else:
            print("✅ Action spaces match")
            
    except Exception as e:
        print(f"\n❌ Check failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check()
