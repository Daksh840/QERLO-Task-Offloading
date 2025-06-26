import os
import csv
import chardet
import pandas as pd
import networkx as nx
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Union
import subprocess

class DAGConverter:
    """Utility class for converting various file formats to DAGs in GML format."""
    
    def __init__(self):
        pass

    # --------------------------
    # Core Conversion Functions
    # --------------------------

    def convert_parquet_to_gml(self, parquet_path: str, gml_path: str, 
                             id_col: str = 'id', 
                             time_col: str = 'runtime',
                             parents_col: str = 'parents') -> None:
        """
        Convert parquet file to GML format.
        
        Args:
            parquet_path: Path to input parquet file
            gml_path: Path to output GML file
            id_col: Column name for task IDs
            time_col: Column name for execution time
            parents_col: Column name for parent tasks
        """
        try:
            df = pd.read_parquet(parquet_path)
            print(f"Converting: {parquet_path}")
            print("Sample data:\n", df.head())

            G = nx.DiGraph()

            # Add nodes
            for _, row in df.iterrows():
                task_id = str(row[id_col])
                exec_time = row.get(time_col, 10)  # Default 10 if not specified
                G.add_node(task_id, exec_time=exec_time)

            # Add edges
            for _, row in df.iterrows():
                task_id = str(row[id_col])
                parents = row.get(parents_col, [])
                
                # Handle different parent formats
                if isinstance(parents, str):
                    parents = parse_dependencies(parents)
                elif isinstance(parents, (np.ndarray, pd.Series)):
                    parents = list(parents)
                
                for parent in parents:
                    if pd.notna(parent):
                        parent_str = str(parent).strip()
                        if parent_str in G.nodes:
                            G.add_edge(parent_str, task_id)

            # Save GML
            nx.write_gml(G, gml_path)
            print(f"✅ Saved: {gml_path}")
            
        except Exception as e:
            print(f"❌ Failed to convert {parquet_path}: {e}")
            raise

    def convert_csv_to_gml(self, csv_path: str, gml_path: str, 
                         id_col: str = 'TaskID',
                         time_col: str = 'CPUNeed_Claimed',
                         imm_dep_col: str = 'SuccessorsImediate',
                         non_imm_dep_col: str = 'SuccessorsNotImmediate') -> None:
        """
        Convert CSV file to GML format with cycle handling.
        
        Args:
            csv_path: Path to input CSV file
            gml_path: Path to output GML file
            id_col: Column name for task IDs
            time_col: Column name for execution time
            imm_dep_col: Column for immediate dependencies
            non_imm_dep_col: Column for non-immediate dependencies
        """
        try:
            # Detect encoding
            encoding = self.detect_encoding(csv_path)
            print(f"Detected encoding: {encoding}")

            tasks = {}
            edges = set()
            edge_pairs = set()

            with open(csv_path, 'r', encoding=encoding) as csv_file:
                # Handle BOM if present
                start_pos = csv_file.tell()
                if csv_file.read(1) != '\ufeff':
                    csv_file.seek(start_pos)
                
                reader = csv.DictReader(csv_file)
                reader.fieldnames = [self.clean_column_name(name) for name in reader.fieldnames]
                print("Processed columns:", reader.fieldnames)
                
                for row in reader:
                    try:
                        task_id = row[id_col].strip()
                        if not task_id:
                            continue
                            
                        # Store task attributes
                        tasks[task_id] = {
                            'exec_time': row.get(time_col, '10').strip(),
                            'OwnerJobID': row.get('OwnerJobID', '').strip(),
                            'Priority': row.get('PriorityNo', '').strip()
                        }
                        
                        # Process dependencies
                        for dep_type, col in [('imm', imm_dep_col), 
                                            ('non-imm', non_imm_dep_col)]:
                            for successor in self.parse_dependencies(row.get(col, '')):
                                edge_key = (task_id, successor)
                                if edge_key not in edge_pairs:
                                    edge_pairs.add(edge_key)
                                    edges.add((task_id, successor, dep_type))
                                    
                    except Exception as e:
                        print(f"Skipping row due to error: {str(e)}")
                        continue

            # Remove cycles if any
            edges = list(edges)
            edges = self.remove_cycles(edges)
            
            # Create and save graph
            self._create_gml_from_edges(tasks, edges, gml_path)
            
        except Exception as e:
            print(f"❌ Failed to convert {csv_path}: {e}")
            raise

    # --------------------------
    # Batch Processing
    # --------------------------

    def batch_convert_parquet_to_gml(self, input_dir: str, output_dir: str, 
                                   id_col: str = 'id',
                                   time_col: str = 'runtime',
                                   parents_col: str = 'parents') -> None:
        """
        Convert all parquet files in a directory to GML format.
        
        Args:
            input_dir: Directory containing parquet files
            output_dir: Directory to save GML files
            id_col: Column name for task IDs
            time_col: Column name for execution time
            parents_col: Column name for parent tasks
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = os.listdir(input_dir)
        print(f"Scanning: {input_dir} | Found: {len(files)} files")

        for file in files:
            if file.lower().endswith(".parquet"):
                in_path = os.path.join(input_dir, file)
                out_path = os.path.join(output_dir, file.replace(".parquet", ".gml"))
                try:
                    print(f"\nProcessing: {file}")
                    self.convert_parquet_to_gml(
                        in_path, out_path, 
                        id_col=id_col,
                        time_col=time_col,
                        parents_col=parents_col
                    )
                except Exception as e:
                    print(f"❌ Failed for {file}: {e}")

    # --------------------------
    # Utility Methods
    # --------------------------

    def remove_cycles(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Identify and break cycles in the dependency graph.
        
        Args:
            edges: List of (source, target, dependency_type) tuples
            
        Returns:
            List of edges with cycles removed
        """
        G = nx.DiGraph()
        G.add_edges_from([(src, tgt) for src, tgt, _ in edges])
        
        try:
            # Test if graph is already a DAG
            list(nx.topological_sort(G))
            return edges
        except nx.NetworkXUnfeasible:
            print("Warning: Found cycles in task dependencies - attempting to break them")
            
            cycles = list(nx.simple_cycles(G))
            print(f"Found {len(cycles)} cycles")
            
            edges_to_remove = set()
            for cycle in cycles:
                print(f"Cycle detected: {' → '.join(cycle)}")
                edge_to_remove = (cycle[-1], cycle[0])
                edges_to_remove.add(edge_to_remove)
            
            filtered_edges = [
                (src, tgt, dep_type) 
                for src, tgt, dep_type in edges 
                if (src, tgt) not in edges_to_remove
            ]
            
            print(f"Removed {len(edges) - len(filtered_edges)} edges to break cycles")
            return filtered_edges

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding automatically."""
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)  # Read first 10KB to guess encoding
        result = chardet.detect(rawdata)
        return result['encoding'] if result['encoding'] else 'utf-8'

    def clean_column_name(self, name: str) -> str:
        """Remove BOM and other artifacts from column names."""
        return name.replace('\ufeff', '').replace('\x00', '').strip()

    def parse_dependencies(self, dep_str: str) -> List[str]:
        """
        Robust dependency parser that handles various formats.
        
        Args:
            dep_str: Dependency string from CSV
            
        Returns:
            List of parsed task IDs
        """
        if not dep_str or str(dep_str).strip().lower() in ['none', 'null', 'na', '']:
            return []
        return [tid.strip() for tid in str(dep_str).replace(';', ',').split(',') if tid.strip()]

    def _create_gml_from_edges(self, tasks: Dict, edges: List[Tuple], gml_path: str) -> None:
        """
        Internal method to create GML file from tasks and edges.
        
        Args:
            tasks: Dictionary of task attributes
            edges: List of (source, target, type) edges
            gml_path: Output file path
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for task_id, attrs in tasks.items():
            G.add_node(task_id, **attrs)
        
        # Add edges
        for src, tgt, dep_type in edges:
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt, type=dep_type)
        
        # Save to GML
        nx.write_gml(G, gml_path)
        print(f"\n✅ Successfully created {gml_path}")
        print(f"Stats: {len(tasks)} tasks, {len(edges)} edges")

    # --------------------------
    # Main Conversion Interface
    # --------------------------

    def convert_to_gml(self, input_path: str, output_path: str, 
                      input_type: Optional[str] = None, **kwargs) -> None:
        """
        Main conversion interface that auto-detects file type.
        
        Args:
            input_path: Path to input file/directory
            output_path: Path to output file/directory
            input_type: Optional file type ('parquet' or 'csv')
            **kwargs: Additional format-specific parameters
        """
        if input_type is None:
            if input_path.lower().endswith('.parquet'):
                input_type = 'parquet'
            elif input_path.lower().endswith('.csv'):
                input_type = 'csv'
            elif os.path.isdir(input_path):
                return self.batch_convert_parquet_to_gml(input_path, output_path, **kwargs)
            else:
                raise ValueError("Could not determine input file type")

        if input_type == 'parquet':
            if os.path.isdir(input_path):
                self.batch_convert_parquet_to_gml(input_path, output_path, **kwargs)
            else:
                self.convert_parquet_to_gml(input_path, output_path, **kwargs)
        elif input_type == 'csv':
            self.convert_csv_to_gml(input_path, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {input_type}")

# --------------------------
# Command Line Interface
# --------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert task files to GML format')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('output', help='Output file or directory path')
    parser.add_argument('--type', choices=['auto', 'parquet', 'csv'], default='auto',
                       help='Input file type (default: auto-detect)')
    parser.add_argument('--id-col', default='id', help='Column name for task IDs')
    parser.add_argument('--time-col', default='runtime', help='Column name for execution time')
    parser.add_argument('--parents-col', default='parents', help='Column name for parent tasks')
    
    args = parser.parse_args()
    
    # Install chardet if not available
    try:
        import chardet
    except ImportError:
        print("Installing chardet for encoding detection...")
        subprocess.check_call(["python", "-m", "pip", "install", "chardet"])
        import chardet
    
    converter = DAGConverter()
    
    input_type = None if args.type == 'auto' else args.type
    converter.convert_to_gml(
        args.input,
        args.output,
        input_type=input_type,
        id_col=args.id_col,
        time_col=args.time_col,
        parents_col=args.parents_col
    )

if __name__ == "__main__":
    main()
    #Convert single parquet file:
    '''from dag_utils import DAGConverter
    converter = DAGConverter()
    converter.convert_parquet_to_gml("input.parquet", "output.gml")'''

    #Convert CSV with custom columns:
    '''converter.convert_csv_to_gml(
    "tasks.csv", 
    "tasks.gml",
    id_col="TaskID",
    time_col="Duration",
    imm_dep_col="ImmediateDeps"
)'''

    #Batch convert parquet files:
    '''converter.batch_convert_parquet_to_gml(
    "input_directory",
    "output_directory",
    id_col="task_id",
    time_col="exec_time"
)'''
    

    
