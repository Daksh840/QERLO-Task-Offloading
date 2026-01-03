# generate_final_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# Load Results
# ==============================
csv_path = Path(r"D:\Desktop Material\DAD_Computing\TaskOffloadingOptimization\results\CyberShake_Outputs\scheduler_results.csv")
df = pd.read_csv(csv_path)

# Ensure algorithms are in the right order
algos = ['GA','PSO','HEFT','MOHEFT','QIPSO','DQN','HCOCP','QERLO']
metrics = ['makespan','energy','runtime_sec','cost']

# Create output folder
out_dir = Path(r"D:\Desktop Material\DAD_Computing\TaskOffloadingOptimization\results\results\CyberShake_Outputs\plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# Function: Plot per metric
# ==============================
def plot_metric(metric, ylabel, fname):
    plt.figure(figsize=(12,6))
    dag_names = df['dag'].unique().tolist()
    x = np.arange(len(dag_names))
    width = 0.1

    for i, algo in enumerate(algos):
        sel = df[df['algorithm'] == algo]
        vals = []
        for dag in dag_names:
            row = sel[sel['dag']==dag]
            if row.empty:
                vals.append(np.nan)
            else:
                vals.append(float(row[metric].values[0]))
        vals = np.array(vals)

        xs = x + i*width
        if algo == "QERLO":
            bars = plt.bar(xs, vals, width, label=algo, color="red", alpha=0.7, edgecolor="black")
        else:
            bars = plt.bar(xs, vals, width, label=algo)

        # Annotate values
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x()+bar.get_width()/2., h+(0.02*h if h!=0 else 0.01),
                         f"{h:.2f}", ha='center', va='bottom', fontsize=7, rotation=90)

    plt.xticks(x + width*len(algos)/2, [d.replace(".gml","") for d in dag_names], rotation=45)
    plt.ylabel(ylabel)
    plt.title(f"{metric.capitalize()} Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / f"{fname}.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Saved {out_path}")

# Generate plots
plot_metric("makespan", "Makespan (time units)", "makespan_comparison")
plot_metric("energy", "Energy Consumption", "energy_comparison")
plot_metric("runtime_sec", "Runtime (s)", "runtime_comparison")
plot_metric("cost", "Cost (composite units)", "cost_comparison")

# ==============================
# Generate LaTeX table
# ==============================
def make_latex_table(df, metrics):
    dag_names = df['dag'].unique().tolist()
    table = []

    for dag in dag_names:
        row = [dag.replace(".gml","")]
        subset = df[df['dag']==dag]

        for metric in metrics:
            vals = subset.set_index("algorithm")[metric].to_dict()
            # find best (lowest) among baselines
            baseline_algos = [a for a in algos if a!="QERLO"]
            best_val = min(vals[a] for a in baseline_algos if a in vals)

            for algo in algos:
                if algo not in vals:
                    row.append("-")
                else:
                    val = vals[algo]
                    if algo=="QERLO":
                        # highlight QERLO row
                        row.append(f"\\cellcolor{{gray!20}} {val:.2f}")
                    elif np.isclose(val, best_val):
                        row.append(f"\\textbf{{{val:.2f}}}")
                    else:
                        row.append(f"{val:.2f}")
        table.append(row)

    # Build LaTeX code
    header = ["DAG"] + [f"{m}-{a}" for m in metrics for a in algos]
    col_format = "l" + "c"*(len(header)-1)

    latex = []
    latex.append("\\begin{table*}[ht]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\begin{tabular}{%s}" % col_format)
    latex.append("\\hline")
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\hline")
    for r in table:
        latex.append(" & ".join(r) + " \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Comparison of scheduling algorithms across DAGs. "
                 "Best baseline is bolded, QERLO highlighted in gray.}")
    latex.append("\\label{tab:results}")
    latex.append("\\end{table*}")

    return "\n".join(latex)

latex_code = make_latex_table(df, metrics)
with open(out_dir/"results_table.tex","w") as f:
    f.write(latex_code)

print(f"✅ LaTeX table saved at {out_dir/'results_table.tex'}")
