#!/usr/bin/env python3
# filepath: /home/azureuser/pytorch-distributed/compare_parallelism_results.py
import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def find_latest_benchmarks():
    """Find the latest benchmark results for each configuration."""
    results_dir = Path("benchmark_results")
    
    # Get all benchmark files
    benchmark_files = {
        "data_parallel": glob.glob(str(results_dir / "benchmark_nodes*.json")),
        "model_parallel": glob.glob(str(results_dir / "benchmark_model_parallel_*.json")),
    }
    
    # Group files by configuration
    data_parallel_configs = {}
    model_parallel_configs = {}
    
    # Process data parallel results
    for filepath in benchmark_files["data_parallel"]:
        filename = os.path.basename(filepath)
        # Extract configuration (e.g., nodes2_procs1)
        parts = filename.split("_")
        if len(parts) >= 2:
            config = "_".join(parts[1:3])  # e.g., nodes2_procs1
            
            # Get timestamp from filename
            timestamp_str = "_".join(parts[3:5]).replace(".json", "")
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Add to dictionary if it's newer or not there yet
                if config not in data_parallel_configs or timestamp > data_parallel_configs[config]["timestamp"]:
                    data_parallel_configs[config] = {
                        "filepath": filepath,
                        "timestamp": timestamp,
                        "config": config
                    }
            except:
                # Skip files with invalid timestamps
                pass
    
    # Process model parallel results
    for filepath in benchmark_files["model_parallel"]:
        filename = os.path.basename(filepath)
        # Extract configuration (e.g., nodes2_split1)
        parts = filename.split("_")
        if len(parts) >= 3:
            config = "_".join(parts[2:4])  # e.g., nodes2_split1
            
            # Get timestamp from filename
            timestamp_str = "_".join(parts[4:6]).replace(".json", "")
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Add to dictionary if it's newer or not there yet
                if config not in model_parallel_configs or timestamp > model_parallel_configs[config]["timestamp"]:
                    model_parallel_configs[config] = {
                        "filepath": filepath,
                        "timestamp": timestamp,
                        "config": config
                    }
            except:
                # Skip files with invalid timestamps
                pass
    
    return {
        "data_parallel": data_parallel_configs,
        "model_parallel": model_parallel_configs
    }

def load_benchmark_data(filepath):
    """Load benchmark data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_metrics(data):
    """Extract relevant metrics from benchmark data."""
    if not data:
        return {}
    
    metrics = {
        "total_time": data.get("total_time", 0),
        "avg_epoch_time": data.get("avg_epoch_time", 0),
        "avg_batch_time": data.get("avg_batch_time", 0),
        "final_loss": data.get("epoch_losses", [0])[-1] if data.get("epoch_losses") else 0,
        "world_size": data.get("node_info", {}).get("world_size", 1),
    }
    
    # Add model parallelism specific metrics
    if "avg_forward_time" in data:
        metrics.update({
            "avg_forward_time": data.get("avg_forward_time", 0),
            "avg_backward_time": data.get("avg_backward_time", 0),
            "avg_optimizer_step_time": data.get("avg_optimizer_step_time", 0),
            "split_size": data.get("model_parallel_config", {}).get("split_size", 1),
        })
    
    return metrics

def create_comparison_table(benchmark_data):
    """Create a comparison table between data parallelism and model parallelism."""
    rows = []
    
    # Process data parallel benchmarks
    for config, info in benchmark_data["data_parallel"].items():
        data = load_benchmark_data(info["filepath"])
        metrics = extract_metrics(data)
        
        row = {
            "Config": config,
            "Parallelism": "Data Parallel",
            "Total Time (s)": metrics.get("total_time", 0),
            "Avg Epoch Time (s)": metrics.get("avg_epoch_time", 0),
            "Avg Batch Time (s)": metrics.get("avg_batch_time", 0),
            "Final Loss": metrics.get("final_loss", 0),
            "World Size": metrics.get("world_size", 1),
            "Split Size": "N/A",
            "Forward Time (s)": "N/A",
            "Backward Time (s)": "N/A",
            "Optimizer Step Time (s)": "N/A",
        }
        rows.append(row)
    
    # Process model parallel benchmarks
    for config, info in benchmark_data["model_parallel"].items():
        data = load_benchmark_data(info["filepath"])
        metrics = extract_metrics(data)
        
        row = {
            "Config": config,
            "Parallelism": "Model Parallel",
            "Total Time (s)": metrics.get("total_time", 0),
            "Avg Epoch Time (s)": metrics.get("avg_epoch_time", 0),
            "Avg Batch Time (s)": metrics.get("avg_batch_time", 0),
            "Final Loss": metrics.get("final_loss", 0),
            "World Size": metrics.get("world_size", 1),
            "Split Size": metrics.get("split_size", "N/A"),
            "Forward Time (s)": metrics.get("avg_forward_time", "N/A"),
            "Backward Time (s)": metrics.get("avg_backward_time", "N/A"),
            "Optimizer Step Time (s)": metrics.get("avg_optimizer_step_time", "N/A"),
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df

def generate_charts(df, output_dir="benchmark_results"):
    """Generate charts comparing data and model parallelism."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract node count from configs
    df['Nodes'] = df['Config'].apply(lambda x: int(x.split('_')[0].replace('nodes', '')))
    
    # Create comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Total Time Comparison
    ax = axes[0, 0]
    data_parallel = df[df['Parallelism'] == 'Data Parallel']
    model_parallel = df[df['Parallelism'] == 'Model Parallel']
    
    x = np.arange(len(data_parallel['Nodes'].unique()))
    width = 0.35
    
    # Group by node count and compute means
    data_by_nodes = data_parallel.groupby('Nodes')['Total Time (s)'].mean()
    model_by_nodes = model_parallel.groupby('Nodes')['Total Time (s)'].mean()
    
    # Plot bars
    ax.bar(x - width/2, data_by_nodes, width, label='Data Parallel')
    ax.bar(x + width/2, model_by_nodes, width, label='Model Parallel')
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Training Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(data_by_nodes.index)
    ax.legend()
    
    # 2. Epoch Time Comparison
    ax = axes[0, 1]
    data_by_nodes = data_parallel.groupby('Nodes')['Avg Epoch Time (s)'].mean()
    model_by_nodes = model_parallel.groupby('Nodes')['Avg Epoch Time (s)'].mean()
    
    ax.bar(x - width/2, data_by_nodes, width, label='Data Parallel')
    ax.bar(x + width/2, model_by_nodes, width, label='Model Parallel')
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Average Epoch Time (s)')
    ax.set_title('Average Epoch Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(data_by_nodes.index)
    ax.legend()
    
    # 3. Batch Time Comparison
    ax = axes[1, 0]
    data_by_nodes = data_parallel.groupby('Nodes')['Avg Batch Time (s)'].mean()
    model_by_nodes = model_parallel.groupby('Nodes')['Avg Batch Time (s)'].mean()
    
    ax.bar(x - width/2, data_by_nodes, width, label='Data Parallel')
    ax.bar(x + width/2, model_by_nodes, width, label='Model Parallel')
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Average Batch Time (s)')
    ax.set_title('Average Batch Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(data_by_nodes.index)
    ax.legend()
    
    # 4. Final Loss Comparison
    ax = axes[1, 1]
    data_by_nodes = data_parallel.groupby('Nodes')['Final Loss'].mean()
    model_by_nodes = model_parallel.groupby('Nodes')['Final Loss'].mean()
    
    ax.bar(x - width/2, data_by_nodes, width, label='Data Parallel')
    ax.bar(x + width/2, model_by_nodes, width, label='Model Parallel')
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(data_by_nodes.index)
    ax.legend()
    
    plt.tight_layout()
    
    # Save chart
    chart_path = output_dir / f"parallelism_comparison_{timestamp}.png"
    plt.savefig(chart_path)
    print(f"Saved comparison chart to {chart_path}")
    
    # Create model parallelism specific charts
    if not model_parallel.empty and 'Forward Time (s)' in model_parallel.columns:
        # Model parallel timing breakdown
        plt.figure(figsize=(10, 6))
        
        # Group by node count
        model_timing = model_parallel.groupby('Nodes')[
            ['Forward Time (s)', 'Backward Time (s)', 'Optimizer Step Time (s)']
        ].mean()
        
        model_timing.plot(kind='bar', stacked=True)
        plt.title('Model Parallel Timing Breakdown')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        
        # Save chart
        timing_path = output_dir / f"model_parallel_timing_{timestamp}.png"
        plt.savefig(timing_path)
        print(f"Saved model parallel timing chart to {timing_path}")
    
    # Additional chart for scaling efficiency
    plt.figure(figsize=(10, 6))
    
    # Compute speedup relative to single node
    if 1 in data_by_nodes.index:
        data_single_node = data_by_nodes[1]
        data_speedup = data_single_node / data_by_nodes
        
        if 1 in model_by_nodes.index:
            model_single_node = model_by_nodes[1]
            model_speedup = model_single_node / model_by_nodes
            
            # Plot speedup
            plt.plot(data_speedup.index, data_speedup.values, 'o-', label='Data Parallel')
            plt.plot(model_speedup.index, model_speedup.values, 's-', label='Model Parallel')
            plt.plot([1, max(data_speedup.index)], [1, max(data_speedup.index)], 'k--', label='Ideal Scaling')
            
            plt.title('Scaling Efficiency')
            plt.xlabel('Number of Nodes')
            plt.ylabel('Speedup Factor')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save chart
            scaling_path = output_dir / f"scaling_efficiency_{timestamp}.png"
            plt.savefig(scaling_path)
            print(f"Saved scaling efficiency chart to {scaling_path}")
    
    return True

def main():
    # Find latest benchmark results
    benchmark_data = find_latest_benchmarks()
    
    # Check if we have benchmark data for both parallelism types
    if not benchmark_data["data_parallel"] or not benchmark_data["model_parallel"]:
        print("Insufficient benchmark data for comparison. Please run benchmarks first.")
        return
    
    # Create comparison table
    df = create_comparison_table(benchmark_data)
    
    # Save comparison table to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results/parallelism_comparison_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved comparison table to {output_file}")
    
    # Generate comparison charts
    generate_charts(df)
    
    # Print comparison summary
    print("\nComparison Summary:")
    print("===================")
    
    # Group by parallelism type and compute means
    summary = df.groupby('Parallelism').mean()
    
    # Print the summary
    print(summary[['Total Time (s)', 'Avg Epoch Time (s)', 'Avg Batch Time (s)', 'Final Loss']])
    
    # Find the best configuration for each parallelism type
    best_data_parallel = df[df['Parallelism'] == 'Data Parallel'].sort_values('Total Time (s)').iloc[0]
    best_model_parallel = df[df['Parallelism'] == 'Model Parallel'].sort_values('Total Time (s)').iloc[0]
    
    print("\nBest Data Parallel Configuration:")
    print(f"Config: {best_data_parallel['Config']}")
    print(f"Total Time: {best_data_parallel['Total Time (s)']:.2f}s")
    print(f"Avg Epoch Time: {best_data_parallel['Avg Epoch Time (s)']:.2f}s")
    print(f"Final Loss: {best_data_parallel['Final Loss']:.4f}")
    
    print("\nBest Model Parallel Configuration:")
    print(f"Config: {best_model_parallel['Config']}")
    print(f"Total Time: {best_model_parallel['Total Time (s)']:.2f}s")
    print(f"Avg Epoch Time: {best_model_parallel['Avg Epoch Time (s)']:.2f}s")
    print(f"Split Size: {best_model_parallel['Split Size']}")
    print(f"Final Loss: {best_model_parallel['Final Loss']:.4f}")
    
    # Calculate the speedup or slowdown between the best configurations
    speedup_factor = best_data_parallel['Total Time (s)'] / best_model_parallel['Total Time (s)']
    
    if speedup_factor > 1:
        print(f"\nModel parallelism is {speedup_factor:.2f}x faster than data parallelism for the best configurations.")
    else:
        print(f"\nData parallelism is {1/speedup_factor:.2f}x faster than model parallelism for the best configurations.")
    
    # Analysis and recommendations
    print("\nAnalysis and Recommendations:")
    print("============================")
    
    if speedup_factor > 1.1:
        print("Model parallelism is performing significantly better. This suggests:")
        print("- Your model is large enough to benefit from being split across nodes")
        print("- Communication overhead in data parallelism is higher than the pipeline overhead in model parallelism")
        print("- Your CPU-based instances are effectively utilizing the model-parallel approach")
        print("\nRecommendation: Continue using model parallelism with the optimal configuration.")
    elif speedup_factor < 0.9:
        print("Data parallelism is performing significantly better. This suggests:")
        print("- The model is small enough that the pipeline overhead of model parallelism outweighs its benefits")
        print("- Your batch size is large enough to efficiently utilize data parallelism")
        print("- The communication overhead for gradients is manageable with your current network configuration")
        print("\nRecommendation: Stick with data parallelism for this model and hardware configuration.")
    else:
        print("Both parallelism strategies are performing similarly. This suggests:")
        print("- Your model and hardware configuration can work well with either approach")
        print("- The choice may depend on other factors like memory usage or batch size requirements")
        print("\nRecommendation: For more complex models, consider model parallelism. For simpler models or larger batch sizes, consider data parallelism.")
    
    # Memory utilization notes
    print("\nMemory Utilization:")
    print("- Data Parallelism: Each node needs to hold the entire model in memory")
    print("- Model Parallelism: Each node only needs to hold part of the model, reducing memory requirements per node")
    print("  but potentially increasing communication overhead between model parts")
    
    print("\nReport complete!")

if __name__ == "__main__":
    main()
